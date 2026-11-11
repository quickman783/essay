"""DC2F 分配器：``C2FAssigner`` 的子类，通过三个开关改动 DCFL。

不改动 ``dmm_gaucho/dcfl/c2f_assigner.py`` —— 这里只覆写 ``assign()``，
三个开关全为 ``False`` 时，分配结果与 DCFL 逐位一致，方便做增量消融。

三处改动的含义见 ``dmm_dc2f/__init__.py``。
"""

import torch
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.structures.bbox import get_box_tensor

from mmrotate.registry import TASK_UTILS

from dmm_gaucho.dcfl.c2f_assigner import C2FAssigner
from dmm_gaucho.dcfl.gjsd_calculator import xy_wh_r_2_xy_sigma

from .prior import read_prior

__all__ = ['DC2FAssigner']


@TASK_UTILS.register_module()
class DC2FAssigner(C2FAssigner):
    """带 SNT / CSC / BIND 的粗到细分配器。

    除 ``C2FAssigner`` 原有的参数外，新增：

    Args:
        snt (bool): 开启按目标形状归一化的粗筛阈值。
        thr_ratio (float): 阈值上限，默认 0.8（与 DCFL 的固定阈值一致）。
        anchor_size (float): 锚框边长，S2A-Net 用 ``scales=[4]`` × stride 8
            得到 32，默认 32.0。
        csc (bool): 开启跨阶段先验（只救零正样本的目标）。
        csc_free_thr (float): 零正样本目标的 dgmm 阈值，默认 3.0（几乎不删）。
        bind_last (bool): 开启「编号最大者优先」的显式归属。
    """

    # 让 dc2f_heads 知道可以直接传 img_id
    supports_img_id = True

    def __init__(self,
                 snt: bool = False,
                 thr_ratio: float = 0.8,
                 anchor_size: float = 32.0,
                 csc: bool = False,
                 csc_free_thr: float = 3.0,
                 bind_last: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.snt = snt
        self.thr_ratio = thr_ratio
        self.anchor_size = anchor_size
        self.csc = csc
        self.csc_free_thr = csc_free_thr
        self.bind_last = bind_last
        self._cur_img_id = None

    # ------------------------------------------------------------------ SNT
    def sim_upper(self, gt_bboxes):
        """每个 GT 在当前锚框尺寸下能达到的 gjsd 相似度上界，返回 (G,)。

        推导：锚框（边长 a 的正方形）与 GT（w×h）都看成二维高斯，
        两者协方差对齐时 JS 散度最小，此时
            m1 = 1 / (2/a² + 2/w²)，m2 = 1 / (2/a² + 2/h²)
            jsd_min = 0.5 * log( (a²·w·h/16) / (m1·m2) )
        相似度 = 1 / (1 + jsd)，与 ``gjsd_calculator`` 的口径一致。

        a=32 时的自检值：32×32 → 1.000，20×20 → 0.904，9×20 → 0.725。
        """
        a2 = float(self.anchor_size) ** 2
        w = gt_bboxes[:, 2].clamp(min=1e-3)
        h = gt_bboxes[:, 3].clamp(min=1e-3)
        m1 = 1.0 / (2.0 / a2 + 2.0 / w.square())
        m2 = 1.0 / (2.0 / a2 + 2.0 / h.square())
        jsd = 0.5 * ((a2 * w * h / 16.0) / (m1 * m2)).log().clamp(min=0)
        return 1.0 / (1.0 + jsd)

    # --------------------------------------------------------------- assign
    def assign(self, pred_instances, gt_instances, gt_instances_ignore=None,
               img_id=None):
        """给每个锚框分配 GT 或者背景。

        比 ``C2FAssigner.assign`` 多两个行为：
        * 接受 ``img_id``，用于 CSC 的按图先验读写；
        * 三个开关按开启状态改变粗筛阈值、归属规则、dgmm 阈值。
        """
        self._cur_img_id = img_id

        bboxes = get_box_tensor(pred_instances.priors)
        # 存进来的是 (N, C) / (N, 5)（为了满足 InstanceData 的长度约束），
        # 这里转成算法需要的 (C, N) / (5, N)
        cls_scores = pred_instances.cls_scores.transpose(0, 1)
        bbox_preds = pred_instances.bbox_preds.transpose(0, 1)
        gt_bboxes = get_box_tensor(gt_instances.bboxes)
        gt_labels = gt_instances.labels

        assign_on_cpu = (self.gpu_assign_thr >= 0
                         and gt_bboxes.shape[0] > self.gpu_assign_thr)
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_instances_ignore is not None:
                gt_instances_ignore.bboxes = gt_instances_ignore.bboxes.cpu()

        # ---- 粗筛：用 gjsd 找候选 ----
        overlaps = self.iou_calculator(gt_bboxes, bboxes,
                                       mode=self.assign_metric)

        if (self.ignore_iof_thr > 0 and gt_instances_ignore is not None
                and gt_instances_ignore.bboxes.numel() > 0
                and bboxes.numel() > 0):
            ignore_boxes = get_box_tensor(gt_instances_ignore.bboxes)
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, ignore_boxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    ignore_boxes, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            assigned_labels = overlaps.new_full((num_bboxes, ),
                                                -1, dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, max_overlaps,
                labels=assigned_labels)

        # 每个锚框对所有 GT 的最大相似度
        max_overlaps, _ = overlaps.max(dim=0)
        # 每个 GT 的 topk 锚框
        k = min(self.topk, num_bboxes)
        gt_max_overlaps, _ = overlaps.topk(k, dim=1, largest=True, sorted=True)

        # ===== DC2F / SNT：阈值按目标形状归一化 =====
        if self.snt:
            # 上界本来就不低于 thr_ratio 的目标，阈值仍是 thr_ratio，
            # 行为与 DCFL 完全一致；只有上界更低的目标阈值才降低。
            thr = self.sim_upper(gt_bboxes).clamp(max=self.thr_ratio)
            pass_thr = (overlaps >= thr[:, None]).any(dim=0)      # (N,)
            assigned_gt_inds[(max_overlaps >= 0) & (~pass_thr)] = 0
        else:
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0

        # ===== DC2F / BIND：把「编号最大者优先」写显式 =====
        if self.bind_last:
            kth = gt_max_overlaps[:, -1]                            # (G,) 第 k 名
            in_topk = overlaps >= kth[:, None]                      # (G, N)
            gi = torch.arange(num_gts, device=overlaps.device)[
                :, None].expand_as(in_topk)
            last = torch.where(in_topk, gi, -1).max(dim=0).values  # (N,)
            sel = last >= 0
            assigned_gt_inds[sel] = last[sel] + 1
        else:
            for i in range(num_gts):
                for j in range(k):
                    max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                    assigned_gt_inds[max_overlap_inds] = i + 1

        # ---- 细选：用解码后的预测框质量重排 ----
        device = bboxes.device
        bbox_preds = bbox_preds.to(device)
        cls_scores = cls_scores.to(device)
        bbox_preds = torch.transpose(bbox_preds, 0, 1)
        bbox_preds = get_box_tensor(
            self.bbox_coder.decode(bboxes, bbox_preds))

        can_positive_mask = assigned_gt_inds > 0
        can_positive_inds = torch.nonzero(can_positive_mask)

        # 粗筛有可能一个正样本都没选出来（预测很离谱时相似度全部过低）。
        # 此时细选无从谈起，直接返回纯背景的分配结果，避免下面
        # topk(0) / 空张量求均值产生 NaN 并沿梯度毁掉权重。
        if can_positive_inds.numel() == 0:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ),
                                                        -1, dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, max_overlaps,
                labels=assigned_labels)

        poscan = assigned_gt_inds[can_positive_inds].squeeze(-1)
        can_other_mask = assigned_gt_inds <= 0

        can_pos_scores = cls_scores[:, can_positive_inds].squeeze(-1)
        can_pos_scores = torch.transpose(can_pos_scores, 0, 1)
        can_bbox_pred = bbox_preds[can_positive_inds, :].squeeze(-1)

        can_pos_iou = self.iou_calculator(
            gt_bboxes.to(device), can_bbox_pred, mode='iou')
        can_pos_iou = can_pos_iou[poscan - 1, range(poscan.size(0))]
        can_pos_cls, _ = torch.max(can_pos_scores, 1)

        can_pos_quality = can_pos_iou + can_pos_cls.sigmoid()
        can_pos_quality = can_pos_quality.unsqueeze(0).repeat(num_gts, 1)

        gt_poscan = torch.zeros_like(can_pos_quality) - 100
        gt_poscan[poscan - 1, range(poscan.size(0))] = can_pos_quality[
            poscan - 1, range(poscan.size(0))]

        topq = min(self.topq, can_pos_quality.size(1))
        _, gt_argmax_quality = gt_poscan.topk(
            topq, dim=1, largest=True, sorted=True)

        assign_result_pre_gt = assigned_gt_inds
        assigned_gt_inds_init = assign_result_pre_gt * can_other_mask
        assigned_pos_prior = torch.zeros((num_gts, topq, 5), device=device)

        for i in range(num_gts):
            for j in range(topq):
                index = gt_argmax_quality[i, j]
                remap_inds = can_positive_inds[index, 0]
                assigned_gt_inds_init[remap_inds] = \
                    assign_result_pre_gt[remap_inds]
                assigned_pos_prior[i, j, :] = bboxes[remap_inds, :]

        assigned_gt_inds = assigned_gt_inds_init

        # ---- dgmm：用高斯距离掩掉离 GT 太远的候选 ----
        if self.constraint == 'dgmm' and assigned_pos_prior.size(1) > 0:
            device1 = gt_bboxes.device
            xy_gt, sigma_t = xy_wh_r_2_xy_sigma(gt_bboxes)
            pos_prior_mean = torch.mean(assigned_pos_prior[..., :2], dim=-2)
            xy_pt = pos_prior_mean
            xy_a = bboxes[..., :2]
            xy_gt = xy_gt[..., None, :, :2].unsqueeze(-1)
            xy_pt = xy_pt[..., None, :, :2].unsqueeze(-1)
            xy_a = xy_a[..., :, None, :2].unsqueeze(-1)

            inv_sigma_t = torch.stack(
                (sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
                 -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
                dim=-1).reshape(-1, 2, 2)
            det = sigma_t.det().unsqueeze(-1).unsqueeze(-1)
            inv_sigma_t = inv_sigma_t / det

            gaussian_gt = torch.exp(
                -0.5 * (xy_a - xy_gt).permute(0, 1, 3, 2)
                .matmul(inv_sigma_t).matmul(xy_a - xy_gt)
            ).squeeze(-1).squeeze(-1)
            gaussian_pt = torch.exp(
                -0.5 * (xy_a - xy_pt).permute(0, 1, 3, 2)
                .matmul(inv_sigma_t).matmul(xy_a - xy_pt)
            ).squeeze(-1).squeeze(-1)
            gaussian = 0.7 * gaussian_gt + 0.3 * gaussian_pt

            # ===== DC2F / CSC：只救「初始阶段零正样本」的目标 =====
            if self.csc:
                thr_vec = torch.full((num_gts, ), float(self.gauss_thr),
                                     device=device1, dtype=torch.float32)
                prior = read_prior(self._cur_img_id)
                if prior is not None and len(prior) == num_gts:
                    zero = torch.tensor([n == 0 for n in prior],
                                        device=device1, dtype=torch.bool)
                    thr_vec = torch.where(
                        zero,
                        torch.full_like(thr_vec, float(self.csc_free_thr)),
                        thr_vec)
                exp_thr = torch.exp(-thr_vec)
                # gaussian 是 (N, G)：阈值按最后一维广播。
                # 万一布局反转成 (G, N)，就改成按第一维广播。
                if gaussian.shape[-1] == exp_thr.numel():
                    exp_thr = exp_thr.unsqueeze(0)
                else:
                    exp_thr = exp_thr.unsqueeze(1)
                inside_flag = gaussian >= exp_thr
            else:
                inside_flag = gaussian >= torch.exp(
                    torch.tensor([-self.gauss_thr])).to(device1)

            length = range(assigned_gt_inds.size(0))
            inside_mask = inside_flag[length,
                                      (assigned_gt_inds - 1).clamp(min=0)]
            assigned_gt_inds = assigned_gt_inds * inside_mask

        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ),
                                                    -1, dtype=torch.long)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1]

        assign_result = AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)

        return assign_result
