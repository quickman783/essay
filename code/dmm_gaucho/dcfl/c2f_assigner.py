"""DCFL: Dynamic Coarse-to-Fine 标签分配（CVPR 2023）。

移植自 ``mmrotate-dcfl-main/mmrotate/core/bbox/assigners/coarse2fine_assigner.py``，
算法逐行保留，只做两处适配：

1. 接口改成 mmrotate 1.0 / mmdet 3.x 的 ``assign(pred_instances, gt_instances,
   gt_instances_ignore)``；额外需要的分类分数与回归预测放在
   ``pred_instances.cls_scores`` / ``pred_instances.bbox_preds`` 里
   （形状分别是 (num_classes, N) 和 (5, N)）。
2. 原始仓库的 PCB 模块依赖 ``DeformConvG``（MMCV-DCFL 分叉才有），
   本实现不移植该模块 —— 分配器本身是纯 PyTorch，不需要任何自定义算子。

两个阶段：
    粗筛：用 gjsd（高斯 JS 散度）找每个 GT 的 topk 锚框，低于 0.8 的判为背景
    细选：用 "预测框与 GT 的 IoU + 分类置信度" 作为质量分，每个 GT 只留 topq 个，
          再用 dgmm 高斯掩码剔除离 GT 太远的候选
"""

import torch
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.structures.bbox import get_box_tensor

from mmrotate.registry import TASK_UTILS

from .gjsd_calculator import xy_wh_r_2_xy_sigma

__all__ = ['C2FAssigner']


@TASK_UTILS.register_module()
class C2FAssigner(BaseAssigner):
    """DCFL 的粗到细分配器。

    Args:
        topk (int): 粗筛时每个 GT 保留的锚框数。
        topq (int): 细选时每个 GT 最终保留的正样本数。
        assign_metric (str): 粗筛度量，``gjsd`` / ``iou``。
        constraint (str, optional): ``'dgmm'`` 时启用高斯距离掩码。
        gauss_thr (float): dgmm 掩码的距离阈值。
        gt_max_assign_all (bool): 是否让所有并列最大的锚框都成为正样本。
        ignore_iof_thr (float): IoF 忽略阈值，<=0 表示不忽略。
        ignore_wrt_candidates (bool): IoF 是锚框相对 GT 还是反过来算。
        gpu_assign_thr (int): GT 数超过该值时把分配搬到 CPU，<=0 表示不搬。
        iou_calculator (dict): 度量计算器配置。
        bbox_coder (dict): 解码网络输出用的编码器。
    """

    def __init__(self,
                 gt_max_assign_all: bool = True,
                 ignore_iof_thr: float = -1,
                 ignore_wrt_candidates: bool = True,
                 gpu_assign_thr: int = 512,
                 iou_calculator: dict = dict(type='RBboxMetrics2D'),
                 assign_metric: str = 'gjsd',
                 topk: int = 16,
                 topq: int = 12,
                 constraint: str = 'dgmm',
                 gauss_thr: float = 0.6,
                 bbox_coder: dict = dict(type='DeltaXYWHTRBBoxCoder'),
                 **kwargs):
        super().__init__()
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr

    def assign(self, pred_instances, gt_instances, gt_instances_ignore=None):
        """给每个锚框分配 GT 或者背景。

        Args:
            pred_instances (:obj:`InstanceData`): 需要包含
                ``priors`` (N,5)、``cls_scores`` (N,C)、``bbox_preds`` (N,5)。
                （分类/回归预测按 (N, ·) 存是为了满足 InstanceData 的长度约束，
                内部会转置成 (C,N) / (5,N) 使用。）
            gt_instances (:obj:`InstanceData`): 包含 ``bboxes`` / ``labels``。
            gt_instances_ignore (:obj:`InstanceData`, optional): 需要忽略的区域。

        Returns:
            :obj:`AssignResult`
        """
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
        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)

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

        assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            assigned_labels = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # 每个锚框对所有 GT 的最大相似度
        max_overlaps, _ = overlaps.max(dim=0)
        # 每个 GT 的 topk 锚框
        k = min(self.topk, num_bboxes)
        gt_max_overlaps, _ = overlaps.topk(k, dim=1, largest=True, sorted=True)

        # 相似度不够的一律当背景
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        # ---- 细选：用解码后的预测框质量重排 ----
        device = bboxes.device
        bbox_preds = bbox_preds.to(device)
        cls_scores = cls_scores.to(device)
        bbox_preds = torch.transpose(bbox_preds, 0, 1)
        bbox_preds = get_box_tensor(self.bbox_coder.decode(bboxes, bbox_preds))

        can_positive_mask = assigned_gt_inds > 0
        can_positive_inds = torch.nonzero(can_positive_mask)

        # 粗筛有可能一个正样本都没选出来（预测很离谱时 gjsd 全部 < 0.8）。
        # 此时细选无从谈起，直接返回纯背景的分配结果，避免下面
        # topk(0) / 空张量求均值产生 NaN 并沿梯度毁掉权重。
        if can_positive_inds.numel() == 0:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            return AssignResult(
                num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

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
                assigned_gt_inds_init[remap_inds] = assign_result_pre_gt[remap_inds]
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
                .matmul(inv_sigma_t).matmul(xy_a - xy_gt)).squeeze(-1).squeeze(-1)
            gaussian_pt = torch.exp(
                -0.5 * (xy_a - xy_pt).permute(0, 1, 3, 2)
                .matmul(inv_sigma_t).matmul(xy_a - xy_pt)).squeeze(-1).squeeze(-1)
            gaussian = 0.7 * gaussian_gt + 0.3 * gaussian_pt

            inside_flag = gaussian >= torch.exp(
                torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assigned_gt_inds.size(0))
            inside_mask = inside_flag[length, (assigned_gt_inds - 1).clamp(min=0)]
            assigned_gt_inds = assigned_gt_inds * inside_mask

        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        assign_result = AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)

        return assign_result
