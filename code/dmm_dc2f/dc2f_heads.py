"""DC2F 的损失入口：在 DCFL 的分配流程上接线 CSC 的跨阶段先验。

``dcfl_loss_by_feat`` 本身没有 ``stage`` 概念（它不知道自己是初始阶段还是
精修阶段），CSC 需要这个信息才能决定「写先验」还是「读先验」。
所以这里复制一份流程并加上：本文件不修改 ``dmm_gaucho/dcfl/dcfl_heads.py``。

两个 head 类沿用基线 S2A-Net 的回归分支（DeltaXYWHTRBBoxCoder + SmoothL1），
DC2F 只替换标签分配，不引入高斯回归：

    DC2FS2AHead        S2A-Net 第一阶段（ODM）+ DC2F 分配
    DC2FS2ARefineHead  S2A-Net 第二阶段（AlignConv 精修）+ DC2F 分配
"""

from typing import List, Optional

import torch
from mmengine.structures import InstanceData
from mmdet.models.task_modules.prior_generators.utils import anchor_inside_flags
from mmdet.models.utils import images_to_levels, multi_apply
from mmdet.structures.bbox import cat_boxes, get_box_tensor
from mmdet.utils import InstanceList, OptInstanceList

from mmrotate.models.dense_heads.s2a_head import S2AHead, S2ARefineHead
from mmrotate.registry import MODELS

from dmm_gaucho.dcfl.dcfl_heads import _flat_scores_preds, _unmap_tensor

from .prior import write_prior

__all__ = [
    'dc2f_loss_by_feat',
    'DC2FS2AHead', 'DC2FS2ARefineHead',
]


def dc2f_loss_by_feat(head, cls_scores, bbox_preds, batch_gt_instances,
                      batch_img_metas, batch_gt_instances_ignore=None,
                      stage: str = None):
    """DC2F 版的多图损失计算。

    与 ``dcfl_loss_by_feat`` 的差别只有两处：
    1. 调用 ``assigner.assign`` 时多传 ``img_id``（CSC 按图存取先验）；
    2. ``stage == 'init'`` 且分配器开了 ``csc`` 时，把每个 GT 的正样本数
       写进先验缓冲，供精修阶段读取。

    其余（锚框类型、target 生成、unmap、avg_factor 下限）与 DCFL 完全一致。

    Args:
        stage (str, optional): ``'init'`` 或 ``'refine'``。
    """
    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    assert len(featmap_sizes) == head.prior_generator.num_levels
    device = cls_scores[0].device
    num_imgs = len(batch_img_metas)

    anchor_list, valid_flag_list = head.get_anchors(
        featmap_sizes, batch_img_metas, device=device)
    num_level_anchors = [a.size(0) for a in anchor_list[0]]

    if batch_gt_instances_ignore is None:
        batch_gt_instances_ignore = [None] * num_imgs

    assigner = head.assigner
    accepts_img_id = getattr(assigner, 'supports_img_id', False)
    csc_on = getattr(assigner, 'csc', False)

    all_labels, all_label_weights = [], []
    all_bbox_targets, all_bbox_weights = [], []
    avg_factor = 0

    for img_id in range(num_imgs):
        # 保留锚框原来的类型（RotatedBoxes）：基线 DeltaXYWHTRBBoxCoder.encode
        # 直接取 ``proposals.tensor``，传裸 Tensor 会抛 AttributeError。
        flat_anchors = cat_boxes(anchor_list[img_id])
        flat_tensor = get_box_tensor(flat_anchors)
        valid_flags = torch.cat(valid_flag_list[img_id])

        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags,
            batch_img_metas[img_id]['img_shape'][:2],
            head.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                '这张图没有任何落在图像内的有效锚框，检查 img_shape 或把 '
                'allowed_border 设成 -1。')

        anchors = flat_anchors[inside_flags]
        template = flat_tensor[inside_flags]
        scores, preds = _flat_scores_preds(
            cls_scores, bbox_preds, img_id, inside_flags)

        gt_instances = batch_gt_instances[img_id]
        # InstanceData 要求各字段第一维一致，所以按 (N, ·) 存
        pred_instances = InstanceData(
            priors=anchors,
            cls_scores=scores.transpose(0, 1),
            bbox_preds=preds.transpose(0, 1))

        if accepts_img_id:
            assign_result = assigner.assign(
                pred_instances, gt_instances,
                batch_gt_instances_ignore[img_id], img_id=img_id)
        else:
            assign_result = assigner.assign(
                pred_instances, gt_instances,
                batch_gt_instances_ignore[img_id])

        # ===== DC2F / CSC：初始阶段把「每个 GT 的正样本数」传给精修阶段 =====
        if stage == 'init' and csc_on:
            inds = assign_result.gt_inds
            write_prior(img_id, [
                int((inds == gi + 1).sum())
                for gi in range(len(gt_instances.bboxes))
            ])

        sampling_result = head.sampler.sample(
            assign_result, pred_instances, gt_instances)
        avg_factor += sampling_result.avg_factor

        # ---- 以下与 mmdet AnchorHead._get_targets_single 完全一致 ----
        num_valid_anchors = template.shape[0]
        target_dim = (gt_instances.bboxes.size(-1) if head.reg_decoded_bbox
                      else head.bbox_coder.encode_size)
        bbox_targets = template.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = template.new_zeros(num_valid_anchors, target_dim)
        labels = template.new_full(
            (num_valid_anchors, ), head.num_classes, dtype=torch.long)
        label_weights = template.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not head.reg_decoded_bbox:
                pos_bbox_targets = head.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = get_box_tensor(
                    sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            labels[pos_inds] = sampling_result.pos_gt_labels
            if head.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = head.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        num_total_anchors = flat_tensor.size(0)
        labels = _unmap_tensor(
            labels, num_total_anchors, inside_flags, head.num_classes)
        label_weights = _unmap_tensor(
            label_weights, num_total_anchors, inside_flags, 0.0)
        bbox_targets = _unmap_tensor(
            bbox_targets, num_total_anchors, inside_flags, 0.0)
        bbox_weights = _unmap_tensor(
            bbox_weights, num_total_anchors, inside_flags, 0.0)

        all_labels.append(labels)
        all_label_weights.append(label_weights)
        all_bbox_targets.append(bbox_targets)
        all_bbox_weights.append(bbox_weights)

    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)

    all_anchor_list = images_to_levels(
        [cat_boxes(a) for a in anchor_list], num_level_anchors)

    losses_cls, losses_bbox = multi_apply(
        head.loss_by_feat_single,
        cls_scores,
        bbox_preds,
        all_anchor_list,
        labels_list,
        label_weights_list,
        bbox_targets_list,
        bbox_weights_list,
        avg_factor=max(float(avg_factor), 1.0))
    return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)


@MODELS.register_module()
class DC2FS2AHead(S2AHead):
    """S2A-Net 第一阶段（ODM）+ DC2F 标签分配。

    回归分支与基线完全一致（``DeltaXYWHTRBBoxCoder`` + ``SmoothL1Loss``），
    只有标签分配从 ``MaxIoUAssigner`` 换成 ``DC2FAssigner``。
    """

    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        return dc2f_loss_by_feat(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore, stage='init')


@MODELS.register_module()
class DC2FS2ARefineHead(S2ARefineHead):
    """S2A-Net 第二阶段（AlignConv 精修）+ DC2F 标签分配。"""

    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     rois: Optional[List[torch.Tensor]] = None) -> dict:
        assert rois is not None
        self.bboxes_as_anchors = rois
        return dc2f_loss_by_feat(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore, stage='refine')

