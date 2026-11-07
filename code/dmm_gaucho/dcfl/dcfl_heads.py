"""把 DCFL 的分配器接到 S2A-Net 的两个检测头上。

mmdet 3.x 的 ``AnchorHead`` 是**静态**分配：``_get_targets_single`` 只拿锚框和 GT
去调 ``assigner.assign``，拿不到分类分数和回归预测，而 DCFL 的细选阶段必须用
这两样东西。所以这里重写 ``loss_by_feat``：自己按图按层算出动态分配结果，
再交给父类的 ``loss_by_feat_single`` 算损失（那部分是 GauCho 的高斯回归）。

除标签分配外，其余流程（锚框、采样器、损失、推理）与基线完全一致。
"""

from typing import List, Optional

import torch
from mmengine.structures import InstanceData
from mmdet.models.utils import images_to_levels, multi_apply
from mmdet.models.task_modules.prior_generators.utils import anchor_inside_flags
from mmdet.structures.bbox import cat_boxes, get_box_tensor
from mmdet.utils import InstanceList, OptInstanceList

from mmrotate.registry import MODELS

from ..gaucho_heads import GauchoS2AHead, GauchoS2ARefineHead

__all__ = ['GauchoDCFLS2AHead', 'GauchoDCFLS2ARefineHead']


def _flat_scores_preds(cls_scores, bbox_preds, img_id, inside_flags):
    """把一张图的各级分类分数与回归预测拼成 (C, N) / (5, N)，并只留有效锚框。"""
    scores = torch.cat([
        cls_scores[l][img_id].reshape(cls_scores[l].size(1), -1)
        for l in range(len(cls_scores))
    ], dim=1)
    preds = torch.cat([
        bbox_preds[l][img_id].reshape(bbox_preds[l].size(1), -1)
        for l in range(len(bbox_preds))
    ], dim=1)
    return scores[:, inside_flags], preds[:, inside_flags]


def dcfl_loss_by_feat(head, cls_scores, bbox_preds, batch_gt_instances,
                      batch_img_metas, batch_gt_instances_ignore=None):
    """DCFL 版的多图损失计算，替换 ``AnchorHead.loss_by_feat``。

    与父类实现有两点不同：
    1. 分配时额外把分类分数和回归预测送进 assigner（DCFL 细选需要）；
    2. ``avg_factor`` 至少取 1，避免整批没有正样本时 ``0 / 0`` 出 NaN。
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

    all_labels, all_label_weights = [], []
    all_bbox_targets, all_bbox_weights = [], []
    avg_factor = 0

    for img_id in range(num_imgs):
        # 保留锚框原来的类型（这里是 RotatedBoxes，不是裸 Tensor）：
        # 基线 DeltaXYWHTRBBoxCoder.encode 里的 bbox2delta 直接取
        # ``proposals.tensor``，传 Tensor 进去会抛
        # AttributeError: 'Tensor' object has no attribute 'tensor'。
        # GauCho 的 coder 自己会 get_box_tensor，两种都能吃。
        flat_anchors = cat_boxes(anchor_list[img_id])
        # 标量目标张量统一从这里生成，避免依赖 BaseBoxes.new_* 的返回类型
        flat_tensor = get_box_tensor(flat_anchors)
        valid_flags = torch.cat(valid_flag_list[img_id])

        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, batch_img_metas[img_id]['img_shape'][:2],
            head.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                '这张图没有任何落在图像内的有效锚框，检查 img_shape 或把 '
                'allowed_border 设成 -1。')

        # anchors 给 coder 用（保持 Boxes 类型），template 给 target 用（Tensor）
        anchors = flat_anchors[inside_flags]
        template = flat_tensor[inside_flags]
        scores, preds = _flat_scores_preds(
            cls_scores, bbox_preds, img_id, inside_flags)

        gt_instances = batch_gt_instances[img_id]
        # InstanceData 要求所有字段第一维一致，所以这里存 (N, C) / (N, 5)，
        # 分配器内部再转置回 (C, N) / (5, N)
        pred_instances = InstanceData(
            priors=anchors,
            cls_scores=scores.transpose(0, 1),
            bbox_preds=preds.transpose(0, 1))
        assign_result = head.assigner.assign(
            pred_instances, gt_instances, batch_gt_instances_ignore[img_id])
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

        # 映射回该图的全部锚框（含图像外的）
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

    num_level_anchors = [a.size(0) for a in anchor_list[0]]
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


def _unmap_tensor(data, count, inds, fill=0):
    """把有效锚框上的 target 放回全部锚框的位置。"""
    shape = (count, ) + tuple(data.size()[1:])
    ret = data.new_full(shape, fill)
    if data.size(0) > 0:
        ret[inds] = data
    return ret


@MODELS.register_module()
class GauchoDCFLS2AHead(GauchoS2AHead):
    """S2A-Net 第一阶段 + GauCho 回归 + DCFL 标签分配。"""

    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        return dcfl_loss_by_feat(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore)


@MODELS.register_module()
class GauchoDCFLS2ARefineHead(GauchoS2ARefineHead):
    """S2A-Net 第二阶段（AlignConv 精修）+ GauCho 回归 + DCFL 标签分配。"""

    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     rois: Optional[List[torch.Tensor]] = None) -> dict:
        # 精修阶段把上一阶段输出的框当锚框，沿用原实现的约定
        assert rois is not None
        self.bboxes_as_anchors = rois
        return dcfl_loss_by_feat(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore)
