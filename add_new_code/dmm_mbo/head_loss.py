"""Loss adapter around unmodified S2A heads and their original target builder.

No Hungarian matching, new box representation, or replacement detection loss.
Each branch assigns its own anchors/ROIs with its configured original assigner.
All branches use the same GT objects/ordering (the baseline IR annotations).
"""
from dataclasses import dataclass

import torch
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils import images_to_levels
from mmdet.structures.bbox import cat_boxes, get_box_tensor

from .balancing import instance_weights
from .dcfl_targets import dcfl_targets
from dmm_mbo.dcfl.dcfl_heads import DCFLS2AHead, DCFLS2ARefineHead


@dataclass
class PreparedLoss:
    head: object
    outputs: tuple
    anchors: list
    labels: list
    label_weights: list
    bbox_targets: list
    bbox_weights: list
    gt_ids: list
    avg_factor: float
    costs: torch.Tensor
    valid: torch.Tensor


@torch.no_grad()
def prepare_loss(head, outputs, gt_instances, img_metas, ignored=None,
                 rois=None, cost_class=1.0, cost_bbox=1.0):
    """Assign once, retain GT IDs, and average positive detection cost per GT.

    The adaptation cost is the configured unreduced classification + regression
    loss, evaluated on assigned positives. SmoothL1 includes all five encoded
    components, including the original le135 angle target. Averaging over the
    positive anchors avoids making GT cost depend on object/anchor count.
    """
    if head.loss_bbox_type != 'normal':
        raise ValueError('MBO baseline adapter requires normal rotated regression')
    if not head.use_sigmoid_cls:
        raise ValueError('MBO baseline adapter expects sigmoid classification')
    if rois is not None:
        head.bboxes_as_anchors = rois
    cls_scores, bbox_preds = outputs
    device = cls_scores[0].device
    featmap_sizes = [x.shape[-2:] for x in cls_scores]
    anchor_list, flag_list = head.get_anchors(featmap_sizes, img_metas, device)
    level_sizes = [len(x) for x in anchor_list[0]]
    flat_anchors = [cat_boxes(x) for x in anchor_list]
    offsets = [0]
    for gt in gt_instances:
        offsets.append(offsets[-1] + len(gt))
    if isinstance(head, (DCFLS2AHead, DCFLS2ARefineHead)):
        labels, label_weights, bbox_targets, bbox_weights, avg_factor, gt_ids = dcfl_targets(
            head, outputs, anchor_list, flag_list, gt_instances, img_metas, ignored)
    else:
        targets = head.get_targets(
            anchor_list, flag_list, gt_instances, img_metas,
            batch_gt_instances_ignore=ignored, return_sampling_results=True)
        labels, label_weights, bbox_targets, bbox_weights, avg_factor, samples = targets
        image_gt_ids = []
        for image, sample in enumerate(samples):
            # SamplingResult.pos_inds indexes inside anchors, not the full grid.
            inside = anchor_inside_flags(
                flat_anchors[image], torch.cat(flag_list[image]),
                img_metas[image]['img_shape'][:2], head.train_cfg['allowed_border'])
            original_inds = inside.nonzero(as_tuple=False).squeeze(1)
            ids = torch.full((len(flat_anchors[image]),), -1,
                             device=device, dtype=torch.long)
            ids[original_inds[sample.pos_inds]] = (
                sample.pos_assigned_gt_inds + offsets[image])
            image_gt_ids.append(ids)
        gt_ids = images_to_levels(image_gt_ids, level_sizes)
    anchors = images_to_levels(
        [get_box_tensor(x) for x in flat_anchors], level_sizes)
    num_gt = offsets[-1]
    cost_sum = cls_scores[0].new_zeros(num_gt, dtype=torch.float32)
    count = torch.zeros_like(cost_sum)
    for cls, box, anchor, label, target, ids in zip(
            cls_scores, bbox_preds, anchors, labels, bbox_targets, gt_ids):
        ids = ids.reshape(-1)
        pos = ids >= 0
        if not pos.any():
            continue
        cls = cls.detach().permute(0, 2, 3, 1).reshape(
            -1, head.cls_out_channels)[pos].float()
        box = box.detach().permute(0, 2, 3, 1).reshape(
            -1, head.bbox_coder.encode_size)[pos].float()
        target = target.reshape(-1, target.shape[-1])[pos].float()
        if head.reg_decoded_bbox:
            box = get_box_tensor(head.bbox_coder.decode(
                anchor.reshape(-1, anchor.shape[-1])[pos].float(), box))
        cls_cost = head.loss_cls(
            cls, label.reshape(-1)[pos], reduction_override='none')
        box_cost = head.loss_bbox(box, target, reduction_override='none')
        cls_cost = cls_cost.reshape(len(cls), -1).sum(dim=1)
        box_cost = box_cost.reshape(len(box), -1).sum(dim=1)
        cost = cost_class * cls_cost + cost_bbox * box_cost
        if not torch.isfinite(cost).all():
            raise FloatingPointError('Non-finite assigned detection cost')
        cost_sum.scatter_add_(0, ids[pos], cost)
        count.scatter_add_(0, ids[pos], torch.ones_like(cost))
    return PreparedLoss(
        head, outputs, anchors, labels, label_weights, bbox_targets,
        bbox_weights, gt_ids, avg_factor,
        cost_sum / count.clamp_min(1), count > 0)


def apply_loss(prepared, weights):
    """Use the original loss function and normalization, changing positives only."""
    head = prepared.head
    cls_losses, bbox_losses = [], []
    for level, (cls, box) in enumerate(zip(*prepared.outputs)):
        ids = prepared.gt_ids[level]
        pos = ids >= 0
        multiplier = torch.ones_like(prepared.label_weights[level])
        multiplier[pos] = weights[ids[pos]]
        cls_loss, bbox_loss = head.loss_by_feat_single(
            cls, box, prepared.anchors[level], prepared.labels[level],
            prepared.label_weights[level] * multiplier,
            prepared.bbox_targets[level],
            prepared.bbox_weights[level] * multiplier.unsqueeze(-1),
            avg_factor=prepared.avg_factor)
        cls_losses.append(cls_loss)
        bbox_losses.append(bbox_loss)
    return dict(loss_cls=cls_losses, loss_bbox=bbox_losses)


def balanced_stage_losses(prepared, temperature=1.0, enabled=True):
    """Balance fusion/RGB/IR at one S2A stage using common GT identity."""
    costs = torch.stack([p.costs for p in prepared])
    valid = torch.stack([p.valid for p in prepared])
    weights = instance_weights(costs, valid, temperature, enabled)
    losses = [apply_loss(p, w) for p, w in zip(prepared, weights)]
    return losses, dict(costs=costs, valid=valid, weights=weights)
