"""Expose GT identity alongside the existing DMM+DCFL targets.

This follows dmm_mbo.dcfl.loss_helpers.dcfl_loss_by_feat's target
construction, using its unchanged helpers, C2FAssigner, sampler and box coder.
The additional output is a global GT index for each positive anchor.
"""
import torch
from mmengine.structures import InstanceData
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils import images_to_levels
from mmdet.structures.bbox import cat_boxes, get_box_tensor

from dmm_mbo.dcfl.loss_helpers import _flat_scores_preds, _unmap_tensor


@torch.no_grad()
def dcfl_targets(head, outputs, anchor_list, flag_list, gt_instances,
                 img_metas, ignored=None):
    num_images = len(img_metas)
    ignored = ignored if ignored is not None else [None] * num_images
    level_sizes = [len(a) for a in anchor_list[0]]
    all_labels, all_lw, all_targets, all_bw, all_ids = [], [], [], [], []
    avg_factor, offset = 0, 0
    for image in range(num_images):
        flat = cat_boxes(anchor_list[image])
        flat_tensor = get_box_tensor(flat)
        inside = anchor_inside_flags(
            flat, torch.cat(flag_list[image]), img_metas[image]['img_shape'][:2],
            head.train_cfg['allowed_border'])
        if not inside.any():
            raise ValueError('No valid DCFL anchor inside the image')
        anchors = flat[inside]
        template = flat_tensor[inside]
        scores, preds = _flat_scores_preds(*outputs, image, inside)
        predictions = InstanceData(priors=anchors, cls_scores=scores.transpose(0, 1),
                                   bbox_preds=preds.transpose(0, 1))
        gt = gt_instances[image]
        assignment = head.assigner.assign(predictions, gt, ignored[image])
        sample = head.sampler.sample(assignment, predictions, gt)
        avg_factor += sample.avg_factor
        dim = gt.bboxes.size(-1) if head.reg_decoded_bbox else head.bbox_coder.encode_size
        targets = template.new_zeros(len(template), dim)
        bw = template.new_zeros(len(template), dim)
        labels = template.new_full((len(template),), head.num_classes, dtype=torch.long)
        lw = template.new_zeros(len(template), dtype=torch.float)
        ids = template.new_full((len(template),), -1, dtype=torch.long)
        pos, neg = sample.pos_inds, sample.neg_inds
        if len(pos):
            targets[pos] = (get_box_tensor(sample.pos_gt_bboxes) if head.reg_decoded_bbox
                            else head.bbox_coder.encode(sample.pos_priors, sample.pos_gt_bboxes))
            bw[pos] = 1.0
            labels[pos] = sample.pos_gt_labels
            lw[pos] = 1.0 if head.train_cfg['pos_weight'] <= 0 else head.train_cfg['pos_weight']
            ids[pos] = sample.pos_assigned_gt_inds + offset
        lw[neg] = 1.0
        count = len(flat_tensor)
        all_labels.append(_unmap_tensor(labels, count, inside, head.num_classes))
        all_lw.append(_unmap_tensor(lw, count, inside, 0.0))
        all_targets.append(_unmap_tensor(targets, count, inside, 0.0))
        all_bw.append(_unmap_tensor(bw, count, inside, 0.0))
        all_ids.append(_unmap_tensor(ids, count, inside, -1))
        offset += len(gt)
    return (*(images_to_levels(x, level_sizes) for x in (
        all_labels, all_lw, all_targets, all_bw)), max(float(avg_factor), 1.0),
        images_to_levels(all_ids, level_sizes))
