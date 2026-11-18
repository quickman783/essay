"""DMM baseline + RGB/IR auxiliary S2A branches, used only by loss()."""
from copy import deepcopy

from torch import nn
from mmdet.models.utils import unpack_gt_instances

from mmrotate.models.detectors.dmm_s2anet import virRefineSingleStageDetector
from mmrotate.registry import MODELS

from .head_loss import balanced_stage_losses, prepare_loss


@MODELS.register_module()
class MBODMMDetector(virRefineSingleStageDetector):
    """Keep the inherited tensor/predict paths strictly fusion-only.

    Auxiliary necks/heads have independent parameters. Both receive features
    before DCFM (RGB after the existing MTA, IR directly from its backbone).
    Set mbo_auxiliary=False when building a deployed model; its state_dict is
    then identical in structure to the baseline detector's state_dict.
    """

    def __init__(self, *args, mbo_cfg=None, mbo_auxiliary=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mbo_cfg = dict(
            enabled=True, temperature=1.0, aux_loss_weight=1.0,
            cost_class=1.0, cost_bbox=1.0)
        if mbo_cfg:
            unknown = set(mbo_cfg) - set(self.mbo_cfg)
            if unknown:
                raise ValueError('Unknown MBO options: ' + str(sorted(unknown)))
            self.mbo_cfg.update(mbo_cfg)
        if self.mbo_cfg['temperature'] <= 0:
            raise ValueError('MBO temperature must be positive')
        if any(self.mbo_cfg[k] < 0 for k in (
                'aux_loss_weight', 'cost_class', 'cost_bbox')):
            raise ValueError('MBO loss/cost coefficients must be nonnegative')
        if self.mbo_cfg['cost_class'] + self.mbo_cfg['cost_bbox'] <= 0:
            raise ValueError('At least one MBO cost coefficient must be positive')
        self.mbo_auxiliary = mbo_auxiliary
        if self.aux_rpn_task:
            raise ValueError('Use the DMM baseline without the legacy auxiliary RPN')
        if not hasattr(self, 'neck'):
            raise ValueError('MBO baseline requires the original FPN')
        if mbo_auxiliary:
            self.mbo_necks = nn.ModuleDict({
                name: deepcopy(self.neck) for name in ('rgb', 'ir')})
            self.mbo_init_heads = nn.ModuleDict({
                name: deepcopy(self.bbox_head_init) for name in ('rgb', 'ir')})
            self.mbo_refine_heads = nn.ModuleDict({
                name: deepcopy(self.bbox_head_refine) for name in ('rgb', 'ir')})

    def init_weights(self):
        super().init_weights()
        if self.mbo_auxiliary:
            # nn.ModuleDict does not implement MMEngine's init_weights().
            # Initialize its children explicitly, especially FocalLoss's
            # low-positive-prior classification bias in both S2A stages.
            for branch in ('rgb', 'ir'):
                self.mbo_necks[branch].init_weights()
                self.mbo_init_heads[branch].init_weights()
                for head in self.mbo_refine_heads[branch]:
                    head.init_weights()

    def _training_features(self, rgb, ir):
        rgb = self.backbone_vi(rgb)
        ir = self.backbone_ir(ir)
        if hasattr(self, 'mtablock'):
            rgb = self.mtablock(rgb)
        if hasattr(self, 'fusblock'):
            fused = self.fusblock(rgb, ir)
        else:
            fused = tuple(v + t for v, t in zip(rgb, ir))
        # Branch from these features, never from the DCFM output.
        return (self.neck(fused), self.mbo_necks['rgb'](rgb),
                self.mbo_necks['ir'](ir))

    def loss(self, batch_inputs, batch_inputs2, batch_data_samples):
        if not self.mbo_auxiliary:
            raise RuntimeError('Auxiliary-free model is for inference only')
        features = self._training_features(batch_inputs, batch_inputs2)
        gt, ignored, metas = unpack_gt_instances(batch_data_samples)
        # The baseline's primary gt_instances are IR annotations. Reuse the
        # identical objects in all branches: RGB/IR annotation lists are NOT
        # one-to-one and must never be zipped by their annotation index.
        names = ('fusion', 'rgb', 'ir')
        heads = (self.bbox_head_init, self.mbo_init_heads['rgb'],
                 self.mbo_init_heads['ir'])
        outputs = [head(x) for head, x in zip(heads, features)]
        losses = {}

        def supervise(stage_heads, stage_outputs, stage, rois=None, scale=1.0):
            prepared = [prepare_loss(
                head, out, gt, metas, ignored,
                rois=None if rois is None else rois[index],
                cost_class=self.mbo_cfg['cost_class'],
                cost_bbox=self.mbo_cfg['cost_bbox'])
                for index, (head, out) in enumerate(zip(stage_heads, stage_outputs))]
            parts, diagnostics = balanced_stage_losses(
                prepared, self.mbo_cfg['temperature'], self.mbo_cfg['enabled'])
            for index, (name, part) in enumerate(zip(names, parts)):
                branch_scale = scale * (1.0 if index == 0 else
                                        self.mbo_cfg['aux_loss_weight'])
                for key, value in part.items():
                    # Preserve the fusion loss names used in baseline logs.
                    suffix = stage if index == 0 else name + '_' + stage
                    losses[key + '_' + suffix] = [x * branch_scale for x in value]
                valid = diagnostics['valid'][index]
                w = diagnostics['weights'][index]
                c = diagnostics['costs'][index]
                # Monitoring keys deliberately do not contain "loss".
                losses['mbo_weight_' + name + '_' + stage] = (
                    w[valid].mean() if valid.any() else w.new_tensor(1.0))
                losses['mbo_cost_' + name + '_' + stage] = (
                    c[valid].mean() if valid.any() else c.new_tensor(0.0))
            losses['mbo_matched_gts_' + stage] = diagnostics['valid'].all(0).sum().float()

        supervise(heads, outputs, 'init')
        rois = [head.filter_bboxes(*out) for head, out in zip(heads, outputs)]
        for stage in range(self.num_refine_stages):
            heads = (self.bbox_head_refine[stage],
                     self.mbo_refine_heads['rgb'][stage],
                     self.mbo_refine_heads['ir'][stage])
            outputs = [head(head.feature_refine(x, roi))
                       for head, x, roi in zip(heads, features, rois)]
            supervise(heads, outputs, 'refine_' + str(stage), rois,
                      self.train_cfg.stage_loss_weights[stage])
            if stage + 1 < self.num_refine_stages:
                rois = [head.refine_bboxes(*out, rois=roi)
                        for head, out, roi in zip(heads, outputs, rois)]
        return losses

    def remove_auxiliary_branches(self):
        """Release training-only modules before deployment; leaves fusion intact."""
        if self.training:
            raise RuntimeError('Call eval() before removing auxiliary branches')
        for name in ('mbo_necks', 'mbo_init_heads', 'mbo_refine_heads'):
            if hasattr(self, name):
                delattr(self, name)
        self.mbo_auxiliary = False
        return self
