"""DMM baseline + RGB/IR auxiliary S2A branches, used only by loss()."""
from copy import deepcopy

import torch
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

    def __init__(self, *args, mbo_cfg=None, mbo_auxiliary=True,
                 mbo_fork='pre', mbo_refine_fusion=False,
                 mbo_gamma_init=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mbo_cfg = dict(
            enabled=True, temperature=1.0, aux_loss_weight=1.0,
            cost_class=1.0, cost_bbox=1.0,
            refine_aux_scale=1.0,
            density_gate=False, density_ref=12.0, density_power=2.0,
            density_value=0.0)
        if mbo_cfg:
            unknown = set(mbo_cfg) - set(self.mbo_cfg)
            if unknown:
                raise ValueError('Unknown MBO options: ' + str(sorted(unknown)))
            self.mbo_cfg.update(mbo_cfg)
        if self.mbo_cfg['temperature'] <= 0:
            raise ValueError('MBO temperature must be positive')
        if any(self.mbo_cfg[k] < 0 for k in (
                'aux_loss_weight', 'cost_class', 'cost_bbox',
                'refine_aux_scale')):
            raise ValueError('MBO loss/cost coefficients must be nonnegative')
        if self.mbo_cfg['cost_class'] + self.mbo_cfg['cost_bbox'] <= 0:
            raise ValueError('At least one MBO cost coefficient must be positive')
        if self.mbo_cfg['density_gate']:
            if self.mbo_cfg['density_ref'] <= 0:
                raise ValueError('MBO density_ref must be positive')
            if self.mbo_cfg['density_power'] <= 0:
                raise ValueError('MBO density_power must be positive')
            if self.mbo_cfg['density_value'] <= 0:
                raise ValueError(
                    'mbo_cfg.density_gate needs a positive density_value: '
                    'the training set mean instances per image')
            self.mbo_gate = min(
                1.0,
                (self.mbo_cfg['density_value'] / self.mbo_cfg['density_ref'])
                ** self.mbo_cfg['density_power'])
        else:
            self.mbo_gate = 1.0
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

        if mbo_fork not in ('pre', 'post'):
            raise ValueError("mbo_fork must be 'pre' or 'post'")
        self.mbo_fork = mbo_fork
        self.mbo_refine_fusion = bool(mbo_refine_fusion)

        if self.mbo_refine_fusion:
            if not self.mbo_auxiliary:
                raise ValueError('mbo_refine_fusion requires mbo_auxiliary=True')
            if self.mbo_fork == 'post':
                raise ValueError("mbo_refine_fusion needs mbo_fork='pre'")
            # feat_refine_module 是 AlignConv 的单实例，直接从已有 head 里
            # deepcopy，别从 frm_cfg 重新 build（配置可能已不在 kwargs 里）。
            self.refine_align = nn.ModuleDict({
                name: nn.ModuleList([
                    deepcopy(head.feat_refine_module)
                    for head in self.bbox_head_refine])
                for name in ('rgb', 'ir')})
            # gamma 初值 0：初始化瞬间 aux 项为 0，前向与原路径逐位相同
            self.refine_gamma = nn.ParameterList([
                nn.Parameter(torch.tensor(float(mbo_gamma_init)))
                for _ in range(self.num_refine_stages)])

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
        if self.mbo_refine_fusion:
            # AlignConv 藏在 ModuleDict / ModuleList 里，父类遍历不到
            for stage in range(self.num_refine_stages):
                for name in ('rgb', 'ir'):
                    self.refine_align[name][stage].init_weights()

    def _training_features(self, rgb, ir):
        rgb = self.backbone_vi(rgb)
        ir = self.backbone_ir(ir)
        if hasattr(self, 'mtablock'):
            rgb = self.mtablock(rgb)
        if hasattr(self, 'fusblock'):
            fused = self.fusblock(rgb, ir)
        else:
            fused = tuple(v + t for v, t in zip(rgb, ir))
        if self.mbo_fork == 'post':
            # 消融：两条辅助支路都吃融合特征（neck 仍各自独立，参数量与 pre 一致）
            branch_rgb = branch_ir = fused
        else:
            branch_rgb, branch_ir = rgb, ir
        return (self.neck(fused), self.mbo_necks['rgb'](branch_rgb),
                self.mbo_necks['ir'](branch_ir))

    def _refine_features(self, stage, features, rois):
        """AlignConv(fused) + gamma * [AlignConv(rgb) + AlignConv(ir)]."""
        base = self.bbox_head_refine[stage].feature_refine(features[0], rois)
        if not self.mbo_refine_fusion:
            return base
        gamma = self.refine_gamma[stage]
        aux_rgb = self.refine_align['rgb'][stage](features[1], rois)
        aux_ir = self.refine_align['ir'][stage](features[2], rois)
        return [b + gamma * (r + i) for b, r, i in zip(base, aux_rgb, aux_ir)]

    def loss(self, batch_inputs, batch_inputs2, batch_data_samples):
        if not self.mbo_auxiliary:
            raise RuntimeError('Auxiliary-free model is for inference only')
        features = self._training_features(batch_inputs, batch_inputs2)
        gt, ignored, metas = unpack_gt_instances(batch_data_samples)
        n_gt = sum(len(g) for g in gt)
        density = n_gt / max(len(batch_data_samples), 1)
        gate = self.mbo_gate
        # The baseline's primary gt_instances are IR annotations. Reuse the
        # identical objects in all branches: RGB/IR annotation lists are NOT
        # one-to-one and must never be zipped by their annotation index.
        names = ('fusion', 'rgb', 'ir')
        heads = (self.bbox_head_init, self.mbo_init_heads['rgb'],
                 self.mbo_init_heads['ir'])
        outputs = [head(x) for head, x in zip(heads, features)]
        losses = {}
        # 监控键（不含 "loss"，不参与反传）
        losses['mbo_gate'] = features[0][0].new_tensor(gate)
        losses['mbo_density'] = features[0][0].new_tensor(density)

        def supervise(stage_heads, stage_outputs, stage, rois=None, scale=1.0,
                      aux_scale=1.0):
            prepared = [prepare_loss(
                head, out, gt, metas, ignored,
                rois=None if rois is None else rois[index],
                cost_class=self.mbo_cfg['cost_class'],
                cost_bbox=self.mbo_cfg['cost_bbox'])
                for index, (head, out) in enumerate(zip(stage_heads, stage_outputs))]
            parts, diagnostics = balanced_stage_losses(
                prepared, self.mbo_cfg['temperature'], self.mbo_cfg['enabled'])
            for index, (name, part) in enumerate(zip(names, parts)):
                branch_scale = scale * (
                    1.0 if index == 0 else
                    self.mbo_cfg['aux_loss_weight'] * gate * aux_scale)
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
            losses['mbo_aux_scale_' + stage] = features[0][0].new_tensor(
                aux_scale)

        supervise(heads, outputs, 'init')
        rois = [head.filter_bboxes(*out) for head, out in zip(heads, outputs)]
        for stage in range(self.num_refine_stages):
            heads = (self.bbox_head_refine[stage],
                     self.mbo_refine_heads['rgb'][stage],
                     self.mbo_refine_heads['ir'][stage])
            outputs = []
            for index, (head, x, roi) in enumerate(zip(heads, features, rois)):
                if index == 0:
                    x = self._refine_features(stage, features, roi)
                else:
                    x = head.feature_refine(x, roi)
                outputs.append(head(x))
            supervise(heads, outputs, 'refine_' + str(stage), rois,
                      self.train_cfg.stage_loss_weights[stage],
                      aux_scale=self.mbo_cfg['refine_aux_scale'])
            if stage + 1 < self.num_refine_stages:
                rois = [head.refine_bboxes(*out, rois=roi)
                        for head, out, roi in zip(heads, outputs, rois)]
        return losses

    def predict(self, batch_inputs, batch_inputs2, batch_data_samples,
                rescale=True):
        if not self.mbo_refine_fusion:
            return super().predict(batch_inputs, batch_inputs2,
                                   batch_data_samples, rescale=rescale)
        features = self._training_features(batch_inputs, batch_inputs2)
        outs = self.bbox_head_init(features[0])
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for stage in range(self.num_refine_stages):
            head = self.bbox_head_refine[stage]
            outs = head(self._refine_features(stage, features, rois))
            if stage + 1 < self.num_refine_stages:
                rois = head.refine_bboxes(*outs, rois)
        batch_img_metas = [s.metainfo for s in batch_data_samples]
        predictions = self.bbox_head_refine[-1].predict_by_feat(
            *outs, rois=rois, batch_img_metas=batch_img_metas, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, predictions)

    def _forward(self, batch_inputs, batch_inputs2, batch_data_samples=None):
        if not self.mbo_refine_fusion:
            return super()._forward(batch_inputs, batch_inputs2)
        features = self._training_features(batch_inputs, batch_inputs2)
        outs = self.bbox_head_init(features[0])
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for stage in range(self.num_refine_stages):
            head = self.bbox_head_refine[stage]
            outs = head(self._refine_features(stage, features, rois))
            if stage + 1 < self.num_refine_stages:
                rois = head.refine_bboxes(*outs, rois)
        return outs

    def remove_auxiliary_branches(self):
        """Release training-only modules before deployment; leaves fusion intact."""
        if self.mbo_refine_fusion:
            raise RuntimeError('refine fusion needs the auxiliary branches')
        if self.training:
            raise RuntimeError('Call eval() before removing auxiliary branches')
        for name in ('mbo_necks', 'mbo_init_heads', 'mbo_refine_heads'):
            if hasattr(self, name):
                delattr(self, name)
        self.mbo_auxiliary = False
        return self
