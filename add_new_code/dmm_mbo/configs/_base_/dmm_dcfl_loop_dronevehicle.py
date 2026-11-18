"""DCFL-only author-loop hooks and unfrozen memory-saving backbones."""
_base_ = ['dmm_dcfl_12e_dronevehicle.py']
custom_imports = dict(
    imports=['dmm_mbo.dcfl', 'dmm_mbo.training.progress_bar',
             'dmm_mbo.training.dota_per_class'], allow_failed_imports=False)
custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
model = dict(backbone_vi=dict(use_checkpoint=True),
             backbone_ir=dict(use_checkpoint=True))
