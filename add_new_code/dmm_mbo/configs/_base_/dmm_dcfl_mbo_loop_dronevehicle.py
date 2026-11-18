"""Original DMM+DCFL + training-only instance-aware modality balancing."""
_base_ = ['dmm_dcfl_loop_dronevehicle.py']

custom_imports = dict(
    imports=['dmm_mbo.dcfl', 'dmm_mbo', 'dmm_mbo.training.progress_bar',
             'dmm_mbo.training.dota_per_class'], allow_failed_imports=False)

model = dict(
    type='MBODMMDetector',
    backbone_vi=dict(use_checkpoint=True),
    backbone_ir=dict(use_checkpoint=True),
    mbo_auxiliary=True,
    mbo_cfg=dict(enabled=True, temperature=1.0, aux_loss_weight=1.0,
                 cost_class=1.0, cost_bbox=1.0))

custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
default_hooks = dict(checkpoint=dict(max_keep_ckpts=-1))
work_dir = './work_dirs/dmm_dcfl_mbo_dronevehicle_loop'
load_from = None
resume = False
