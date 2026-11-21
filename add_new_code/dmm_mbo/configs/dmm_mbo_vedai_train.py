"""DMM + MBO on VEDAI; from-scratch training, no pretrained checkpoint."""
_base_ = [
    './_base_/dmm_s2anet_vedai_rgbt.py',
    './_base_/schedule_dmm_1x.py',
    './_base_/datasets/VEDAI.py',
    './_base_/default_runtime.py',
]
import os
custom_imports = dict(
    imports=['dmm_mbo', 'dmm_mbo.training.progress_bar',
             'dmm_mbo.training.dota_per_class'], allow_failed_imports=False)
model = dict(
    type='MBODMMDetector',
    backbone_vi=dict(use_checkpoint=True),
    backbone_ir=dict(use_checkpoint=True),
    mbo_auxiliary=True,
    mbo_fork='pre',
    mbo_refine_fusion=True,
    mbo_gamma_init=0.0,
    mbo_cfg=dict(enabled=True, temperature=1., aux_loss_weight=1.,
                 cost_class=1., cost_bbox=1.))
randomness = dict(seed=int(os.environ.get('DMM_TRAIN_SEED', '20261121')))
custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(max_keep_ckpts=-1))
work_dir = './work_dirs/dmm_mbo_vedai_loop'
load_from = None
resume = False
