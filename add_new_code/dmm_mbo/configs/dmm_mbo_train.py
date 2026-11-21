"""DMM + MBO: original S2A heads and MaxIoU assignment, no DCFL."""
_base_ = ['./_base_/dmm_s2anet_1x_dronevehicle.py']
import os
custom_imports = dict(
    imports=['dmm_mbo', 'dmm_mbo.training.progress_bar',
             'dmm_mbo.training.dota_per_class'], allow_failed_imports=False)
pretrained_vmamba = os.path.join(
    os.environ.get('DMM_BASE_ROOT', '{{fileDirname}}/../../../code'),
    'weights/vssm/vssm_tiny_0230_ckpt_epoch_262.pth')
model = dict(
    type='MBODMMDetector',
    backbone_vi=dict(use_checkpoint=True, pretrained=pretrained_vmamba),
    backbone_ir=dict(use_checkpoint=True, pretrained=pretrained_vmamba),
    mbo_auxiliary=True,
    mbo_cfg=dict(enabled=True, temperature=1., aux_loss_weight=1.,
                 cost_class=1., cost_bbox=1.))
randomness = dict(seed=int(os.environ.get('DMM_TRAIN_SEED', '1017387474')))
custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
if os.environ.get('DMM_CHECKPOINT_ARCHIVE_ROOT'):
    custom_imports['imports'].append('dmm_mbo.server_tools.checkpoint_storage')
    custom_hooks.append(dict(type='ArchiveEarlyCheckpointsHook',
                             archive_root=os.environ['DMM_CHECKPOINT_ARCHIVE_ROOT'],
                             epochs=[1, 2, 3, 4]))
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
default_hooks = dict(checkpoint=dict(max_keep_ckpts=-1))
work_dir = './work_dirs/dmm_mbo_pretrained_dronevehicle_loop'
load_from = None
resume = False
