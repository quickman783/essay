"""MBO with mathematically identical vectorized DCFL selection loops."""
_base_ = ['dmm_dcfl_mbo_loop_dronevehicle.py']
import os
randomness = dict(seed=int(os.environ.get('DMM_TRAIN_SEED', '20261116')))
custom_imports = dict(
    imports=['dmm_mbo.dcfl', 'dmm_mbo',
             'dmm_mbo.fast_assigner',
             'dmm_mbo.training.progress_bar', 'dmm_mbo.training.dota_per_class'],
    allow_failed_imports=False)
custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
if os.environ.get('DMM_CHECKPOINT_ARCHIVE_ROOT'):
    custom_imports['imports'].append('dmm_mbo.server_tools.checkpoint_storage')
    custom_hooks.append(dict(type='ArchiveEarlyCheckpointsHook',
                             archive_root=os.environ['DMM_CHECKPOINT_ARCHIVE_ROOT'],
                             epochs=[1, 2, 3, 4]))
model = dict(train_cfg=dict(
    init=dict(assigner=dict(type='VectorizedC2FAssigner')),
    refine=[dict(assigner=dict(
        type='VectorizedC2FAssigner', assign_metric='gjsd', topk=16, topq=12,
        constraint='dgmm', gauss_thr=0.6,
        iou_calculator=dict(type='RBboxMetrics2D'),
        bbox_coder=dict(type='DeltaXYWHTRBBoxCoder', angle_version='le135',
                        norm_factor=1, edge_swap=False, proj_xy=True,
                        target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.), use_box_type=False)),
                 allowed_border=-1, pos_weight=-1, debug=False)]))
