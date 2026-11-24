"""DMM + DCFL + stage-decayed MBO on VEDAI.

Heads and assigners follow code/dmm_dcfl/configs/dmm_dcfl_1x_vedai.py;
MBO switches follow dmm_mbo_vedai_weakaux.py (refine_aux_scale = 0).
"""
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

angle_version = 'le135'

model = dict(
    type='MBODMMDetector',
    backbone_vi=dict(use_checkpoint=True),
    backbone_ir=dict(use_checkpoint=True),
    mbo_auxiliary=True,
    mbo_fork='pre',
    mbo_refine_fusion=False,
    mbo_gamma_init=0.0,
    mbo_cfg=dict(enabled=True, temperature=1., aux_loss_weight=1.,
                 cost_class=1., cost_bbox=1.,
                 refine_aux_scale=0.),
    # 基配置里 bbox_head_refine 是 list，mmengine 对 list 整体替换而不是递归合并，
    # 所以这里必须整段重写，否则 frm_cfg / anchor_generator / bbox_coder 会丢。
    bbox_head_init=dict(type='DCFLS2AHead'),
    bbox_head_refine=[
        dict(
            type='DCFLS2ARefineHead',
            num_classes=6,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(type='AlignConv', feat_channels=256, kernel_size=3,
                         strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(type='PseudoRotatedAnchorGenerator',
                                  strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=1,
                edge_swap=False,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11,
                           loss_weight=1.0),
        )
    ],
    train_cfg=dict(
        init=dict(
            assigner=dict(
                _delete_=True,
                type='C2FAssigner',
                assign_metric='gjsd',
                topk=16,
                topq=12,
                constraint='dgmm',
                gauss_thr=0.6,
                iou_calculator=dict(type='RBboxMetrics2D'),
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder',
                    angle_version=angle_version,
                    norm_factor=1,
                    edge_swap=False,
                    proj_xy=True,
                    target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                    target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                    use_box_type=False,
                ),
            ),
        ),
        refine=[
            dict(
                assigner=dict(
                    _delete_=True,
                    type='C2FAssigner',
                    assign_metric='gjsd',
                    topk=16,
                    topq=12,
                    constraint='dgmm',
                    gauss_thr=0.6,
                    iou_calculator=dict(type='RBboxMetrics2D'),
                    bbox_coder=dict(
                        type='DeltaXYWHTRBBoxCoder',
                        angle_version=angle_version,
                        norm_factor=1,
                        edge_swap=False,
                        proj_xy=True,
                        target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                        use_box_type=False,
                    ),
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            )
        ],
    ),
)
randomness = dict(seed=int(os.environ.get('DMM_TRAIN_SEED', '20261121')))
custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(max_keep_ckpts=-1))
work_dir = './work_dirs/dmm_dcfl_mbo_vedai_weakaux'
load_from = None
resume = False
