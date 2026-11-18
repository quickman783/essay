"""Author DCFL-only ablation: original delta coder and SmoothL1 regression."""
_base_ = ['dmm_s2anet_1x_dronevehicle.py']
custom_imports = dict(imports=['dmm_mbo.dcfl'], allow_failed_imports=False)
angle_version = 'le135'
dcfl_assigner = dict(
    type='C2FAssigner', assign_metric='gjsd', topk=16, topq=12,
    constraint='dgmm', gauss_thr=0.6,
    iou_calculator=dict(type='RBboxMetrics2D'),
    bbox_coder=dict(
        type='DeltaXYWHTRBBoxCoder', angle_version=angle_version,
        norm_factor=1, edge_swap=False, proj_xy=True,
        target_means=(0., 0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1., 1.), use_box_type=False))
model = dict(
    bbox_head_init=dict(type='DCFLS2AHead'),
    bbox_head_refine=[dict(
        type='DCFLS2ARefineHead', num_classes=5,
        in_channels=256, stacked_convs=2, feat_channels=256,
        frm_cfg=dict(type='AlignConv', feat_channels=256, kernel_size=3,
                     strides=[8, 16, 32, 64, 128]),
        anchor_generator=dict(type='PseudoRotatedAnchorGenerator',
                              strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder', angle_version=angle_version,
            norm_factor=1, edge_swap=False, proj_xy=True,
            target_means=(0., 0., 0., 0., 0.),
            target_stds=(1., 1., 1., 1., 1.)),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True,
                      gamma=2., alpha=0.25, loss_weight=1.),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.))],
    train_cfg=dict(
        init=dict(assigner=dict(_delete_=True, **dcfl_assigner)),
        refine=[dict(assigner=dcfl_assigner, allowed_border=-1,
                     pos_weight=-1, debug=False)]))
load_from = None
resume = False
work_dir = './work_dirs/dmm_dcfl_12e_dronevehicle'
