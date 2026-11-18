"""Shared DMM model definition (Disparity-guided Multispectral Mamba).

Both backbones are trained from scratch, no pretrained checkpoint is loaded.
``num_classes`` and the dataset come from the child config, e.g.
``configs/dmm/dmm_s2anet_1x_dronevehicle.py`` or
``configs/dmm/dmm_s2anet_1x_vedai.py``.
"""

angle_version = "le135"
num_classes = 5

model = dict(
    type="virRefineSingleStageDetector",
    data_preprocessor=dict(
        type="mmdet.DetRGBTDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        mean2=[123.675, 116.28, 103.53],
        std2=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False,
    ),
    backbone_vi=dict(
        type="mmdet.MM_VSSM",
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        dims=96,
        depths=(2, 2, 5, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
    ),
    backbone_ir=dict(
        type="mmdet.MM_VSSM",
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        dims=96,
        depths=(2, 2, 5, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
    ),
    fusblock=dict(
        type="DCFModule",
        in_channels=[96, 192, 384, 768],
        drop_path=0.2,
        ssm_ratio=2.0,
        d_state=16,
        dt_rank="auto",
        mlp_ratio=0.0,
        attn_drop_rate=0.0,
        mlp_drop=0.0,
        disable_z=False,
    ),
    neck=dict(
        type="mmdet.FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5,
    ),
    mtablock=dict(type="MTAttentionBlock", in_channels=[96, 192, 384, 768]),
    bbox_head_init=dict(
        type="S2AHead",
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type="FakeRotatedAnchorGenerator",
            angle_version=angle_version,
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHTRBBoxCoder",
            angle_version=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            use_box_type=False,
        ),
        loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=0.11, loss_weight=1.0),
    ),
    bbox_head_refine=[
        dict(
            type="S2ARefineHead",
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(type="AlignConv", feat_channels=256, kernel_size=3, strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(type="PseudoRotatedAnchorGenerator", strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type="DeltaXYWHTRBBoxCoder",
                angle_version=angle_version,
                norm_factor=1,
                edge_swap=False,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=0.11, loss_weight=1.0),
        )
    ],
    train_cfg=dict(
        init=dict(
            assigner=dict(
                type="mmdet.MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type="RBboxOverlaps2D"),
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        refine=[
            dict(
                assigner=dict(
                    type="mmdet.MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type="RBboxOverlaps2D"),
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            )
        ],
        stage_loss_weights=[1.0],
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms_rotated", iou_threshold=0.1),
        max_per_img=2000,
    ),
)
