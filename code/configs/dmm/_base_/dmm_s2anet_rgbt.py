"""DMM 模型的共用定义（Disparity-guided Multispectral Mamba，视差引导多光谱 Mamba）。

两个主干默认加载 VMamba 的 ImageNet 预训练权重；若权重文件不存在则自动
退化为从头训练。``num_classes`` 与所用数据集由子配置决定，见
``configs/dmm/dmm_s2anet_1x_vedai.py`` 和 ``configs/dmm/dmm_s2anet_1x_dronevehicle.py``。
"""

import os

# VMamba-Tiny 的 ImageNet 预训练权重（论文配置里的 vmamba_pretained）。
# 可用环境变量 DMM_VMAMBA_CKPT 指定其他路径；文件不存在时自动改为 None。
vmamba_pretrained = os.environ.get(
    "DMM_VMAMBA_CKPT", "weights/vssm/vssm_tiny_0230_ckpt_epoch_262.pth")
if not os.path.isfile(vmamba_pretrained):
    vmamba_pretrained = None

# 角度编码方式：le135 表示角度范围 [-pi/2, pi/4)
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
    # ??????VMamba-Tiny?????
    backbone_vi=dict(
        type="mmdet.MM_VSSM",
        out_indices=(0, 1, 2, 3),
        pretrained=vmamba_pretrained,
        use_checkpoint=True,  # ??????????
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
    # ?????????????????
    backbone_ir=dict(
        type="mmdet.MM_VSSM",
        out_indices=(0, 1, 2, 3),
        pretrained=vmamba_pretrained,
        use_checkpoint=True,  # ??????????
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
    # DCF ???????????????????
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
    # FPN ???????? 5 ???
    neck=dict(
        type="mmdet.FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5,
    ),
    # ?????? MT ?????
    mtablock=dict(type="MTAttentionBlock", in_channels=[96, 192, 384, 768]),
    # S2A-Net ????????????
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
    # S2A-Net ?????? AlignConv ????????
    bbox_head_refine=dict(
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
    ),
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
