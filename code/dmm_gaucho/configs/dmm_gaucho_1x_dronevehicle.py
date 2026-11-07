# GauCho 版 DMM，在 DroneVehicle 上训练（可见光 + 红外，5 类）
#
# 与 dmm_s2anet_1x_dronevehicle.py 的差别只有三处，其余（主干、DCF、S2A 结构、
# 数据管线、batch=2、12 epoch、学习率策略、进度条）全部继承基配置不变：
#   * 检测头换成 GauchoS2AHead / GauchoS2ARefineHead
#   * bbox_coder 换成 GauchoAnchorOBBDecoder（Cholesky 高斯参数）
#   * loss_bbox 换成 GauchoGDLoss（GWD），并把 reg_decoded_bbox 打开
#
# use_box_type 必须和基配置保持一致：
#   第一阶段（bbox_head_init）False —— 输出裸 Tensor，AlignConv 的
#       torch.stack 才能吃；基配置里也是显式写的 False
#   第二阶段（bbox_head_refine）True —— 输出 RotatedBoxes，推理时
#       scale_boxes 才会走 rescale_ 分支；基配置里没写，用的就是类默认值 True
#
# 训练请用仓库根目录的 train_gaucho_dronevehicle_loop.py（或直接点 ▶）。
_base_ = ['../../configs/dmm/dmm_s2anet_1x_dronevehicle.py']

# 注册 GauCho 的 coder / loss / head（config 加载时自动 import）
custom_imports = dict(
    imports=['mmrotate.hooks.progress_bar', 'dmm_gaucho'],
    allow_failed_imports=False,
)

# classes: car, truck, bus, van, freight-car
num_classes = 5
# 角度定义必须与 _base_/dmm_s2anet_rgbt.py 里的 angle_version 保持一致
angle_version = 'le135'

model = dict(
    bbox_head_init=dict(
        type='GauchoS2AHead',
        # 目标直接用原始 gt 框，损失在高斯参数上算
        reg_decoded_bbox=True,
        bbox_coder=dict(
            _delete_=True,
            type='GauchoAnchorOBBDecoder',
            angle_version=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            use_box_type=False,
        ),
        loss_bbox=dict(
            _delete_=True,
            type='GauchoGDLoss',
            loss_type='gwd',
            fun='log1p',
            tau=1.0,
            gaussian_prediction=True,
            reduction='mean',
            loss_weight=1.0,
        ),
    ),
    bbox_head_refine=dict(
        type='GauchoS2ARefineHead',
        reg_decoded_bbox=True,
        bbox_coder=dict(
            _delete_=True,
            type='GauchoAnchorOBBDecoder',
            angle_version=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            use_box_type=True,
        ),
        loss_bbox=dict(
            _delete_=True,
            type='GauchoGDLoss',
            loss_type='gwd',
            fun='log1p',
            tau=1.0,
            gaussian_prediction=True,
            reduction='mean',
            loss_weight=1.0,
        ),
    ),
)
