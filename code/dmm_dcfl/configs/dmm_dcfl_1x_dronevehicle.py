# DMM + DCFL 对照组，DroneVehicle（可见光 + 红外，5 类）
#
# 相对基线 dmm_s2anet_1x_dronevehicle.py 只改一件事：把两级检测头的分配器
# 从 MaxIoUAssigner 换成 DCFL 的 C2FAssigner。
# 回归分支保持基线原样（DeltaXYWHTRBBoxCoder + SmoothL1），**不含 GauCho**，
# 用来和 train_gaucho_dcfl_dronevehicle_loop.py 做 A/B 对照：
#     两者唯一差别就是回归分支是不是高斯 Cholesky。
#
# 主干、DCF、S2A 结构、数据管线、batch=2、12 epoch、学习率策略、
# 进度条全部继承基线，不做任何改动。
#
# 训练请用仓库根目录的 train_dmm_dcfl_dronevehicle_loop.py（或直接点 ▶）。
_base_ = ['../../configs/dmm/dmm_s2anet_1x_dronevehicle.py']

custom_imports = dict(
    imports=['mmrotate.hooks.progress_bar', 'dmm_dcfl'],
    allow_failed_imports=False,
)

# 角度定义必须与 _base_/dmm_s2anet_rgbt.py 里的 angle_version 保持一致
angle_version = 'le135'

# topk / topq / gauss_thr 与 DCFL 原文 DOTA 配置一致，
# 也与 GauCho + DCFL 那一组完全相同（对照组必须同参）。
# 分配器内部要把预测解回 xywha 才能算 IoU，这里用基线自己的 coder。
model = dict(
    bbox_head_init=dict(type='DCFLS2AHead'),
    bbox_head_refine=dict(type='DCFLS2ARefineHead'),
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
