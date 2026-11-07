# GauCho + DCFL 版 DMM，VEDAI（可见光 + 红外，6 类）
#
# 在 dmm_gaucho_1x_vedai.py 的基础上只改一件事：把两级检测头的分配器
# 从 MaxIoUAssigner 换成 DCFL 的 C2FAssigner（动态由粗到细分配）。
# 主干、DCF、GauCho 的高斯回归、数据管线、batch=4、12 epoch、
# 学习率策略、进度条全部不变，方便和纯 GauCho 做 A/B 对照。
#
# DCFL 只移植了分配器，没有移植依赖 DeformConvG 的 PCB 模块
# （那个算子只在 MMCV-DCFL 分叉里；分配器本身是纯 PyTorch，不需要它）。
#
# 训练请用仓库根目录的 train_gaucho_dcfl_vedai_loop.py（或直接点 ▶）。
_base_ = ['./dmm_gaucho_1x_vedai.py']

custom_imports = dict(
    imports=['mmrotate.hooks.progress_bar', 'dmm_gaucho'],
    allow_failed_imports=False,
)

# 角度定义必须与 _base_/dmm_s2anet_rgbt.py 里的 angle_version 保持一致
angle_version = 'le135'

# topk / topq / gauss_thr 用原文 DOTA 配置的取值。
# 分配器内部要把预测解回 xywha 才能算 IoU，这里统一用 GauCho 的 coder
# （use_box_type=False，解出来是裸 Tensor，配合 get_box_tensor 最省事）。
model = dict(
    bbox_head_init=dict(type='GauchoDCFLS2AHead'),
    bbox_head_refine=dict(type='GauchoDCFLS2ARefineHead'),
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
                    type='GauchoAnchorOBBDecoder',
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
                        type='GauchoAnchorOBBDecoder',
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
