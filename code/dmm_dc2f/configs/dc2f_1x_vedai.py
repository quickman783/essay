# DMM + DC2F，VEDAI（可见光 + 红外，6 类）
#
# 在 dmm_dcfl_1x_vedai.py（DMM + DCFL）的基础上，把 DCFL 的 C2FAssigner
# 换成 DC2FAssigner，并打开三处改进开关：
#     SNT  粗筛阈值按目标形状归一化（min(0.8, 上界)）
#     CSC  精修阶段只给「初始阶段零正样本」的目标放开 dgmm 掩码
#     BIND 归属写成「编号最大者优先」的显式形式
#
# **只改标签分配**：回归分支仍是基线原样
# （DeltaXYWHTRBBoxCoder + SmoothL1，不含 GauCho），
# 所以和 DMM + DCFL 那一组只差 DC2F 的三处开关，可以做严格 A/B。
#
# 主干、DCF、S2A 结构、数据管线、batch=4、12 epoch、学习率策略、进度条
# 全部继承，不做任何改动。
#
# 训练请用仓库根目录的 train_dc2f_vedai_loop.py（或直接点 ▶）。
_base_ = ['../../dmm_dcfl/configs/dmm_dcfl_1x_vedai.py']

custom_imports = dict(
    imports=['mmrotate.hooks.progress_bar', 'dmm_dc2f'],
    allow_failed_imports=False,
)

angle_version = 'le135'

# topk / topq / gauss_thr 与 DCFL 那组完全一致（对照必须同参），
# 只有 DC2F 的三处开关不同。
_dc2f_assigner = dict(
    _delete_=True,
    type='DC2FAssigner',
    assign_metric='gjsd',
    topk=16,
    topq=12,
    constraint='dgmm',
    gauss_thr=0.6,
    # ---- DC2F 的三处改进 ----
    snt=True,
    thr_ratio=0.8,
    anchor_size=32.0,
    csc=True,
    csc_free_thr=3.0,
    bind_last=True,
    iou_calculator=dict(type='RBboxMetrics2D'),
    # 用基线自己的 coder（原始 DCFL 仓库也是普通角度 coder）
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
)

model = dict(
    bbox_head_init=dict(type='DC2FS2AHead'),
    bbox_head_refine=dict(type='DC2FS2ARefineHead'),
    train_cfg=dict(
        init=dict(assigner=_dc2f_assigner),
        refine=[
            dict(
                assigner=_dc2f_assigner,
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            )
        ],
    ),
)
