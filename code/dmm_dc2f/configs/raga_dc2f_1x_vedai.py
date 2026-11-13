# DMM + DCFL + DC2F + RAGA，VEDAI（可见光 + 红外，6 类）
#
# 只把第二阶段检测头（精修头）从 S2ARefineHead 换成 DC2FS2ARefineHeadRAGA，
# 其余（主干、DCF、DSSM 融合、neck、DC2F 三处开关、数据管线、batch=4、
# 12 epoch、学习率策略、进度条）全部原样继承 —— 和 DMM + DC2F 只差精修头。
#
# RAGA = Refinement-stage Adaptive Geometric Alignment，三个子模块：
#     TPM  任务先验调制（cls / reg 各一套，可学习温度 tau）
#     GRR  几何重采样精修（偏移场与采样权重解耦为双分支）
#     SCG  语义一致性门控（可学习基线偏置 beta）
#
# 第一阶段 bbox_head_init 不动：它的输出会被精修覆盖，装在那里等于白加参数。
#
# 训练请用仓库根目录的 train_raga_dc2f_vedai_loop.py（或直接点 ▶）。
_base_ = ['./dc2f_1x_vedai.py']

model = dict(
    bbox_head_refine=dict(
        type='DC2FS2ARefineHeadRAGA',
        # 三个开关用于出消融表，关掉任一即可
        raga_cfg=dict(
            stacked_convs=2,      # TPM 的动态权重段数
            down_rate=16,         # TPM 压缩率，改 8 参数翻倍
            use_grr=True,         # GRR 几何重采样（论文里增益主要来自这一项）
            use_scg=True,         # SCG 语义一致性门控
        ),
    ),
)
