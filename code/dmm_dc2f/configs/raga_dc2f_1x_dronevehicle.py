# DMM + DCFL + DC2F + RAGA，DroneVehicle（可见光 + 红外，5 类）
#
# 只把第二阶段检测头（精修头）从 S2ARefineHead 换成 DC2FS2ARefineHeadRAGA，
# 其余（主干、DCF、DSSM 融合、neck、DC2F 三处开关、数据管线、batch=2、
# 12 epoch、学习率策略、进度条）全部原样继承 —— 和 DMM + DC2F 只差精修头。
#
# 训练请用仓库根目录的 train_raga_dc2f_dronevehicle_loop.py（或直接点 ▶）。
_base_ = ['./dc2f_1x_dronevehicle.py']

model = dict(
    bbox_head_refine=dict(
        type='DC2FS2ARefineHeadRAGA',
        raga_cfg=dict(
            stacked_convs=2,
            down_rate=16,
            use_grr=True,
            use_scg=True,
        ),
    ),
)
