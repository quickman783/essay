# DMM + DCFL + DC2F + C2Former，DroneVehicle（可见光 + 红外，5 类）
#
# 相对 dc2f_1x_dronevehicle.py 只多一件事：把 fusblock 的 DSSM 换成 C2Former（ICA + AFS），
# 只作用于 level 1/2/3（192/384/768）；level 0 仍走 DSSM
# （neck 是 start_level=1，level 0 无人读取）。
#
# DC2F 的三处开关（snt / csc / bind_last）、分配超参、数据管线、训练策略、
# 进度条、batch=2、12 epoch 全部原样继承，不做任何改动。
#
# 相对官方 C2Former 只加一处结构：校准后的两支各自回支，再用逐通道门控 λ
# 把差分 x_vi' - x_ir' 重注入（λ 初始化为 0）。
#
# 训练请用仓库根目录的 train_c2f_dc2f_dronevehicle_loop.py（或直接点 ▶）。
_base_ = ['./dc2f_1x_dronevehicle.py']

model = dict(
    fusblock=dict(
        type='DCFModule',
        in_channels=[96, 192, 384, 768],
        drop_path=0.2,
        ssm_ratio=2.0,
        d_state=16,
        dt_rank='auto',
        mlp_ratio=0.0,
        attn_drop_rate=0.0,
        mlp_drop=0.0,
        disable_z=False,
        # ---- 切换到 C2Former 分支 ----
        fus_type='c2f',
        c2f_start_level=1,
        use_diff_gate=True,
        c2f_cfg=dict(
            # 按 level 索引，长度必须等于 len(in_channels)；
            # 第 0 项是占位，level 0 仍走 DSSM，不会被读取。
            dims_out=[36, 72, 144, 288],
            num_heads=[1, 3, 6, 12],
            n_groups=[1, 2, 3, 6],
            stride=3,
            offset_range_factor=2,
            no_off=False,
            attn_drop=0.0,
            proj_drop=0.0,
        ),
    ),
)
