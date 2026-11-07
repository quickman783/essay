"""DMM + DCFL 对照组在 DroneVehicle 上的循环训练（可见光 + 红外，5 类）。

在 VS Code 里打开本文件，点右上角 ▶ 运行。

这是消融实验用的**对照组**：只把标签分配换成 DCFL，回归分支保持基线原样
（DeltaXYWHTRBBoxCoder + SmoothL1），没有 GauCho。

四组实验两两对照，就能拆出「高斯回归」和「DCFL 分配」各自的贡献：

    基线            L1 + MaxIoU      train_dronevehicle_loop.py
    GauCho          高斯 + MaxIoU    train_gaucho_dronevehicle_loop.py
    DCFL（本脚本）  L1 + DCFL        work_dirs/dmm_dcfl_dronevehicle_loop
    GauCho + DCFL   高斯 + DCFL      work_dirs/gaucho_dcfl_dronevehicle_loop

batch size 沿用配置文件里的设置（2），本脚本不做覆盖。

训练机制（进度条、学习率跨次保留、指标表格、权重只增不删）与实验组一致。
输出目录与实验组完全分开：

    权重   work_dirs/dmm_dcfl_dronevehicle_loop/run_<时间戳>/cycle_N/
    表格   work_dirs/dmm_dcfl_dronevehicle_metrics.xlsx
    状态   work_dirs/dmm_dcfl_dronevehicle_loop/run_state.json

常用参数：

    python train_dmm_dcfl_dronevehicle_loop.py              # 从头训
    python train_dmm_dcfl_dronevehicle_loop.py --cycle 2    # 接已训权重，从第 2 轮算学习率
    python train_dmm_dcfl_dronevehicle_loop.py --reset      # 清空续训状态

把练好的 .pth 放进 weights/dmm_dcfl_dronevehicle/，脚本会自动挑最新的当起点。
"""
import os
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dmm_dcfl.train_loop import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("dronevehicle_dcfl"))
