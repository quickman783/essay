"""GauCho 版 DMM 在 DroneVehicle 上的循环训练（可见光 + 红外，5 类）。

在 VS Code 里打开本文件，点右上角 ▶ 运行。

模型代码在 ``code/dmm_gaucho/`` 下，**没有改动原来的 mmrotate 实现**。
训练机制（进度条、学习率跨次保留、指标表格、权重不覆盖）与
``train_dronevehicle_loop.py`` 完全一致，但输出目录和表格都是独立的一份：

    权重   work_dirs/gaucho_dronevehicle_loop/run_<时间戳>/cycle_N/
    表格   work_dirs/gaucho_dronevehicle_metrics.xlsx
    状态   work_dirs/gaucho_dronevehicle_loop/run_state.json

batch size 沿用配置文件里的设置（``data/dv`` 那套是 2），本脚本不做覆盖。

常用参数：

    python train_gaucho_dronevehicle_loop.py            # 从头训
    python train_gaucho_dronevehicle_loop.py --cycle 2  # 接已训好的权重，从第 2 轮算学习率
    python train_gaucho_dronevehicle_loop.py --init xxx.pth --cycle 2
    python train_gaucho_dronevehicle_loop.py --reset    # 清空续训状态（旧文件改名存档）

把练好的 .pth 放进 ``weights/gaucho_dronevehicle/``，脚本会自动挑最新的当起点。
"""
import os
import sys

# 让 code/ 下的包（mmrotate、dmm_gaucho）可以被 import
CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dmm_gaucho.train_loop import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("dronevehicle"))
