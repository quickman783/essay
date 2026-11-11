"""DMM + DC2F 在 VEDAI 上的循环训练（可见光 + 红外，6 类）。

在 VS Code 里打开本文件，点右上角 ▶ 运行。

相对 DMM + DCFL 只改了标签分配器（C2FAssigner → DC2FAssigner），
三处改进开关全部打开：

    SNT  粗筛阈值按目标形状归一化（小目标 / 极端长宽比不再被整批误杀）
    CSC  精修阶段只给「初始阶段零正样本」的目标放开 dgmm 掩码
    BIND 归属写成「编号最大者优先」的显式形式

回归分支保持基线原样（DeltaXYWHTRBBoxCoder + SmoothL1），不引入高斯回归。
其余（主干、DCF、S2A 结构、batch=4、12 epoch、学习率策略、进度条）全部不变。

输出路径与其它实验完全分开：
    权重   work_dirs/dc2f_vedai_loop/run_<时间戳>/cycle_N/
    表格   work_dirs/dc2f_vedai_metrics.xlsx
    状态   work_dirs/dc2f_vedai_loop/run_state.json

常用参数：

    python train_dc2f_vedai_loop.py                 # 从头训
    python train_dc2f_vedai_loop.py --cycle 3       # 接已训权重，从第 3 轮算学习率
    python train_dc2f_vedai_loop.py --init xxx.pth --cycle 4
    python train_dc2f_vedai_loop.py --reset         # 清空续训状态（旧文件改名存档）

把练好的 .pth 放进 weights/dc2f_vedai/，脚本会自动挑最新的当起点。
"""
import os
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dmm_dc2f.train_loop import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("vedai_dc2f"))
