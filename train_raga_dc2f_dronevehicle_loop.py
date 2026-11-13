"""DMM + DC2F + RAGA 在 DroneVehicle 上的循环训练（可见光 + 红外，5 类）。

在 VS Code 里打开本文件，点右上角 ▶ 运行。

相对 DMM + DC2F 只改一件事：第二阶段精修头换成 DC2FS2ARefineHeadRAGA。
RAGA 只作用于精修阶段，三个子模块（TPM 任务先验调制 / GRR 几何重采样精修 /
SCG 语义一致性门控），只多 3 个标量自由度（tau ×2、beta ×1）。

DC2F 的三处改进开关（SNT / CSC / BIND）自动带过来，不用重写配置。
第一阶段 bbox_head_init 不动；回归仍是基线 L1 + SmoothL1。
batch size 沿用配置文件里的设置（2）。

输出路径与其它实验完全分开：
    权重   work_dirs/raga_dc2f_dronevehicle_loop/run_<时间戳>/cycle_N/
    表格   work_dirs/raga_dc2f_dronevehicle_metrics.xlsx
    状态   work_dirs/raga_dc2f_dronevehicle_loop/run_state.json

常用参数：

    python train_raga_dc2f_dronevehicle_loop.py           # 从头训
    python train_raga_dc2f_dronevehicle_loop.py --cycle 2
    python train_raga_dc2f_dronevehicle_loop.py --reset   # 清空续训状态

把练好的 .pth 放进 weights/raga_dc2f_dronevehicle/，脚本会自动挑最新的当起点。
"""
import os
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dmm_dc2f.train_loop import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("dronevehicle_raga_dc2f"))
