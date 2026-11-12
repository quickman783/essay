"""DMM + DC2F + C2Former 在 VEDAI 上的循环训练（可见光 + 红外，6 类）。

在 VS Code 里打开本文件，点右上角 ▶ 运行。

相对 DMM + DC2F 只改一件事：把 fusblock 的 DSSM 换成 C2Former（ICA + AFS），
只作用于 level 1/2/3，level 0 保持 DSSM（neck 是 start_level=1，无人读取）。
另加一处结构：校准后的两支各自回支，再用逐通道门控 λ 把差分重注入
（λ 初始化为 0，训练初期严格等价于两路求和）。

DC2F 的三处改进开关（SNT + CSC + BIND）自动带过来，不用重写配置。
回归分支仍是基线 L1 + SmoothL1。

输出路径与其它实验完全分开：
    权重   work_dirs/c2f_dc2f_vedai_loop/run_<时间戳>/cycle_N/
    表格   work_dirs/c2f_dc2f_vedai_metrics.xlsx
    状态   work_dirs/c2f_dc2f_vedai_loop/run_state.json

常用参数：

    python train_c2f_dc2f_vedai_loop.py                 # 从头训
    python train_c2f_dc2f_vedai_loop.py --cycle 3        # 接已训权重，从第 3 轮算学习率
    python train_c2f_dc2f_vedai_loop.py --init xxx.pth --cycle 4
    python train_c2f_dc2f_vedai_loop.py --reset          # 清空续训状态（旧文件改名存档）

把练好的 .pth 放进 weights/c2f_dc2f_vedai/，脚本会自动挑最新的当起点。
"""
import os
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dmm_dc2f.train_loop import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("vedai_c2f_dc2f"))
