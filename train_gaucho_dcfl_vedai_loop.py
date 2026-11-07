"""GauCho + DCFL 版 DMM 在 VEDAI 上的循环训练（可见光 + 红外，6 类）。

在 VS Code 里打开本文件，点右上角 ▶ 运行。

在 GauCho（高斯 Cholesky 回归）基础上，把标签分配换成 DCFL 的动态由粗到细分配：
    粗筛  用高斯 JS 散度（gjsd）给每个 GT 挑 topk=16 个候选锚框
    细选  用「预测框 IoU + 分类置信度」重排，每个 GT 最终只保留 topq=12 个
    掩码  dgmm 高斯距离，剔除离 GT 太远的候选
其余（主干、DCF、S2A 结构、batch=4、12 epoch、学习率策略、进度条）都不变，
可以直接和 train_gaucho_vedai_loop.py 的结果做 A/B 对照。

输出目录与纯 GauCho 分开：
    权重   work_dirs/gaucho_dcfl_vedai_loop/run_<时间戳>/cycle_N/
    表格   work_dirs/gaucho_dcfl_vedai_metrics.xlsx
    状态   work_dirs/gaucho_dcfl_vedai_loop/run_state.json

常用参数：

    python train_gaucho_dcfl_vedai_loop.py              # 从头训
    python train_gaucho_dcfl_vedai_loop.py --cycle 3    # 接已训权重，从第 3 轮算学习率
    python train_gaucho_dcfl_vedai_loop.py --reset      # 清空续训状态（旧文件改名存档）

把练好的 .pth 放进 weights/gaucho_dcfl_vedai/，脚本会自动挑最新的当起点。
"""
import os
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dmm_gaucho.train_loop import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("vedai_dcfl"))
