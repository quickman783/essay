"""一键开始训练 VEDAI（可见光 + 红外双模态，6 类）。

用法：在 VS Code 里打开本文件，点右上角的 ▶ 运行按钮。

脚本会自动完成下面几件事：
  1. 切到仓库根目录，让配置里的相对路径 datasets/vedai 生效；
  2. 使用装了 mmcv / mmdet / mmrotate 的 conda 解释器，
     不受 VS Code 当前选中解释器的影响；
  3. 把日志和权重写到 work_dirs/vedai_6cls/ 下。

需要临时改参数时，往 EXTRA_ARGS 里加即可，例如：
    EXTRA_ARGS = ["--cfg-options", "train_dataloader.batch_size=2"]
"""
import warnings
warnings.filterwarnings("ignore")
import os
import subprocess
import sys

# 装了训练依赖的解释器（路径不存在时自动退回当前解释器）
# paper310 是按论文要求配置的环境：torch 2.1.2 / mmcv 2.1.0 / selective_scan
CONDA_PYTHON = r"C:\Users\feng1\anaconda3\envs\paper310\python.exe"

# 训练配置：模型 + 数据 + 12 epoch 的训练策略
CONFIG = os.path.join("code", "configs", "dmm", "dmm_s2anet_1x_vedai.py")

# 日志与权重的输出目录
WORK_DIR = os.path.join("work_dirs", "vedai_6cls")

# 额外命令行参数，目前留空
EXTRA_ARGS = []

# 本文件所在目录就是仓库根目录
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    python = CONDA_PYTHON if os.path.isfile(CONDA_PYTHON) else sys.executable
    train_script = os.path.join(REPO_ROOT, "code", "tools", "train.py")

    if not os.path.isdir(os.path.join(REPO_ROOT, "datasets", "vedai")):
        raise SystemExit(
            "找不到 datasets/vedai，请先用 code/tools 下的脚本准备数据，"
            "或设置环境变量 DMM_VEDAI_ROOT 指向数据目录。"
        )

    command = [
        python,
        train_script,
        os.path.join(REPO_ROOT, CONFIG),
        "--work-dir",
        os.path.join(REPO_ROOT, WORK_DIR),
    ] + EXTRA_ARGS

    print("仓库根目录 :", REPO_ROOT)
    print("解释器     :", python)
    print("完整命令   :", " ".join(command))
    print("提示       : 每 50 次迭代打印一行，按 Ctrl+C 可中断；")
    print("             中断后加 --resume 参数可接着训练。")
    print("-" * 78, flush=True)
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
