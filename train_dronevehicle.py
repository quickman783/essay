"""一键开始训练 DroneVehicle（可见光 + 红外双模态，5 类，单轮 12 epoch）。

用法：在 VS Code 里打开本文件，点右上角的 ▶ 运行按钮。

只跑一轮、不循环、不写指标表格 —— 循环训练 + 自动记录表格请用 train_dronevehicle_loop.py。

脚本会自动完成下面几件事：
  1. 切到仓库根目录，让配置里的相对路径 data/dv 生效；
  2. 使用装了 mmcv / mmdet / mmrotate 的 conda 解释器，
     不受 VS Code 当前选中解释器的影响；
  3. 把日志和权重写到 work_dirs/dronevehicle_6cls/ 下。

想接着某个权重训练，往 EXTRA_ARGS 里加 load_from：
    EXTRA_ARGS = ["--cfg-options", "load_from=weights/dronevehicle/xxx.pth"]
"""
import warnings
warnings.filterwarnings("ignore")
import os
import subprocess
import sys

# 装了训练依赖的解释器（路径不存在时自动退回当前解释器）
CONDA_PYTHON = r"C:\Users\feng1\anaconda3\envs\paper310\python.exe"

# 训练配置：模型 + 数据 + 12 epoch 的训练策略
CONFIG = os.path.join("code", "configs", "dmm", "dmm_s2anet_1x_dronevehicle.py")

# 日志与权重的输出目录
WORK_DIR = os.path.join("work_dirs", "dronevehicle_6cls")

# 训练 batch size（DroneVehicle 图像比 VEDAI 大，按 2 跑）
BATCH_SIZE = 2

# 额外命令行参数，目前只用来固定 batch size
EXTRA_ARGS = ["--cfg-options", "train_dataloader.batch_size={0}".format(BATCH_SIZE)]

# 数据目录：训练读的是转换后的 DMM 格式（不是 data/DroneVehicle 原始目录）
# 留空 = 仓库内的 data/dv
# 想放到仓库外（DroneVehicle 转换后约 21 GB）就填绝对路径，例如：
#     DATA_ROOT = r"E:\datasets\dronevehicle"
DATA_ROOT = r""
DEFAULT_DATA_ROOT = os.path.join("data", "dv")

# 本文件所在目录就是仓库根目录
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def data_root_path():
    raw = DATA_ROOT or DEFAULT_DATA_ROOT
    return os.path.abspath(os.path.join(REPO_ROOT, raw))


def main():
    python = CONDA_PYTHON if os.path.isfile(CONDA_PYTHON) else sys.executable
    train_script = os.path.join(REPO_ROOT, "code", "tools", "train.py")

    root = data_root_path()
    # 让子进程里的配置读到同一个目录
    os.environ["DMM_DV_ROOT"] = root

    if not os.path.isdir(root):
        raise SystemExit(
            "找不到 DMM 格式数据：{0}。DroneVehicle 的划分不用重做，\n"
            "只需把已有的划分转成 DMM 格式：\n"
            "  python code/tools/data_process.py --dataset dronevehicle "
            "--src data/DroneVehicle --dst \"{0}\"\n"
            "  python code/tools/VOC2DOTA.py --dataset dronevehicle "
            "--src data/DroneVehicle --dst \"{0}\"".format(root)
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
    print("数据目录   :", root)
    print("batch size :", BATCH_SIZE)
    print("完整命令   :", " ".join(command))
    print("提示       : 每 50 次迭代打印一行，按 Ctrl+C 可中断；")
    print("             中断后加 --resume 参数可接着训练。")
    print("-" * 78, flush=True)
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
