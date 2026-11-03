"""循环训练 VEDAI：每轮 12 个 epoch，逐轮降低学习率，并记录每轮指标。

在 VS Code 里打开本文件，点右上角的 ▶ 运行按钮即可。

每一轮的完整流程：
    1. 训练 12 个 epoch（每个 epoch 结束都会在验证集上算一次指标）
    2. 挑出本轮最好的权重
    3. 在**验证集**上评估一遍，得到 mAP 与每个类别的 AP
    4. 在**测试集**上评估一遍，得到 mAP 与每个类别的 AP
    5. 把这一轮的所有指标追加到同一个 xlsx 表格里

表格位置：work_dirs/vedai_metrics.xlsx（旁边的 .csv 是数据源，内容相同）
始终是同一个文件，每轮往后追加一行，不会新建表格。

学习率序列（默认）：
    第 1 轮 1e-4 → 第 2 轮 5e-5 → 第 3 轮 2.5e-5 → … 降到 1e-6 为止
    每一轮内部仍是原配置：前 1000 次迭代线性预热，第 8、11 epoch 各乘 0.1

什么时候停：
    1. 手动：在仓库根目录建一个名为 STOP 的空文件，当前轮跑完后停止
    2. 提前收敛：连续 PATIENCE 轮验证集 mAP 没有提升就自动停
    3. 轮数上限：跑满 MAX_CYCLES 轮（0 表示不限）
    4. Ctrl+C：立即中断
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

# 装了训练依赖的解释器（路径不存在时自动退回当前解释器）
CONDA_PYTHON = r"C:\Users\feng1\anaconda3\envs\paper310\python.exe"

# 训练配置：模型 + 数据 + 12 epoch 的训练策略
CONFIG = os.path.join("code", "configs", "dmm", "dmm_s2anet_1x_vedai.py")

# 所有轮次的输出根目录，每一轮一个子目录 cycle_1、cycle_2 …
WORK_ROOT = os.path.join("work_dirs", "vedai_loop")

# 所有轮次的指标都写进这一个表格
METRICS_XLSX = os.path.join("work_dirs", "vedai_metrics.xlsx")

# 第一轮的峰值学习率（与论文配置一致）
BASE_LR = 1e-4
# 每轮学习率乘以这个系数（0.5 = 每轮减半）
LR_DECAY = 0.5
# 学习率下限，降到这个值以下就不再继续减
MIN_LR = 1e-6
# 最多跑多少轮，0 表示不限制
MAX_CYCLES = 0
# 连续多少轮验证集 mAP 没有提升就提前停止，0 表示不启用
PATIENCE = 3
# 单轮训练失败（非 Ctrl+C）后最多重试几次
MAX_FAILURES = 3

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
STOP_FILE = os.path.join(REPO_ROOT, "STOP")

sys.path.insert(0, os.path.join(REPO_ROOT, "code", "scripts"))
import excel_log  # noqa: E402


def to_posix(path):
    """mmengine 解析 --cfg-options 时对反斜杠不友好，统一改成正斜杠。"""
    return os.path.abspath(path).replace("\\", "/")


def cycle_lr(cycle):
    """第 cycle 轮（从 1 开始）的峰值学习率。"""
    return max(BASE_LR * (LR_DECAY ** (cycle - 1)), MIN_LR)


def find_checkpoint(work_dir):
    """挑出本轮最值得用于下一轮初始化的权重：优先验证最优，其次最后一轮。"""
    best = glob.glob(os.path.join(work_dir, "best_dota_mAP_epoch_*.pth"))
    if best:
        def epoch_of(path):
            match = re.search(r"_epoch_(\d+)\.pth$", path)
            return int(match.group(1)) if match else -1
        return max(best, key=epoch_of)
    last = os.path.join(work_dir, "epoch_12.pth")
    return last if os.path.isfile(last) else None


def run_eval(python, checkpoint, split, work_dir):
    """在指定划分上评估权重，返回 {指标名: 数值}；失败返回 None。"""
    out_json = os.path.join(work_dir, "{0}_metrics.json".format(split))
    command = [
        python,
        os.path.join(REPO_ROOT, "code", "scripts", "evaluate.py"),
        "--config", os.path.join(REPO_ROOT, CONFIG),
        "--checkpoint", checkpoint,
        "--split", split,
        "--out", out_json,
        "--work-dir", os.path.join(work_dir, "_eval_{0}".format(split)),
    ]
    label = "验证集" if split == "val" else "测试集"
    print("\n--- 在{0}上评估 ---".format(label), flush=True)
    try:
        returncode = subprocess.call(command, cwd=REPO_ROOT)
    except KeyboardInterrupt:
        print("\n评估被中断。")
        return None
    if returncode != 0 or not os.path.isfile(out_json):
        print("评估失败（退出码 {0}）".format(returncode))
        return None
    with open(out_json, encoding="utf-8") as handle:
        return json.load(handle)


def build_header(val_metrics, test_metrics):
    classes = []
    for metrics in (val_metrics, test_metrics):
        if not metrics:
            continue
        for key in metrics:
            if key.startswith("AP_") and key[len("AP_"):] not in classes:
                classes.append(key[len("AP_"):])
    header = ["轮次", "峰值学习率", "训练耗时(分钟)", "val_mAP", "val_AP50", "test_mAP", "test_AP50"]
    header += ["val_AP_{0}".format(name) for name in classes]
    header += ["test_AP_{0}".format(name) for name in classes]
    header.append("检查点")
    return header


def build_values(cycle, lr, minutes, val_metrics, test_metrics, checkpoint):
    values = {
        "轮次": cycle,
        "峰值学习率": lr,
        "训练耗时(分钟)": round(minutes, 1),
        "val_mAP": (val_metrics or {}).get("mAP", ""),
        "val_AP50": (val_metrics or {}).get("AP50", ""),
        "test_mAP": (test_metrics or {}).get("mAP", ""),
        "test_AP50": (test_metrics or {}).get("AP50", ""),
        "检查点": os.path.relpath(checkpoint, REPO_ROOT),
    }
    for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
        for key, value in (metrics or {}).items():
            if key.startswith("AP_"):
                values["{0}_{1}".format(prefix, key)] = value
    return values


def print_summary(history, best):
    print()
    print("=" * 78)
    print("循环训练结果汇总")
    print("=" * 78)
    print("{0:<8}{1:>14}{2:>12}{3:>12}".format("轮次", "峰值学习率", "验证 mAP", "测试 mAP"))
    for cycle, lr, val_map, test_map in history:
        print("{0:<8}{1:>14.2e}{2:>12}{3:>12}".format(
            cycle, lr,
            "失败" if val_map is None else "{0:.4f}".format(val_map),
            "失败" if test_map is None else "{0:.4f}".format(test_map)))

    if best:
        cycle, lr, val_map, test_map, checkpoint = best
        print()
        print("最佳轮次   : 第 {0} 轮（按验证集 mAP 选，峰值学习率 {1:.2e}）".format(cycle, lr))
        print("验证集 mAP : {0:.4f}".format(val_map))
        print("测试集 mAP : {0}".format("未取到" if test_map is None else "{0:.4f}".format(test_map)))
        print("推荐权重   : {0}".format(os.path.relpath(checkpoint, REPO_ROOT)))
    print()
    print("全部指标已记录在 : {0}".format(METRICS_XLSX))
    print("（同目录下的同名 .csv 内容相同，可作备份）")


def main():
    python = CONDA_PYTHON if os.path.isfile(CONDA_PYTHON) else sys.executable
    train_script = os.path.join(REPO_ROOT, "code", "tools", "train.py")

    if not os.path.isdir(os.path.join(REPO_ROOT, "datasets", "vedai")):
        raise SystemExit(
            "找不到 datasets/vedai，请先运行 python code/scripts/rebuild_vedai.py 准备数据。"
        )

    os.makedirs(os.path.join(REPO_ROOT, WORK_ROOT), exist_ok=True)
    if os.path.isfile(STOP_FILE):
        os.remove(STOP_FILE)

    print("仓库根目录 :", REPO_ROOT)
    print("解释器     :", python)
    print("训练输出   :", WORK_ROOT)
    print("指标表格   :", METRICS_XLSX)
    print("学习率     : 首轮 {0:.2e}，每轮 ×{1:g}，下限 {2:.2e}".format(BASE_LR, LR_DECAY, MIN_LR))
    print("每轮流程   : 训练 12 epoch → 验证集评估 → 测试集评估 → 写入表格")
    print("停止方式   : 在仓库根目录新建 STOP 文件；或按 Ctrl+C")
    if PATIENCE:
        print("提前收敛   : 连续 {0} 轮验证集 mAP 不提升则自动停止".format(PATIENCE))
    print("-" * 78, flush=True)

    history = []
    best = None
    best_map = -1.0
    stale = 0
    failures = 0
    cycle = 1
    init_checkpoint = None

    try:
        while True:
            if MAX_CYCLES and cycle > MAX_CYCLES:
                print("\n已达到设定的最大轮数 {0}，结束。".format(MAX_CYCLES))
                break
            if os.path.isfile(STOP_FILE):
                print("\n检测到 STOP 文件，本轮结束后停止。")
                break

            lr = cycle_lr(cycle)
            work_dir = os.path.join(WORK_ROOT, "cycle_{0}".format(cycle))
            os.makedirs(os.path.join(REPO_ROOT, work_dir), exist_ok=True)

            overrides = [
                "optim_wrapper.optimizer.lr={0}".format(lr),
                "default_hooks.checkpoint.max_keep_ckpts=2",
            ]
            if init_checkpoint:
                overrides.append("load_from={0}".format(to_posix(init_checkpoint)))

            command = [
                python, train_script,
                os.path.join(REPO_ROOT, CONFIG),
                "--work-dir", os.path.join(REPO_ROOT, work_dir),
                "--cfg-options", *overrides,
            ]

            source = ("从 {0} 继续".format(os.path.basename(init_checkpoint))
                      if init_checkpoint else "从头训练")
            print()
            print("=" * 78)
            print("第 {0} 轮开始：12 个 epoch，峰值学习率 {1:.2e}，{2}".format(cycle, lr, source))
            print("=" * 78, flush=True)

            started = time.time()
            try:
                returncode = subprocess.call(command, cwd=REPO_ROOT)
            except KeyboardInterrupt:
                print("\n\n已中断，正在退出循环。")
                break

            elapsed = time.time() - started
            if returncode != 0:
                failures += 1
                history.append((cycle, lr, None, None))
                print("\n第 {0} 轮训练失败（退出码 {1}），已连续失败 {2}/{3} 次。".format(
                    cycle, returncode, failures, MAX_FAILURES))
                if failures >= MAX_FAILURES:
                    print("失败次数达到上限，结束循环。")
                    break
                time.sleep(30)
                continue
            failures = 0

            checkpoint = find_checkpoint(os.path.join(REPO_ROOT, work_dir))
            if checkpoint is None:
                print("\n第 {0} 轮没有产出权重文件，结束循环。".format(cycle))
                break
            checkpoint = os.path.abspath(checkpoint)

            val_metrics = run_eval(python, checkpoint, "val", os.path.join(REPO_ROOT, work_dir))
            test_metrics = run_eval(python, checkpoint, "test", os.path.join(REPO_ROOT, work_dir))

            header = build_header(val_metrics, test_metrics)
            values = build_values(cycle, lr, elapsed / 60, val_metrics, test_metrics, checkpoint)
            xlsx_path = excel_log.append_row(
                os.path.join(REPO_ROOT, METRICS_XLSX), header, values)

            val_map = (val_metrics or {}).get("mAP")
            test_map = (test_metrics or {}).get("mAP")
            history.append((cycle, lr, val_map, test_map))

            print("\n第 {0} 轮完成，用时 {1:.0f} 分钟".format(cycle, elapsed / 60))
            print("  验证集 mAP : {0}".format("读取失败" if val_map is None else "{0:.4f}".format(val_map)))
            print("  测试集 mAP : {0}".format("读取失败" if test_map is None else "{0:.4f}".format(test_map)))
            print("  已写入表格 : {0}".format(os.path.relpath(xlsx_path, REPO_ROOT)))

            if val_map is not None:
                if val_map > best_map:
                    best_map = val_map
                    best = (cycle, lr, val_map, test_map, checkpoint)
                    stale = 0
                else:
                    stale += 1
                    print("  已连续 {0} 轮验证集 mAP 没有提升（当前最佳 {1:.4f}）".format(stale, best_map))
                    if PATIENCE and stale >= PATIENCE:
                        print("  连续 {0} 轮未提升，提前停止。".format(PATIENCE))
                        break

            init_checkpoint = checkpoint
            cycle += 1
            time.sleep(2)
    finally:
        print_summary(history, best)

    return 0


if __name__ == "__main__":
    sys.exit(main())
