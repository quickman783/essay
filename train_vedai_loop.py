"""循环训练 VEDAI：每轮 12 个 epoch，逐轮降低学习率，并记录每轮指标。

在 VS Code 里打开本文件，点右上角的 ▶ 运行按钮即可。

每一轮的完整流程：
    1. 训练 12 个 epoch（每个 epoch 结束都会在验证集上算一次指标）
    2. 挑出本轮验证集 mAP 最高的权重，复制成带时间戳的文件名
    3. 在**验证集**上评估一遍，得到 mAP 与每个类别的 AP
    4. 在**测试集**上评估一遍，得到 mAP 与每个类别的 AP
    5. 把这一轮的所有指标追加到同一个 xlsx 表格里

权重不会被覆盖，也不会被删除：
    * 每次启动都会新建 work_dirs/vedai_loop/run_<启动时间>/ 目录
    * 每个 epoch 的权重都保留（不自动清理），文件名带本次运行的时间戳
    * 每轮额外复制一份 best_mAP<分数>_epoch<轮次>_<时间戳>.pth 作为该轮最优
    * 脚本从不删除任何 .pth，清理只能由你手动完成

表格位置：work_dirs/vedai_metrics.xlsx（旁边的 .csv 是数据源，内容相同）
始终是同一个文件，每轮往后追加一行，不会新建表格。

学习率序列（默认）：
    第 1 轮 1e-4 → 第 2 轮 5e-5 → 第 3 轮 2.5e-5 → … 降到 1e-6 为止
    每一轮内部仍是原配置：前 1000 次迭代线性预热，第 8、11 epoch 各乘 0.1

跨次运行续训（重启也不会把学习率打回 1e-4）：
    * 每轮跑完会写一次 work_dirs/vedai_loop/run_state.json
    * 里面记录 next_cycle（下一轮编号）、best_map、best_ckpt（历史最佳权重）
    * 再次启动脚本时自动读它：从 next_cycle 继续、用 best_ckpt 当起点，
      于是学习率按轮次继续下降，不会重新回到 1e-4
    * 只有本轮验证集 mAP 超过历史最佳，才把本轮权重视为新起点；
      没超过就仍从历史最佳继续，避免越训越差
    * 想彻底从头开始：python train_vedai_loop.py --reset
      （旧状态文件会改名另存为 run_state_<时间戳>.json，不会被删）

什么时候停：
    1. 手动：在仓库根目录建一个名为 STOP 的空文件，当前轮跑完后停止
    2. 提前收敛：连续 PATIENCE 轮验证集 mAP 没有提升就自动停
    3. 轮数上限：跑满 MAX_CYCLES 轮（0 表示不限）
    4. Ctrl+C：立即中断
"""

import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

# 装了训练依赖的解释器（路径不存在时自动退回当前解释器）
CONDA_PYTHON = r"C:\Users\feng1\anaconda3\envs\paper310\python.exe"

# 训练配置：模型 + 数据 + 12 epoch 的训练策略
CONFIG = os.path.join("code", "configs", "dmm", "dmm_s2anet_1x_vedai.py")

# 所有轮次的输出根目录；每次启动会在其下新建 run_<启动时间> 子目录
WORK_ROOT = os.path.join("work_dirs", "vedai_loop")

# 本次启动的唯一标识（精确到秒），会写进目录名和权重文件名，
# 保证多次运行的结果互不覆盖
RUN_ID = time.strftime("%Y%m%d_%H%M%S")

# 跨次运行的训练状态：记录下一轮的轮次编号、历史最佳权重、学习率等。
# 重启脚本时会读它，从而接着上次的轮次和学习率继续，不会退回 1e-4。
STATE_FILE = os.path.join(WORK_ROOT, "run_state.json")

# 没有状态文件时（第一次运行）使用的起始权重，留空则从头训练。
# 这里用测试集表现最好的那份（test mAP 0.7043，验证集 0.7516）
BOOTSTRAP_CHECKPOINT = os.path.join(
    "")
# 上一轮是第 7 轮，所以从这里起步算作第 8 轮
# （学习率按 1e-4 × 0.5^7 = 7.8e-7 算，但被 MIN_LR 兜到下限 1e-6）
BOOTSTRAP_CYCLE = 1

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


def load_module(name, path):
    """按文件路径加载同仓库里的工具模块（不依赖 sys.path，Pylance 也能解析）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


excel_log = load_module("excel_log", os.path.join(REPO_ROOT, "code", "scripts", "excel_log.py"))


def to_posix(path):
    """mmengine 解析 --cfg-options 时对反斜杠不友好，统一改成正斜杠。"""
    return os.path.abspath(path).replace("\\", "/")


def cycle_lr(cycle):
    """第 cycle 轮（从 1 开始）的峰值学习率。"""
    return max(BASE_LR * (LR_DECAY ** (cycle - 1)), MIN_LR)


def epoch_map_from_log(work_dir):
    """解析训练日志，返回 {epoch: 验证集 mAP}。"""
    pattern = re.compile(r"Epoch\(val\) \[(\d+)\]\[[^\]]*\]\s+dota/mAP:\s*([0-9.]+)")
    result = {}
    for path in sorted(glob.glob(os.path.join(work_dir, "**", "*.log"), recursive=True)):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = pattern.search(line)
                    if match:
                        epoch = int(match.group(1))
                        value = float(match.group(2))
                        result[epoch] = max(result.get(epoch, -1.0), value)
        except OSError:
            continue
    return result


def save_best_copy(work_dir, run_id):
    """把本轮验证集 mAP 最高的 epoch 权重另存一份带时间戳的副本。

    只复制、不移动、不删除：原始 epoch 权重全部留在原处，
    目标文件已存在时直接复用，绝不覆盖。

    Returns:
        tuple(str | None, float | None): 副本路径与该轮的验证集 mAP
    """
    epoch_map = epoch_map_from_log(work_dir)
    pattern = re.compile(r"epoch(\d+)_")
    epoch_files = {}
    for path in glob.glob(os.path.join(work_dir, "epoch*_{0}.pth".format(run_id))):
        match = pattern.match(os.path.basename(path))
        if match:
            epoch_files[int(match.group(1))] = path

    if not epoch_files:
        return None, None

    best_epoch = max(epoch_map, key=epoch_map.get) if epoch_map else max(epoch_files)
    if best_epoch not in epoch_files:
        best_epoch = max(epoch_files)
    best_value = epoch_map.get(best_epoch)

    if best_value is None:
        destination = os.path.join(
            work_dir, "last_epoch{0}_{1}.pth".format(best_epoch, run_id))
    else:
        destination = os.path.join(
            work_dir, "best_mAP{0:.4f}_epoch{1}_{2}.pth".format(best_value, best_epoch, run_id))

    if not os.path.exists(destination):
        shutil.copy2(epoch_files[best_epoch], destination)
    return destination, best_value


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
    print("本次权重目录     : {0}".format(os.path.join(WORK_ROOT, "run_{0}".format(RUN_ID))))
    print("脚本没有删除任何权重，清理请自行手动完成。")


def load_state():
    """读取跨次运行的训练状态；文件不存在或损坏时返回 None。"""
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            raise ValueError("状态文件内容不是字典")
        return state
    except (OSError, ValueError) as exc:
        print("状态文件读取失败，本次当作首次运行：{0}".format(exc))
        return None


def save_state(state):
    """把训练状态写回文件（先写临时文件再原子替换，避免写坏）。"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temporary = STATE_FILE + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, STATE_FILE)


def backup_state():
    """把现有状态文件改名存档（不删除），用于 --reset。"""
    if not os.path.isfile(STATE_FILE):
        return
    backup = os.path.join(WORK_ROOT, "run_state_{0}.json".format(RUN_ID))
    os.replace(STATE_FILE, backup)
    print("旧状态文件已另存为：{0}".format(os.path.relpath(backup, REPO_ROOT)))


def check_splits():
    """启动前看一眼三个划分，重复/为空时提醒（不阻断，免得挡住正常训练）。"""
    names = {}
    for split in ("train", "val", "test"):
        folder = os.path.join(REPO_ROOT, "datasets", "vedai", split, "annofiles_vi")
        if not os.path.isdir(folder):
            print("警告       : 找不到 {0}，请先运行 rebuild_vedai.py".format(folder))
            return
        names[split] = {os.path.splitext(n)[0] for n in os.listdir(folder)}

    empty = [split for split, ids in names.items() if not ids]
    if empty:
        print("警告       : {0} 划分没有任何标注，请先运行".format("/".join(empty)))
        print("             python code/scripts/rebuild_vedai.py --no-split")
        return

    shared = names["val"] & names["test"]
    if shared:
        print("警告       : val 与 test 有 {0} 张重复（val {1} 张 / test {2} 张），".format(
            len(shared), len(names["val"]), len(names["test"])))
        print("             表格里的 test_mAP 会和 val_mAP 几乎一样。修复：")
        print("             python code/scripts/rebuild_vedai.py --no-split")

    leaked = (names["train"] & names["val"]) | (names["train"] & names["test"])
    if leaked:
        print("警告       : train 与 val/test 有 {0} 张重复（数据泄漏），建议重建：".format(
            len(leaked)))
        print("             python code/scripts/rebuild_vedai.py --no-split")


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
    print("本次运行号 :", RUN_ID)
    print("训练输出   :", os.path.join(WORK_ROOT, "run_{0}".format(RUN_ID)))
    print("指标表格   :", METRICS_XLSX)
    print("训练状态   :", STATE_FILE, "（重启后自动接着训练，不会退回 1e-4）")
    print("权重策略   : 保留全部 epoch 权重，文件名带时间戳，脚本不删除任何 .pth")
    print("学习率     : 首轮 {0:.2e}，每轮 ×{1:g}，下限 {2:.2e}".format(BASE_LR, LR_DECAY, MIN_LR))
    print("每轮流程   : 训练 12 epoch → 验证集评估 → 测试集评估 → 写入表格")
    print("停止方式   : 在仓库根目录新建 STOP 文件；或按 Ctrl+C")
    if PATIENCE:
        print("提前收敛   : 连续 {0} 轮验证集 mAP 不提升则自动停止".format(PATIENCE))
    check_splits()
    print("-" * 78, flush=True)

    run_dir = os.path.join(WORK_ROOT, "run_{0}".format(RUN_ID))

    history = []
    best = None
    best_map = -1.0
    stale = 0
    failures = 0

    # ---- 改动 1+2：读状态文件，决定从第几轮、用哪份权重、什么学习率继续 ----
    state = None
    if "--reset" in sys.argv[1:]:
        backup_state()
    else:
        state = load_state()

    cycle = 1
    init_checkpoint = None
    resumed = False
    if state:
        resume_ckpt = state.get("best_ckpt") or ""
        if resume_ckpt and os.path.isfile(resume_ckpt):
            resumed = True
            cycle = max(1, int(state.get("next_cycle", 1)))
            init_checkpoint = resume_ckpt
            best_map = float(state.get("best_map", -1.0))
            stale = int(state.get("stale_cycles", 0))
            for record in state.get("cycles") or []:
                history.append((
                    int(record.get("cycle", 0)),
                    float(record.get("lr", 0.0)),
                    record.get("val_map"),
                    record.get("test_map"),
                ))
            if best_map > 0:
                best = (int(state.get("best_cycle", 0)),
                        float(state.get("best_lr", 0.0)),
                        best_map,
                        state.get("best_test_map"),
                        resume_ckpt)
            print("续训状态   : 第 {0} 轮继续，峰值学习率 {1:.2e}".format(cycle, cycle_lr(cycle)))
            print("起始权重   : {0}".format(os.path.relpath(resume_ckpt, REPO_ROOT)))
            if best_map > 0:
                print("历史最佳   : 验证集 mAP {0:.4f}".format(best_map))
        else:
            print("续训状态   : 状态文件里的最佳权重已不存在，改为从头训练")
    if not resumed:
        if BOOTSTRAP_CHECKPOINT:
            bootstrap = os.path.join(REPO_ROOT, BOOTSTRAP_CHECKPOINT)
            if not os.path.isfile(bootstrap):
                # 宁可报错也不从随机初始化开始，免得白跑一整轮
                raise SystemExit(
                    "指定的起始权重不存在：{0}\n"
                    "如果确实想从头训练，请把脚本顶部的 BOOTSTRAP_CHECKPOINT 改成空字符串。".format(
                        BOOTSTRAP_CHECKPOINT))
            cycle = BOOTSTRAP_CYCLE
            init_checkpoint = bootstrap
            print("续训状态   : 没有状态文件，从 {0} 起步（算作第 {1} 轮）".format(
                os.path.basename(bootstrap), cycle))
        else:
            print("续训状态   : 没有状态文件也没有起始权重，从随机初始化开始")

    try:
        while True:
            if MAX_CYCLES and cycle > MAX_CYCLES:
                print("\n已达到设定的最大轮数 {0}，结束。".format(MAX_CYCLES))
                break
            if os.path.isfile(STOP_FILE):
                print("\n检测到 STOP 文件，本轮结束后停止。")
                break

            lr = cycle_lr(cycle)
            work_dir = os.path.join(run_dir, "cycle_{0}".format(cycle))
            os.makedirs(os.path.join(REPO_ROOT, work_dir), exist_ok=True)

            overrides = [
                "optim_wrapper.optimizer.lr={0}".format(lr),
                # -1：保留全部 epoch 权重，不自动清理旧的
                "default_hooks.checkpoint.max_keep_ckpts=-1",
                # 关掉 save_best —— 它会在出现新最优时删掉上一个 best 文件，
                # 改由下面的 save_best_copy 另存副本
                "default_hooks.checkpoint.save_best=None",
                # 权重文件名带上本次运行的时间戳，避免跨次运行重名
                "default_hooks.checkpoint.filename_tmpl=epoch{{}}_{0}.pth".format(RUN_ID),
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

            checkpoint, best_val = save_best_copy(os.path.join(REPO_ROOT, work_dir), RUN_ID)
            if checkpoint is None:
                print("\n第 {0} 轮没有产出权重文件，结束循环。".format(cycle))
                break
            checkpoint = os.path.abspath(checkpoint)
            print("\n本轮最优权重已另存：{0}".format(os.path.relpath(checkpoint, REPO_ROOT)))
            if best_val is not None:
                print("  （训练日志里的验证集 mAP {0:.4f}）".format(best_val))

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

            # ---- 改动 3：只有验证集 mAP 变好才把它当作下次的起点 ----
            state = state or {}
            previous_best = float(state.get("best_map") or -1.0)
            improved = val_map is not None and val_map > previous_best

            state.setdefault("cycles", []).append({
                "cycle": cycle,
                "lr": lr,
                "val_map": val_map,
                "test_map": test_map,
                "checkpoint": os.path.relpath(checkpoint, REPO_ROOT),
                "run_id": RUN_ID,
            })
            if improved:
                state["best_map"] = val_map
                state["best_ckpt"] = checkpoint
                state["best_cycle"] = cycle
                state["best_lr"] = lr
                state["best_test_map"] = test_map
                state["stale_cycles"] = 0
            else:
                state["stale_cycles"] = stale
                if val_map is not None and previous_best > 0:
                    # 本轮没变好：下次仍从历史最佳权重继续，避免越训越差
                    print("  本轮不如历史最佳 {0:.4f}，下次从历史最佳权重继续".format(previous_best))
            state["next_cycle"] = cycle + 1
            runs = list(state.get("runs") or [])
            if RUN_ID not in runs:
                runs.append(RUN_ID)
            state["runs"] = runs
            save_state(state)
            print("  训练状态已保存：{0}".format(STATE_FILE))

            init_checkpoint = state.get("best_ckpt") or checkpoint
            cycle += 1

            if PATIENCE and stale >= PATIENCE:
                print("  连续 {0} 轮未提升，提前停止。".format(PATIENCE))
                break
            time.sleep(2)
    finally:
        print_summary(history, best)

    return 0


if __name__ == "__main__":
    sys.exit(main())
