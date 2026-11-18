"""MBO cycle engine: 12 epochs, checkpoints, val/test and CSV/XLSX recording.

Relocated from the shared author loop; model/loss/schedule behavior is preserved.
"""

import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

# 装了训练依赖的解释器（路径不存在时自动退回当前解释器）
CONDA_PYTHON = os.environ.get("DMM_PYTHON", "")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROJECT_ROOT = os.path.join(REPO_ROOT, "code")
# The essay repository uses code/mmrotate; the existing local workspace keeps it in DMM-main.
if not os.path.isfile(os.path.join(PROJECT_ROOT, "mmrotate", "__init__.py")):
    PROJECT_ROOT = os.path.join(PROJECT_ROOT, "DMM-main")
os.environ["DMM_BASE_ROOT"] = PROJECT_ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
STOP_FILE = os.path.join(REPO_ROOT, "STOP")

# 第一轮的峰值学习率；每轮 ×LR_DECAY，最低 MIN_LR
BASE_LR = 1e-4
LR_DECAY = 0.5
MIN_LR = 1e-6
# 最多跑多少轮，0 表示不限制
MAX_CYCLES = 0
# 连续多少轮验证集 mAP 没有提升就提前停止，0 表示不启用
PATIENCE = 3
# 单轮训练失败（非 Ctrl+C）后最多重试几次
MAX_FAILURES = 3
# 每轮训练多少个 epoch（与基配置的 max_epochs 一致，仅用于打印）
EPOCHS_PER_CYCLE = 12

# 环境变量留空表示用默认的 data_root，由脚本按 profiles 里的默认值填
DATA_ROOT = ""

# 没有状态文件时用的起始权重（留空 = 自动挑 weights_dir 里最新的 .pth）
BOOTSTRAP_CHECKPOINT = ""
# 上面那份权重相当于已经跑完了第几轮，下一轮按这个编号算学习率
BOOTSTRAP_CYCLE = 1

PROFILES = {}  # Filled by dmm_mbo.train_loop.


def load_module(name, path):
    """按文件路径加载同仓库里的工具模块（不依赖 sys.path）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


excel_log = load_module(
    "excel_log", os.path.join(HERE, "excel_log.py"))


def to_posix(path):
    """mmengine 解析 --cfg-options 时对反斜杠不友好，统一改成正斜杠。"""
    return os.path.abspath(path).replace("\\", "/")


def cycle_lr(cycle):
    """第 cycle 轮（从 1 开始）的峰值学习率。"""
    return max(BASE_LR * (LR_DECAY ** (cycle - 1)), MIN_LR)


def parse_cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="DMM+DCFL+MBO training loop. One cycle = 12 epochs.")
    parser.add_argument("--reset", action="store_true", help="Archive old loop state and start a new sequence")
    parser.add_argument("--init", default="", help="Full checkpoint from this combined model; not a VMamba-only checkpoint")
    parser.add_argument("--cycle", type=int, help="First cycle number when no saved loop state is used")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="Absolute cycle-number limit (0: unlimited, author's default); use 1 for a fresh 12-epoch run")
    parser.add_argument("--output-root", default="", help="Optional output parent on a disk with sufficient space")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Initialize VMamba backbones from ImageNet; separate output/state")
    parser.add_argument("--keep-epochs", type=int, default=-1,
                        help="Keep latest N regular checkpoints plus validation best; -1 retains all")
    parser.add_argument("--dry-run", action="store_true", help="Read-only path/data check; do not train, evaluate or create output")
    args = parser.parse_args()
    if args.cycle is not None and args.cycle < 1:
        parser.error("--cycle must be >= 1")
    if args.max_cycles is not None and args.max_cycles < 0:
        parser.error("--max-cycles must be >= 0")
    if args.keep_epochs != -1 and args.keep_epochs < 1:
        parser.error("--keep-epochs must be -1 or >= 1")
    return vars(args)


def data_root_path(profile):
    raw = os.environ.get(profile["env_var"]) or DATA_ROOT or profile["default_data_root"]
    return os.path.abspath(os.path.join(REPO_ROOT, raw))


def _digest(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def only_name_collision(root, left, right, shared, sample=20):
    """同名文件是不是同一张图。

    DroneVehicle 的三个划分各自从 00001 编号，名字大量重合但内容不同，
    所以抽几张比对内容摘要；全都不同说明只是编号撞车，不是数据泄漏。
    """
    checked = 0
    for name in sorted(shared)[:sample]:
        left_path = os.path.join(root, left, "image_vi", name + ".png")
        right_path = os.path.join(root, right, "image_vi", name + ".png")
        if not (os.path.isfile(left_path) and os.path.isfile(right_path)):
            return False
        if _digest(left_path) == _digest(right_path):
            return False
        checked += 1
    return checked > 0


def prepare_data(profile):
    """检查数据目录并把路径写进环境变量，子进程（train/evaluate）读同一个值。"""
    root = data_root_path(profile)
    os.environ[profile["env_var"]] = root

    missing = [split for split in ("train", "val", "test")
               if not os.path.isdir(os.path.join(root, split, "annofiles_vi"))]
    if missing:
        raise SystemExit(
            "找不到 {0} 的数据：{1}（缺 {2}）。".format(
                profile["label"], root, "/".join(missing)))

    names = {}
    for split in ("train", "val", "test"):
        folder = os.path.join(root, split, "annofiles_vi")
        names[split] = {os.path.splitext(n)[0] for n in os.listdir(folder)}

    print("数据目录   : {0}".format(root))
    print("数据规模   : train {0} / val {1} / test {2} 张".format(
        len(names["train"]), len(names["val"]), len(names["test"])))

    empty = [split for split, ids in names.items() if not ids]
    if empty:
        raise SystemExit("{0} 划分没有任何标注文件，请检查数据。".format(
            "/".join(empty)))

    shared = names["val"] & names["test"]
    if shared:
        if only_name_collision(root, "val", "test", shared):
            print("说明       : val 与 test 有 {0} 个同名文件，抽检内容完全不同".format(
                len(shared)))
        else:
            print("警告       : val 与 test 有 {0} 张真重复，test_mAP 会失真".format(
                len(shared)))

    leaked = (names["train"] & names["val"]) | (names["train"] & names["test"])
    if leaked:
        if only_name_collision(root, "train", "test", leaked):
            print("说明       : train 与 val/test 有 {0} 个同名文件，抽检内容完全不同".format(
                len(leaked)))
        else:
            print("警告       : train 与 val/test 有 {0} 张真重复（数据泄漏）".format(
                len(leaked)))


def find_bootstrap(profile, explicit):
    """找起始权重，返回 (绝对路径 或 None, 是否来自 weights 目录自动挑)。"""
    if explicit:
        path = os.path.abspath(os.path.join(REPO_ROOT, explicit))
        if not os.path.isfile(path):
            raise SystemExit("--init 指定的权重不存在：{0}".format(path))
        return path, False

    if BOOTSTRAP_CHECKPOINT:
        path = os.path.join(REPO_ROOT, BOOTSTRAP_CHECKPOINT)
        if not os.path.isfile(path):
            raise SystemExit(
                "BOOTSTRAP_CHECKPOINT 指定的权重不存在：{0}\n"
                "想从头训练请把它改成空字符串。".format(BOOTSTRAP_CHECKPOINT))
        return os.path.abspath(path), False

    folder = os.path.join(REPO_ROOT, profile["weights_dir"])
    if not os.path.isdir(folder):
        return None, False
    candidates = [p for p in glob.glob(
        os.path.join(folder, "**", "*.pth"), recursive=True) if os.path.isfile(p)]
    if not candidates:
        return None, False
    return os.path.abspath(max(candidates, key=os.path.getmtime)), True


def epoch_map_from_log(work_dir):
    """解析训练日志，返回 {epoch: 验证集 mAP}。"""
    pattern = re.compile(
        r"Epoch\(val\) \[(\d+)\]\[[^\]]*\]\s+dota/mAP:\s*([0-9.]+)")
    result = {}
    for path in sorted(glob.glob(
            os.path.join(work_dir, "**", "*.log"), recursive=True)):
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


def save_best_copy(work_dir, run_id, retained_only=False):
    """把本轮验证集 mAP 最高的 epoch 权重另存一份带时间戳的副本。

    只复制、不移动、不删除；目标已存在时直接复用，绝不覆盖。
    """
    epoch_map = epoch_map_from_log(work_dir)
    if retained_only:
        # CheckpointHook preserves the validation best even after regular files expire.
        candidates = glob.glob(os.path.join(work_dir, "best_dota_mAP_epoch*.pth"))
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one validation-best checkpoint: " + repr(candidates))
        match = re.search(r"_epoch_?(\d+)(?:_|\.pth)", os.path.basename(candidates[0]))
        if match is None or int(match.group(1)) not in epoch_map:
            raise RuntimeError("Cannot match validation-best checkpoint to epoch log")
        return candidates[0], epoch_map[int(match.group(1))]
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
            work_dir,
            "best_mAP{0:.4f}_epoch{1}_{2}.pth".format(best_value, best_epoch, run_id))

    if not os.path.exists(destination):
        shutil.copy2(epoch_files[best_epoch], destination)
    return destination, best_value


def run_eval(python, config, checkpoint, split, work_dir):
    """在指定划分上评估权重，返回 {指标名: 数值}；失败返回 None。"""
    out_json = os.path.join(work_dir, "{0}_metrics.json".format(split))
    command = [
        python,
        os.path.join(HERE, "evaluate.py"),
        "--config", os.path.join(REPO_ROOT, config),
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
    header = ["轮次", "峰值学习率", "训练耗时(分钟)",
              "val_mAP", "val_AP50", "test_mAP", "test_AP50"]
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


def print_summary(profile, history, best, metrics_xlsx, work_root, run_id):
    print()
    print("=" * 78)
    print("循环训练结果汇总（{0} · {1}）".format(
        profile["variant"], profile["label"]))
    print("=" * 78)
    print("{0:<8}{1:>14}{2:>12}{3:>12}".format(
        "轮次", "峰值学习率", "验证 mAP", "测试 mAP"))
    for cycle, lr, val_map, test_map in history:
        print("{0:<8}{1:>14.2e}{2:>12}{3:>12}".format(
            cycle, lr,
            "失败" if val_map is None else "{0:.4f}".format(val_map),
            "失败" if test_map is None else "{0:.4f}".format(test_map)))

    if best:
        cycle, lr, val_map, test_map, checkpoint = best
        print()
        print("最佳轮次   : 第 {0} 轮（按验证集 mAP 选，峰值学习率 {1:.2e}）".format(
            cycle, lr))
        print("验证集 mAP : {0:.4f}".format(val_map))
        print("测试集 mAP : {0}".format(
            "未取到" if test_map is None else "{0:.4f}".format(test_map)))
        print("推荐权重   : {0}".format(os.path.relpath(checkpoint, REPO_ROOT)))
    print()
    print("全部指标已记录在 : {0}".format(metrics_xlsx))
    print("（同目录下的同名 .csv 内容相同，可作备份）")
    print("本次权重目录     : {0}".format(
        os.path.join(work_root, "run_{0}".format(run_id))))
    print("脚本没有删除任何权重，清理请自行手动完成。")


def load_state(state_file):
    if not os.path.isfile(state_file):
        return None
    try:
        with open(state_file, encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            raise ValueError("状态文件内容不是字典")
        return state
    except (OSError, ValueError) as exc:
        print("状态文件读取失败，本次当作首次运行：{0}".format(exc))
        return None


def save_state(state_file, state):
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temporary = state_file + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, state_file)


def backup_state(state_file, work_root, run_id):
    if not os.path.isfile(state_file):
        return
    backup = os.path.join(work_root, "run_state_{0}.json".format(run_id))
    os.replace(state_file, backup)
    print("旧状态文件已另存为：{0}".format(os.path.relpath(backup, REPO_ROOT)))


def main(dataset):
    """循环训练入口。dataset 只能是 PROFILES 里的键。"""
    if dataset not in PROFILES:
        raise SystemExit("未知数据集：{0}".format(dataset))
    profile = dict(PROFILES[dataset])
    options = parse_cli()
    cycle_limit = MAX_CYCLES if options["max_cycles"] is None else options["max_cycles"]
    if options["output_root"]:
        output_root = os.path.abspath(options["output_root"])
        profile["work_root"] = os.path.join(output_root, os.path.basename(profile["work_root"]))
        profile["metrics_xlsx"] = os.path.join(output_root, os.path.basename(profile["metrics_xlsx"]))

    python = CONDA_PYTHON if os.path.isfile(CONDA_PYTHON) else sys.executable
    train_script = os.path.join(PROJECT_ROOT, "tools", "train.py")
    config = profile["config"]
    work_root = profile["work_root"]

    if options["pretrained"]:
        checkpoint = os.path.join(PROJECT_ROOT, "weights", "vssm", "vssm_tiny_0230_ckpt_epoch_262.pth")
        if not os.path.isfile(checkpoint):
            raise SystemExit("VMamba checkpoint is missing: " + checkpoint)
    if options["dry_run"]:
        prepare_data(profile)
        assert os.path.isfile(train_script), train_script
        assert os.path.isfile(os.path.join(REPO_ROOT, config)), config
        bootstrap, _ = find_bootstrap(profile, options["init"])
        print("DRY RUN: no training, evaluation or output writes")
        print("Project:", PROJECT_ROOT)
        print("Python:", python)
        print("Config:", os.path.join(REPO_ROOT, config))
        print("Variant:", profile["variant"])
        print("Detail:", profile["detail"])
        print("Output:", os.path.join(REPO_ROOT, work_root))
        print("Metrics:", os.path.join(REPO_ROOT, profile["metrics_xlsx"]))
        print("Cycle limit:", cycle_limit, "(0 means unlimited)")
        print("Backbone pretraining:", options["pretrained"])
        print("Full-model bootstrap:", bootstrap)
        return 0
    if os.path.isfile(STOP_FILE):
        print("STOP exists; no training started. Remove it manually when ready to run.")
        return 0
    os.makedirs(os.path.join(REPO_ROOT, work_root), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, profile["weights_dir"]), exist_ok=True)

    state_file = os.path.join(REPO_ROOT, work_root, "run_state.json")
    metrics_xlsx = profile["metrics_xlsx"]

    # 状态文件必须在运行号确定后才能备份，所以先把参数解析完
    run_id = time.strftime("%Y%m%d_%H%M%S")

    prepare_data(profile)

    print("数据集     : {0} · {1}".format(profile["label"], profile["classes"]))
    print("模型       : {0}".format(profile["variant"]))
    print("构成       : {0}".format(profile["detail"]))
    print("仓库根目录 :", REPO_ROOT)
    print("解释器     :", python)
    print("训练配置   :", config)
    print("本次运行号 :", run_id)
    print("训练输出   :", os.path.join(work_root, "run_{0}".format(run_id)))
    print("指标表格   :", metrics_xlsx)
    print("训练状态   :", state_file, "（重启后接着训练，学习率不会退回 1e-4）")
    print("权重策略   : " + ("保留全部 epoch 权重" if options["keep_epochs"] == -1 else
                            "保留最新 {0} 轮及验证集最佳权重".format(options["keep_epochs"])))
    print("batch size : 沿用配置文件里的设置（本脚本不覆盖）")
    print("学习率     : 首轮 {0:.2e}，每轮 ×{1:g}，下限 {2:.2e}".format(
        BASE_LR, LR_DECAY, MIN_LR))
    print("每轮流程   : 训练 {0} epoch → 验证集评估 → 测试集评估 → 写入表格".format(
        EPOCHS_PER_CYCLE))
    print("停止方式   : 在仓库根目录新建 STOP 文件；或按 Ctrl+C")
    if PATIENCE:
        print("提前收敛   : 连续 {0} 轮验证集 mAP 不提升则自动停止".format(PATIENCE))
    print("-" * 78, flush=True)

    run_dir = os.path.join(work_root, "run_{0}".format(run_id))

    history = []
    best = None
    best_map = -1.0
    stale = 0
    failures = 0

    state = None
    if options["reset"]:
        backup_state(state_file, work_root, run_id)
    else:
        state = load_state(state_file)

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
            print("续训状态   : 第 {0} 轮继续，峰值学习率 {1:.2e}".format(
                cycle, cycle_lr(cycle)))
            print("起始权重   : {0}".format(os.path.relpath(resume_ckpt, REPO_ROOT)))
            if best_map > 0:
                print("历史最佳   : 验证集 mAP {0:.4f}".format(best_map))
        else:
            print("续训状态   : 状态文件里的最佳权重已不存在，改为按起始权重开始")

    if not resumed:
        bootstrap, auto = find_bootstrap(profile, options["init"])
        cycle = options["cycle"] if options["cycle"] else BOOTSTRAP_CYCLE
        if bootstrap:
            init_checkpoint = bootstrap
            print("起始权重   : {0}{1}".format(
                os.path.relpath(bootstrap, REPO_ROOT),
                "（{0} 里最新的）".format(profile["weights_dir"]) if auto else ""))
            print("起始轮次   : 第 {0} 轮，峰值学习率 {1:.2e}".format(
                cycle, cycle_lr(cycle)))
            if cycle <= 1:
                print("提示       : 现在按第 1 轮跑（学习率 1e-4）。若这份权重已经训过，")
                print("             用 --cycle <已训轮数+1> 指定，免得大学习率把已学的冲掉。")
        else:
            print("起始权重   : 没找到，从随机初始化开始")
            print("             （把练好的 .pth 放进 {0} 就会自动被采用）".format(
                profile["weights_dir"]))

    try:
        while True:
            if cycle_limit and cycle > cycle_limit:
                print("\n已达到设定的最大轮数 {0}，结束。".format(cycle_limit))
                break
            if os.path.isfile(STOP_FILE):
                print("\n检测到 STOP 文件，本轮结束后停止。")
                break

            lr = cycle_lr(cycle)
            work_dir = os.path.join(run_dir, "cycle_{0}".format(cycle))
            os.makedirs(os.path.join(REPO_ROOT, work_dir), exist_ok=True)

            overrides = [
                "optim_wrapper.optimizer.lr={0}".format(lr),
                # -1：保留全部 epoch 权重
                "default_hooks.checkpoint.max_keep_ckpts={0}".format(options["keep_epochs"]),
                # 关掉 save_best —— 它会在出现新最优时删掉上一个 best 文件
                "default_hooks.checkpoint.save_best={0}".format(
                    "None" if options["keep_epochs"] == -1 else "dota/mAP"),
                # 文件名带上本次运行的时间戳，避免跨次运行重名
                "default_hooks.checkpoint.filename_tmpl=epoch{{}}_{0}.pth".format(
                    run_id),
            ]
            for split in ("train", "val", "test"):
                overrides.append("{0}_dataloader.dataset.data_root={1}".format(
                    split, to_posix(data_root_path(profile))))
            if init_checkpoint:
                overrides.append("load_from={0}".format(to_posix(init_checkpoint)))

            command = [
                python, train_script,
                os.path.join(REPO_ROOT, config),
                "--work-dir", os.path.join(REPO_ROOT, work_dir),
                "--cfg-options", *overrides,
            ]

            source = ("从 {0} 继续".format(os.path.basename(init_checkpoint))
                      if init_checkpoint else "从头训练")
            print()
            print("=" * 78)
            print("第 {0} 轮开始：{1} 个 epoch，峰值学习率 {2:.2e}，{3}".format(
                cycle, EPOCHS_PER_CYCLE, lr, source))
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

            checkpoint, best_val = save_best_copy(
                os.path.join(REPO_ROOT, work_dir), run_id,
                retained_only=options["keep_epochs"] > 0)
            if checkpoint is None:
                print("\n第 {0} 轮没有产出权重文件，结束循环。".format(cycle))
                break
            checkpoint = os.path.abspath(checkpoint)
            print("\n本轮最优权重已另存：{0}".format(
                os.path.relpath(checkpoint, REPO_ROOT)))
            if best_val is not None:
                print("  （训练日志里的验证集 mAP {0:.4f}）".format(best_val))

            val_metrics = run_eval(python, config, checkpoint, "val",
                                   os.path.join(REPO_ROOT, work_dir))
            test_metrics = run_eval(python, config, checkpoint, "test",
                                    os.path.join(REPO_ROOT, work_dir))

            header = build_header(val_metrics, test_metrics)
            values = build_values(cycle, lr, elapsed / 60, val_metrics,
                                  test_metrics, checkpoint)
            xlsx_path = excel_log.append_row(
                os.path.join(REPO_ROOT, metrics_xlsx), header, values)

            val_map = (val_metrics or {}).get("mAP")
            test_map = (test_metrics or {}).get("mAP")
            history.append((cycle, lr, val_map, test_map))

            print("\n第 {0} 轮完成，用时 {1:.0f} 分钟".format(cycle, elapsed / 60))
            print("  验证集 mAP : {0}".format(
                "读取失败" if val_map is None else "{0:.4f}".format(val_map)))
            print("  测试集 mAP : {0}".format(
                "读取失败" if test_map is None else "{0:.4f}".format(test_map)))
            print("  已写入表格 : {0}".format(
                os.path.relpath(xlsx_path, REPO_ROOT)))

            if val_map is not None:
                if val_map > best_map:
                    best_map = val_map
                    best = (cycle, lr, val_map, test_map, checkpoint)
                    stale = 0
                else:
                    stale += 1
                    print("  已连续 {0} 轮验证集 mAP 没有提升（当前最佳 {1:.4f}）".format(
                        stale, best_map))

            # 只有验证集 mAP 变好才把它当作下次的起点
            state = state or {}
            previous_best = float(state.get("best_map") or -1.0)
            improved = val_map is not None and val_map > previous_best

            state.setdefault("cycles", []).append({
                "cycle": cycle,
                "lr": lr,
                "val_map": val_map,
                "test_map": test_map,
                "checkpoint": os.path.relpath(checkpoint, REPO_ROOT),
                "run_id": run_id,
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
                    print("  本轮不如历史最佳 {0:.4f}，下次从历史最佳权重继续".format(
                        previous_best))
            state["next_cycle"] = cycle + 1
            runs = list(state.get("runs") or [])
            if run_id not in runs:
                runs.append(run_id)
            state["runs"] = runs
            save_state(state_file, state)
            print("  训练状态已保存：{0}".format(state_file))

            init_checkpoint = state.get("best_ckpt") or checkpoint
            cycle += 1

            if PATIENCE and stale >= PATIENCE:
                print("  连续 {0} 轮未提升，提前停止。".format(PATIENCE))
                break
            time.sleep(2)
    finally:
        print_summary(profile, history, best, metrics_xlsx, work_root, run_id)

    return 0
