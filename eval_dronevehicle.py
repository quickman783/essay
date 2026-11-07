"""直接用已有权重评估 DroneVehicle 的验证集和测试集，不训练。

用法（在仓库根目录执行）：

    # 用 weights/dronevehicle 里最新的权重，val + test 都评估
    python eval_dronevehicle.py

    # 指定权重
    python eval_dronevehicle.py --checkpoint weights/dronevehicle/best_dota_mAP_epoch_12.pth

    # 只评估测试集
    python eval_dronevehicle.py --split test

    # 顺手把这次结果写进训练表格（轮次列写"评估"）
    python eval_dronevehicle.py --append

结果：
    * 终端打印 mAP / AP50 与每个类别的 AP
    * JSON 存到 work_dirs/dronevehicle_eval/<时间戳>/{val,test}_metrics.json

底层调用的是 code/scripts/evaluate.py，和循环训练里用的完全是同一套评估流程
（DOTAMetricPerClass，IoU=0.5），所以数字可以直接和表格里的对齐。
"""

import argparse
import glob
import importlib.util
import os
import sys
import time


def load_module(name, path):
    """按文件路径加载同仓库里的脚本（不依赖 sys.path）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
loop = load_module("dv_loop", os.path.join(REPO_ROOT, "train_dronevehicle_loop.py"))


def default_checkpoint():
    """默认评估的权重：脚本里写死的那份（weights/dronevehicle 里你放的），没有才退回目录里最新的。"""
    pinned = loop.BOOTSTRAP_CHECKPOINT
    if pinned:
        path = os.path.join(REPO_ROOT, pinned)
        if os.path.isfile(path):
            return path
    return newest_checkpoint()


def newest_checkpoint():
    folder = os.path.join(REPO_ROOT, loop.WEIGHTS_DIR)
    candidates = [p for p in glob.glob(os.path.join(folder, "**", "*.pth"), recursive=True)
                  if os.path.isfile(p)]
    return max(candidates, key=os.path.getmtime) if candidates else ""


def print_metrics(split, metrics):
    label = "验证集" if split == "val" else "测试集"
    if not metrics:
        print("\n[{0}] 评估失败，没有拿到指标。".format(label))
        return
    print()
    print("=" * 60)
    print("{0}（{1}）".format(label, split))
    print("=" * 60)
    print("  mAP      : {0:.4f}".format(metrics.get("mAP", float("nan"))))
    print("  AP50     : {0:.4f}".format(metrics.get("AP50", float("nan"))))
    others = [(k, v) for k, v in metrics.items()
              if k.startswith("AP_") and isinstance(v, (int, float))]
    if others:
        print("  逐类 AP  :")
        for key, value in others:
            print("    {0:<14}{1:.4f}".format(key[len("AP_"):], value))


def main():
    parser = argparse.ArgumentParser(description="评估 DroneVehicle 的 val / test 划分")
    parser.add_argument("--checkpoint", default="",
                        help="要评估的权重，默认用 train_dronevehicle_loop.py 里指定的那份"
                             "（weights/dronevehicle/best_dota_mAP_epoch_12.pth）")
    parser.add_argument("--split", nargs="+", choices=["val", "test"],
                        default=["val", "test"], help="评估哪些划分，默认两个都评估")
    parser.add_argument("--append", action="store_true",
                        help="把结果追加到 work_dirs/dronevehicle_metrics.xlsx")
    parser.add_argument("--out-dir", default="",
                        help="JSON 输出目录，默认 work_dirs/dronevehicle_eval/<时间戳>")
    args = parser.parse_args()

    python = loop.CONDA_PYTHON if os.path.isfile(loop.CONDA_PYTHON) else sys.executable

    checkpoint = args.checkpoint or default_checkpoint()
    if not checkpoint:
        raise SystemExit(
            "没有可评估的权重。请把 .pth 放进 {0}，或用 --checkpoint 指定。".format(
                loop.WEIGHTS_DIR))
    checkpoint = os.path.abspath(os.path.join(REPO_ROOT, checkpoint))
    if not os.path.isfile(checkpoint):
        raise SystemExit("权重不存在：{0}".format(checkpoint))

    # 检查数据目录，并把 DMM_DV_ROOT 设好给子进程用
    loop.prepare_data()

    out_dir = args.out_dir or os.path.join(
        "work_dirs", "dronevehicle_eval", time.strftime("%Y%m%d_%H%M%S"))
    out_dir = os.path.abspath(os.path.join(REPO_ROOT, out_dir))
    os.makedirs(out_dir, exist_ok=True)

    print("解释器     :", python)
    print("权重       :", os.path.relpath(checkpoint, REPO_ROOT))
    print("评估划分   :", " / ".join(args.split))
    print("结果目录   :", os.path.relpath(out_dir, REPO_ROOT))
    print("-" * 78, flush=True)

    results = {}
    for split in args.split:
        results[split] = loop.run_eval(python, checkpoint, split, out_dir)
        print_metrics(split, results[split])

    print()
    print("=" * 60)
    print("汇总（权重：{0}）".format(os.path.basename(checkpoint)))
    print("=" * 60)
    for split in args.split:
        value = (results.get(split) or {}).get("mAP")
        print("  {0:<6}{1}".format(
            split, "失败" if value is None else "{0:.4f}".format(value)))
    print("\nJSON 已保存到 : {0}".format(os.path.relpath(out_dir, REPO_ROOT)))

    if args.append:
        header = loop.build_header(results.get("val"), results.get("test"))
        values = loop.build_values("评估", "", 0, results.get("val"), results.get("test"),
                                   checkpoint)
        path = loop.excel_log.append_row(
            os.path.join(REPO_ROOT, loop.METRICS_XLSX), header, values)
        print("已写入表格   : {0}".format(os.path.relpath(path, REPO_ROOT)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
