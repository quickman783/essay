"""一键重建 VEDAI 训练数据：准备图像对 → 转换标注（可选先重新划分）。

默认按 7:2:1 划分 train / val / test。

为什么必须用这个脚本而不是分别跑三个命令：
    旧的 `datasets/vedai` 里可能残留着上一次划分的文件。如果只重新划分而不清理，
    同一张图片会同时出现在 train 和 val（或 val 和 test）里，验证/测试 mAP
    会失真。本脚本会先清空 `datasets/vedai`，各步骤内部也会剔除不属于当前
    划分的残留文件，最后做一致性校验，不通过就以非零码退出。

用法（在仓库根目录执行）：
    # 完整重建：重新划分 + 准备数据（默认 7:2:1）
    python code/scripts/rebuild_vedai.py

    # 划分已经对了，只重建 datasets/vedai（推荐：不会动 data/VEDAI 的划分）
    python code/scripts/rebuild_vedai.py --no-split

    # 其他比例，例如 8:1:1
    python code/scripts/rebuild_vedai.py --ratios 0.8 0.1 0.1

    # 换随机种子（同样的种子结果可复现）
    python code/scripts/rebuild_vedai.py --seed 2024
"""

import argparse
import importlib.util
import os
import shutil
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, "code")

RAW_DIR = os.path.join(ROOT, "data", "VEDAI")
XML_DIR = os.path.join(RAW_DIR, "AnnotationsXML512")
IMAGE_DIR = os.path.join(RAW_DIR, "Vehicules512")
RAW_ANN_DIR = os.path.join(RAW_DIR, "Annotations512")
SPLIT_DIR = RAW_DIR                      # data/VEDAI/{train,val,test}
DATASET_DIR = os.path.join(ROOT, "datasets", "vedai")

SPLITS = ("train", "val", "test")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


splitter = load_module("split_vedai_dataset",
                       os.path.join(CODE, "scripts", "split_vedai_dataset.py"))
data_process = load_module("data_process",
                           os.path.join(CODE, "tools", "data_process.py"))
VOC2DOTA = load_module("VOC2DOTA",
                       os.path.join(CODE, "tools", "VOC2DOTA.py"))


def check_inputs():
    missing = [p for p in (XML_DIR, IMAGE_DIR, RAW_ANN_DIR) if not os.path.isdir(p)]
    if missing:
        raise SystemExit(
            "缺少输入目录：\n  " + "\n  ".join(missing)
            + '\n请先运行 python "code/scripts/vedai_txt_to_xml.py" 生成 XML。'
        )


def main():
    parser = argparse.ArgumentParser(description="重建 VEDAI 训练数据（默认 7:2:1）。")
    parser.add_argument("--ratios", nargs=3, type=float, default=splitter.DEFAULT_RATIOS,
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="train/val/test 比例，默认 0.7 0.2 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink",
                        help="图像放入划分目录的方式，默认 hardlink（同盘不额外占空间）")
    parser.add_argument("--no-split", action="store_true",
                        help="跳过重新划分，直接用现有的 data/VEDAI/{train,val,test} 重建 datasets/vedai")
    args = parser.parse_args()

    check_inputs()

    if args.no_split:
        for split in SPLITS:
            source = os.path.join(SPLIT_DIR, split, "{0}label".format(split))
            if not os.path.isdir(source):
                raise SystemExit(
                    "找不到 {0}，无法使用 --no-split。\n"
                    "请先不带 --no-split 跑一次，或检查 data/VEDAI 的划分目录。".format(source))
        print("=" * 70)
        print("第 1 步 / 3：跳过重新划分，沿用现有的 train / val / test")
        for split in SPLITS:
            count = len([n for n in os.listdir(os.path.join(SPLIT_DIR, split, "{0}img".format(split)))
                         if os.path.splitext(n)[1].lower() in data_process.IMAGE_SUFFIXES])
            print("  {0:<6}{1:>6} 张".format(split, count))
        print("=" * 70)
    else:
        ratios = list(args.ratios)
        total = sum(ratios)
        if total <= 0:
            raise SystemExit("比例之和必须大于 0。")
        ratios = [r / total for r in ratios]

        print("=" * 70)
        print("第 1 步 / 3：按 {0:g} : {1:g} : {2:g} 重新划分 train / val / test".format(*ratios))
        print("=" * 70)
        splitter.split_dataset(
            XML_DIR, IMAGE_DIR, SPLIT_DIR,
            ratios=ratios, seed=args.seed, mode=args.mode, force=True,
        )

    print()
    print("=" * 70)
    print("第 2 步 / 3：清空旧训练数据目录，重建可见光 / 红外图像对")
    print("=" * 70)
    if os.path.isdir(DATASET_DIR):
        print("删除 {0}".format(DATASET_DIR))
        shutil.rmtree(DATASET_DIR)
    for split in SPLITS:
        data_process.prepare_split(SPLIT_DIR, DATASET_DIR, split,
                                   crop=None, ir_to_gray=False, overwrite=True)
    print("图像对已生成：{0}".format(DATASET_DIR))

    print()
    print("=" * 70)
    print("第 3 步 / 3：把原始四角点标注转成 DOTA 格式")
    print("=" * 70)
    for split in SPLITS:
        written, skipped = VOC2DOTA.prepare_split(
            SPLIT_DIR, DATASET_DIR, split,
            dataset="vedai", shift=0.0,
            raw_annotations=RAW_ANN_DIR, overwrite=True,
        )
        print("{0:<6}{1:>6} 个标注文件，{2} 张图无目标类别".format(split, written, skipped))

    verify()


def verify():
    print()
    print("=" * 70)
    print("校验")
    print("=" * 70)

    ids = {}
    ok = True
    for split in SPLITS:
        base = os.path.join(DATASET_DIR, split)
        sub_ids = {
            sub: {os.path.splitext(f)[0] for f in os.listdir(os.path.join(base, sub))}
            for sub in ("image_vi", "image_ir", "annofiles_vi", "annofiles_ir")
        }
        ids[split] = sub_ids["annofiles_vi"]
        same = len(set(map(frozenset, sub_ids.values()))) == 1
        print("{0:<6} image_vi={1:<5} image_ir={2:<5} annofiles_vi={3:<5} annofiles_ir={4:<5} {5}".format(
            split, len(sub_ids["image_vi"]), len(sub_ids["image_ir"]),
            len(sub_ids["annofiles_vi"]), len(sub_ids["annofiles_ir"]),
            "一致" if same else "不一致 !!"))
        ok = ok and same

        if not ids[split]:
            print("       {0} 划分没有任何标注文件 !!".format(split))
            ok = False
            continue
        source = os.path.join(SPLIT_DIR, split, "{0}label".format(split))
        expected = {os.path.splitext(f)[0] for f in os.listdir(source)
                    if f.lower().endswith(".xml")}
        extra = ids[split] - expected
        if extra:
            print("       有 {0} 个标注不在源划分 {1} 里（上一次划分的残留）!!".format(
                len(extra), os.path.basename(source)))
            ok = False

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = ids[left] & ids[right]
        print("{0} & {1:<5} 重叠 {2} 张 {3}".format(
            left, right, len(inter), "" if not inter else "!! 数据泄漏"))
        ok = ok and not inter

    total = sum(len(v) for v in ids.values())
    print("合计 {0} 张（train {1} / val {2} / test {3}）".format(
        total, len(ids["train"]), len(ids["val"]), len(ids["test"])))
    if total:
        print("比例 {0:.1f}% / {1:.1f}% / {2:.1f}%".format(
            100 * len(ids["train"]) / total,
            100 * len(ids["val"]) / total,
            100 * len(ids["test"]) / total))

    classes = Counter()
    for split in SPLITS:
        folder = os.path.join(DATASET_DIR, split, "annofiles_vi")
        for name in os.listdir(folder):
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 10:
                        classes[parts[8]] += 1
    print("类别分布 " + ", ".join("{0}={1}".format(k, v) for k, v in sorted(classes.items())))

    print()
    print("结果：{0}".format("通过" if ok else "存在问题，请检查上面的提示"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
