"""在指定划分（val 或 test）上评估一个权重，把 mAP 与每类 AP 写成 JSON。

供 train_vedai_loop.py 在每个训练轮次结束后调用，也可以单独使用：

    python code/scripts/evaluate.py \
        --config code/configs/dmm/dmm_s2anet_1x_vedai.py \
        --checkpoint work_dirs/vedai_loop/cycle_1/best_dota_mAP_epoch_12.pth \
        --split test \
        --out work_dirs/vedai_loop/cycle_1/test_metrics.json
"""

import argparse
import json
import os
import os.path as osp
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="在 val / test 划分上评估权重")
    parser.add_argument("--config", required=True, help="训练配置路径")
    parser.add_argument("--checkpoint", required=True, help="要评估的权重")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="在哪个划分上评估，默认 test")
    parser.add_argument("--out", required=True, help="评估结果的 json 输出路径")
    parser.add_argument("--work-dir", default=None, help="评估过程的临时目录")
    parser.add_argument("--iou-thrs", nargs="+", type=float, default=[0.5],
                        help="评估用的 IoU 阈值，默认 0.5")
    return parser.parse_args()


def main():
    args = parse_args()

    from mmdet.utils import register_all_modules as register_all_modules_mmdet
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmrotate.utils import register_all_modules

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.work_dir or osp.join("work_dirs", "_eval")
    os.makedirs(cfg.work_dir, exist_ok=True)

    # 把测试用的 dataloader 指向要评估的划分
    stride = args.split
    dataset = cfg.test_dataloader.dataset
    dataset["ann_file"] = "{0}/annofiles_ir".format(stride)
    dataset["aux_ann_file"] = "{0}/annofiles_vi".format(stride)
    dataset["data_prefix"] = dict(
        img_path="{0}/image_vi".format(stride),
        img_path2="{0}/image_ir".format(stride),
    )

    # 用带逐类 AP 的评估器
    cfg.test_evaluator = dict(type="DOTAMetricPerClass", metric="mAP",
                              iou_thrs=args.iou_thrs)

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    # 去掉 dota/ 前缀，方便后面写表格
    clean = {}
    for key, value in (metrics or {}).items():
        name = key.split("/")[-1]
        try:
            clean[name] = float(value)
        except (TypeError, ValueError):
            clean[name] = value
    clean["split"] = args.split
    clean["checkpoint"] = args.checkpoint

    os.makedirs(osp.dirname(osp.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, ensure_ascii=False, indent=2)

    print("评估完成：{0} -> mAP={1}".format(
        args.split, clean.get("mAP", "n/a")), flush=True)
    return 0


def _dump_failure():
    """把 traceback 落到 --out 旁边，训练循环会读这个文件。

    评估是以子进程方式被调用的，它的 traceback 只会打在控制台上。
    用户如果用 IDE 的 ▶ 运行、或控制台被刷屏，报错就彻底丢了，
    表格里只会留下一行空指标。这里强制留档。
    """
    import traceback

    traceback.print_exc()
    argv = sys.argv[1:]
    if "--out" not in argv:
        return
    out = argv[argv.index("--out") + 1]
    err_path = os.path.splitext(os.path.abspath(out))[0] + "_error.txt"
    try:
        os.makedirs(os.path.dirname(err_path) or ".", exist_ok=True)
        with open(err_path, "w", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
    except OSError:
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        _dump_failure()
        sys.exit(1)
