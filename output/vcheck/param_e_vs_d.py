"""对比 D 组（无 RMR）与 E 组（有 RMR）的参数量，区分训练期与推理期。"""
import os
import sys
from pathlib import Path

ROOT = Path(r'E:/write_essay/essay')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'add_new_code'))
os.chdir(ROOT)

import torch  # noqa: E402
import mmrotate  # noqa: E402,F401
import dmm_mbo  # noqa: E402,F401
from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402

init_default_scope('mmrotate')
from mmdet.registry import MODELS  # noqa: E402


def build(cfg_path):
    cfg = Config.fromfile(cfg_path)
    return cfg, MODELS.build(cfg.model)


def stats(model):
    total = sum(p.numel() for p in model.parameters())
    aux = 0
    aux_names = ('mbo_necks', 'mbo_init_heads', 'mbo_refine_heads')
    rmr = 0
    rmr_detail = {}
    for n, p in model.named_parameters():
        if n.split('.')[0] in aux_names:
            aux += p.numel()
        if 'refine_align' in n or 'refine_gamma' in n:
            rmr += p.numel()
            rmr_detail[n] = p.numel()
    return total, aux, rmr, rmr_detail


rows = []
for tag, path in [
    ('D  (pre, 无 RMR, MBO on)', 'add_new_code/dmm_mbo/configs/dmm_mbo_vedai_groupd.py'),
    ('E  (pre, 有 RMR, MBO on)', 'add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py'),
]:
    cfg, model = build(path)
    total, aux, rmr, det = stats(model)
    infer = total - aux
    rows.append((tag, total, aux, infer, rmr, det))
    print(f'{tag}')
    print(f'  训练期总参数 {total/1e6:8.2f} M')
    print(f'  其中 MBO 辅助支路 {aux/1e6:8.2f} M  (推理丢弃)')
    print(f'  推理期参数 {infer/1e6:8.2f} M')
    print(f'  其中 RMR {rmr/1e6:8.2f} M')
    for k, v in det.items():
        print(f'     {k}  {v/1e6:.4f} M')
    print()

d, e = rows[0], rows[1]
print('================ 差值 ================')
print(f'E - D 训练期   {(e[1]-d[1])/1e6:+.2f} M')
print(f'E - D 推理期   {(e[3]-d[3])/1e6:+.2f} M')
print(f'相对 D 推理期  {(e[3]-d[3])/d[3]*100:+.3f} %')
