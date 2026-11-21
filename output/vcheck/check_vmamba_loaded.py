"""确认 VEDAI 配置里的 VMamba 预训练是真能加载上的，不是只写了个路径。"""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(r'E:/write_essay/essay')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'add_new_code'))
os.chdir(ROOT)

from mmengine.config import Config  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
import mmrotate  # noqa: F401,E402

cfg = Config.fromfile('add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py')
backbone = MODELS.build(cfg.model['backbone_vi'])

ckpt = torch.load(cfg.model['backbone_vi']['pretrained'], map_location='cpu')
sd = backbone.state_dict()
key = 'patch_embed.proj.weight'
same = torch.equal(sd[key], ckpt['model'][key])
print('key:', key)
print('model[:4] :', [round(float(x), 6) for x in sd[key].flatten()[:4]])
print('ckpt [:4] :', [round(float(x), 6) for x in ckpt['model'][key].flatten()[:4]])
print('LOADED_EQUAL =', same)

matched = sum(1 for k in sd if k in ckpt['model'])
print('matched tensors:', matched, '/', len(sd))
