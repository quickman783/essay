import os, sys, torch
from pathlib import Path
ROOT = Path(r'E:/write_essay/essay')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'add_new_code'))
os.chdir(ROOT)

import mmrotate
import dmm_mbo
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('mmrotate')
from mmdet.registry import MODELS
from mmengine.runner import load_checkpoint

cfg = Config.fromfile('add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py')
model = MODELS.build(cfg.model)
print('模型构建 OK，参数量 %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

ck = torch.load('work_dirs/dmm_mbo_vedai_loop/run_20261121_161305/cycle_2/epoch7_20261121_161305.pth',
                map_location='cpu')
sd = ck['state_dict']
print('原始 ckpt 键数:', len(sd))

# 伪造 D 组 checkpoint：删掉 RMR 的两个模块
sd_d = {k: v for k, v in sd.items() if 'refine_align' not in k and 'refine_gamma' not in k}
print('删掉 RMR 后键数:', len(sd_d), ' 删除:', sorted({k.split('.')[1] for k in sd if 'refine_align' in k or 'refine_gamma' in k}))
torch.save({'state_dict': sd_d, 'meta': ck.get('meta', {})}, 'output/vcheck/fake_groupD.pth')

print('\n===== 用 load_checkpoint(strict=False) 加载 D 组权重到 E 组模型 =====')
load_checkpoint(model, 'output/vcheck/fake_groupD.pth', map_location='cpu', strict=False)
print('===== 加载结束，未抛异常 =====')

# 检查 γ 是否停在初值 0
print('refine_gamma 值:', [float(g) for g in model.refine_gamma])
