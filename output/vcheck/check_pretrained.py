"""静态检查：VEDAI 配置是否恢复了 VMamba ImageNet 预训练。"""
import os
import sys
from pathlib import Path

ROOT = Path(r'E:/write_essay/essay')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'add_new_code'))
os.chdir(ROOT)

from mmengine.config import Config  # noqa: E402

cfg = Config.fromfile('add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py')
vi = cfg.model['backbone_vi']['pretrained']
ir = cfg.model['backbone_ir']['pretrained']
print('backbone_vi.pretrained =', vi)
print('backbone_ir.pretrained =', ir)
print('exists =', bool(vi) and os.path.isfile(vi))
print('num_classes =', cfg.model['bbox_head_init']['num_classes'])
print('mbo_fork =', cfg.model.get('mbo_fork'),
      '| refine_fusion =', cfg.model.get('mbo_refine_fusion'),
      '| gamma_init =', cfg.model.get('mbo_gamma_init'))
