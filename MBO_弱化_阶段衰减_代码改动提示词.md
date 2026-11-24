# MBO 阶段衰减（弱化辅助监督）代码改动提示词

目标：给 MBO 增加一个 `refine_aux_scale`，只缩放**辅助分支在 refine 阶段**的损失。设成 `0.0` 时，辅助监督只保留 init 阶段，refine 阶段不再竞争最终检测输出。

默认值 `1.0`，旧配置行为逐位不变。

---

## 一、代码改动（唯一需要修改的文件）

**文件**：`E:\write_essay\essay\add_new_code\dmm_mbo\detector.py`

### 改动 1：`mbo_cfg` 默认值增加一个键

找到 `__init__` 里的这段（约第 28-32 行）：

```python
        self.mbo_cfg = dict(
            enabled=True, temperature=1.0, aux_loss_weight=1.0,
            cost_class=1.0, cost_bbox=1.0,
            density_gate=False, density_ref=12.0, density_power=2.0,
            density_value=0.0)
```

改成：

```python
        self.mbo_cfg = dict(
            enabled=True, temperature=1.0, aux_loss_weight=1.0,
            cost_class=1.0, cost_bbox=1.0,
            refine_aux_scale=1.0,
            density_gate=False, density_ref=12.0, density_power=2.0,
            density_value=0.0)
```

### 改动 2：非负校验加上这个键

找到（约第 40-42 行）：

```python
        if any(self.mbo_cfg[k] < 0 for k in (
                'aux_loss_weight', 'cost_class', 'cost_bbox')):
```

改成：

```python
        if any(self.mbo_cfg[k] < 0 for k in (
                'aux_loss_weight', 'cost_class', 'cost_bbox',
                'refine_aux_scale')):
```

### 改动 3：`supervise()` 增加参数

找到（约第 159 行）：

```python
        def supervise(stage_heads, stage_outputs, stage, rois=None, scale=1.0):
```

改成：

```python
        def supervise(stage_heads, stage_outputs, stage, rois=None, scale=1.0,
                      aux_scale=1.0):
```

### 改动 4：辅助分支的缩放系数乘上它

找到（约第 169-170 行）：

```python
                branch_scale = scale * (1.0 if index == 0 else
                                        self.mbo_cfg['aux_loss_weight'] * gate)
```

改成：

```python
                branch_scale = scale * (
                    1.0 if index == 0 else
                    self.mbo_cfg['aux_loss_weight'] * gate * aux_scale)
```

### 改动 5：增加一个监控键

找到（约第 183 行）：

```python
            losses['mbo_matched_gts_' + stage] = diagnostics['valid'].all(0).sum().float()
```

在它下一行加：

```python
            losses['mbo_aux_scale_' + stage] = features[0][0].new_tensor(
                aux_scale)
```

### 改动 6：refine 阶段传入这个系数

找到（约第 198-199 行）：

```python
            supervise(heads, outputs, 'refine_' + str(stage), rois,
                      self.train_cfg.stage_loss_weights[stage])
```

改成：

```python
            supervise(heads, outputs, 'refine_' + str(stage), rois,
                      self.train_cfg.stage_loss_weights[stage],
                      aux_scale=self.mbo_cfg['refine_aux_scale'])
```

`supervise(heads, outputs, 'init')` 那一行不动，走默认 `aux_scale=1.0`。

### 兼容性

- 六个改动点都在 `detector.py`，不改任何其他已有文件。
- 默认 `refine_aux_scale=1.0`，`groupb` / `groupc` / `groupd` / `gate` / `dmm_mbo_vedai_train` / `dmm_mbo_train` 的 loss 数值与改动前完全一致。
- `mbo_aux_scale_*` 不含 `loss` 字样，只作为日志指标，不参与反传。
- 辅助分支的前向仍然计算（refine 阶段），只是损失被乘 0；不改计算图结构。

---

## 二、验证组：纯 MBO + 阶段衰减（3 个新文件）

用途：单变量验证「掉点是否来自 refine 阶段的辅助监督」。对照组是 `dmm_mbo_vedai_groupd.py`。

### 文件 1：`add_new_code/dmm_mbo/configs/dmm_mbo_vedai_weakaux.py`

```python
"""DMM + MBO on VEDAI, group D with stage-decayed auxiliary supervision.

Same switches as group D (pre-fork, per-GT reweighting, no RMR). The only
difference is mbo_cfg.refine_aux_scale = 0: the auxiliary branches are still
computed at the refine stage, but their loss is scaled to zero, so only the
init stage supervises the shared features.
"""
_base_ = [
    './_base_/dmm_s2anet_vedai_rgbt.py',
    './_base_/schedule_dmm_1x.py',
    './_base_/datasets/VEDAI.py',
    './_base_/default_runtime.py',
]
import os
custom_imports = dict(
    imports=['dmm_mbo', 'dmm_mbo.training.progress_bar',
             'dmm_mbo.training.dota_per_class'], allow_failed_imports=False)
model = dict(
    type='MBODMMDetector',
    backbone_vi=dict(use_checkpoint=True),
    backbone_ir=dict(use_checkpoint=True),
    mbo_auxiliary=True,
    mbo_fork='pre',
    mbo_refine_fusion=False,
    mbo_gamma_init=0.0,
    mbo_cfg=dict(enabled=True, temperature=1., aux_loss_weight=1.,
                 cost_class=1., cost_bbox=1.,
                 refine_aux_scale=0.))
randomness = dict(seed=int(os.environ.get('DMM_TRAIN_SEED', '20261121')))
custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(max_keep_ckpts=-1))
work_dir = './work_dirs/dmm_mbo_vedai_weakaux'
load_from = None
resume = False
```

### 文件 2：`add_new_code/dmm_mbo/vedai_weakaux_loop.py`

```python
"""Independent DMM + stage-decayed MBO on VEDAI; reuse the 12-epoch engine."""
import importlib.util
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'training/train_loop.py'
    spec = importlib.util.spec_from_file_location(
        'mbo_vedai_weakaux_engine', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    # 引擎的 --pretrained 恒为 True，会检查 code/weights/vssm/*.pth
    # （本地不存在，直接 SystemExit）。它不参与建模，压回 False 绕开。
    _engine_parse_cli = engine.parse_cli

    def _vedai_parse_cli():
        options = _engine_parse_cli()
        options['pretrained'] = False
        return options

    engine.parse_cli = _vedai_parse_cli

    engine.PROFILES['vedai_mbo_weakaux'] = dict(
        label='VEDAI', variant='DMM + MBO (stage-decayed aux)',
        detail='6 classes; three branches; pre-fork; refine aux scale 0',
        config='add_new_code/dmm_mbo/configs/dmm_mbo_vedai_weakaux.py',
        work_root='work_dirs/dmm_mbo_vedai_weakaux',
        metrics_xlsx='work_dirs/dmm_mbo_vedai_weakaux_metrics.xlsx',
        default_data_root='datasets/vedai', env_var='DMM_VEDAI_ROOT',
        weights_dir='weights/dmm_mbo_vedai_weakaux',
        classes='6 classes (car / truck / tractor / camping_car / van / pickup)')
    return engine.main('vedai_mbo_weakaux')
```

### 文件 3：`train_mbo_vedai_weakaux_loop.py`（仓库根目录）

```python
"""DMM + stage-decayed MBO on VEDAI; run from the essay repository."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT / 'code'
EXTENSION_ROOT = ROOT / 'add_new_code'
BASE_ROOT = CODE_ROOT
if not (BASE_ROOT / 'mmrotate/__init__.py').is_file():
    BASE_ROOT = CODE_ROOT / 'DMM-main'
os.environ['DMM_BASE_ROOT'] = str(BASE_ROOT)
sys.path.insert(0, str(BASE_ROOT))
sys.path.insert(0, str(EXTENSION_ROOT))
os.environ['PYTHONPATH'] = os.pathsep.join(
    (str(EXTENSION_ROOT), str(BASE_ROOT), os.environ.get('PYTHONPATH', '')))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location(
        'mbo_vedai_weakaux_entry',
        EXTENSION_ROOT / 'dmm_mbo/vedai_weakaux_loop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raise SystemExit(module.main())
```

---

## 三、目标组：DCFL + 阶段衰减 MBO（3 个新文件）

用途：主模型切回 DCFL 之后的正式配置。与第二节的差别只有检测头类型和分配器。

**这一步不要在验证组出结果前启动。**

### 文件 4：`add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_vedai_weakaux.py`

```python
"""DMM + DCFL + stage-decayed MBO on VEDAI.

Heads and assigners follow code/dmm_dcfl/configs/dmm_dcfl_1x_vedai.py;
MBO switches follow dmm_mbo_vedai_weakaux.py (refine_aux_scale = 0).
"""
_base_ = [
    './_base_/dmm_s2anet_vedai_rgbt.py',
    './_base_/schedule_dmm_1x.py',
    './_base_/datasets/VEDAI.py',
    './_base_/default_runtime.py',
]
import os
custom_imports = dict(
    imports=['dmm_mbo', 'dmm_mbo.training.progress_bar',
             'dmm_mbo.training.dota_per_class'], allow_failed_imports=False)

angle_version = 'le135'

model = dict(
    type='MBODMMDetector',
    backbone_vi=dict(use_checkpoint=True),
    backbone_ir=dict(use_checkpoint=True),
    mbo_auxiliary=True,
    mbo_fork='pre',
    mbo_refine_fusion=False,
    mbo_gamma_init=0.0,
    mbo_cfg=dict(enabled=True, temperature=1., aux_loss_weight=1.,
                 cost_class=1., cost_bbox=1.,
                 refine_aux_scale=0.),
    bbox_head_init=dict(type='DCFLS2AHead'),
    bbox_head_refine=[
        dict(
            type='DCFLS2ARefineHead',
            num_classes=6,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(type='AlignConv', feat_channels=256, kernel_size=3,
                         strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(type='PseudoRotatedAnchorGenerator',
                                  strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=1,
                edge_swap=False,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11,
                           loss_weight=1.0),
        )
    ],
    train_cfg=dict(
        init=dict(
            assigner=dict(
                _delete_=True,
                type='C2FAssigner',
                assign_metric='gjsd',
                topk=16,
                topq=12,
                constraint='dgmm',
                gauss_thr=0.6,
                iou_calculator=dict(type='RBboxMetrics2D'),
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder',
                    angle_version=angle_version,
                    norm_factor=1,
                    edge_swap=False,
                    proj_xy=True,
                    target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                    target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                    use_box_type=False,
                ),
            ),
        ),
        refine=[
            dict(
                assigner=dict(
                    _delete_=True,
                    type='C2FAssigner',
                    assign_metric='gjsd',
                    topk=16,
                    topq=12,
                    constraint='dgmm',
                    gauss_thr=0.6,
                    iou_calculator=dict(type='RBboxMetrics2D'),
                    bbox_coder=dict(
                        type='DeltaXYWHTRBBoxCoder',
                        angle_version=angle_version,
                        norm_factor=1,
                        edge_swap=False,
                        proj_xy=True,
                        target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                        use_box_type=False,
                    ),
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            )
        ],
    ),
)
randomness = dict(seed=int(os.environ.get('DMM_TRAIN_SEED', '20261121')))
custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(max_keep_ckpts=-1))
work_dir = './work_dirs/dmm_dcfl_mbo_vedai_weakaux'
load_from = None
resume = False
```

**`bbox_head_refine` 必须整段重写，这是这套基配置特有的坑。** 基配置 `add_new_code/dmm_mbo/configs/_base_/dmm_s2anet_vedai_rgbt.py:114` 里它是 list（`[dict(...)]`），而 mmengine 只对 **dict 型**的键做递归合并，list 型直接整体替换：

- 写 `bbox_head_refine=dict(type='DCFLS2ARefineHead')` → 直接 `TypeError: ... is a dict in the child config but is of type <class 'list'>`
- 写 `bbox_head_refine=[dict(type='DCFLS2ARefineHead')]` → list 被替换成只剩 `type` 一项，`frm_cfg` / `anchor_generator` / `bbox_coder` 全部丢失，模型构建时报缺参数

`code/dmm_dcfl/configs/dmm_dcfl_1x_vedai.py` 里能写 `dict(type='DCFLS2ARefineHead')`，是因为它继承的 `code/configs/dmm/_base_/dmm_s2anet_rgbt.py:123` 里这个是 **dict**，两套基配置结构不同，不能照抄。

上面这份写法已实测：解析后 `bbox_head_refine` 仍是长度 1 的 list，`[0]['type']` 是 `DCFLS2ARefineHead`，且 `frm_cfg` 保留。

不要用变量复用 assigner 字典（如 `_c2f = dict(..., _delete_=True, ...)` 引用两次）。mmengine 合并时会 `pop` 掉 `_delete_`，同一个 dict 对象第二次引用时这个键已经不存在。两处各写一份完整字典。

### 文件 5：`add_new_code/dmm_mbo/vedai_dcfl_weakaux_loop.py`

```python
"""Independent DMM + DCFL + stage-decayed MBO on VEDAI."""
import importlib.util
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'training/train_loop.py'
    spec = importlib.util.spec_from_file_location(
        'dcfl_mbo_vedai_weakaux_engine', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    _engine_parse_cli = engine.parse_cli

    def _vedai_parse_cli():
        options = _engine_parse_cli()
        options['pretrained'] = False
        return options

    engine.parse_cli = _vedai_parse_cli

    engine.PROFILES['vedai_dcfl_mbo_weakaux'] = dict(
        label='VEDAI', variant='DMM + DCFL + MBO (stage-decayed aux)',
        detail='6 classes; C2FAssigner; three branches; refine aux scale 0',
        config='add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_vedai_weakaux.py',
        work_root='work_dirs/dmm_dcfl_mbo_vedai_weakaux',
        metrics_xlsx='work_dirs/dmm_dcfl_mbo_vedai_weakaux_metrics.xlsx',
        default_data_root='datasets/vedai', env_var='DMM_VEDAI_ROOT',
        weights_dir='weights/dmm_dcfl_mbo_vedai_weakaux',
        classes='6 classes (car / truck / tractor / camping_car / van / pickup)')
    return engine.main('vedai_dcfl_mbo_weakaux')
```

### 文件 6：`train_dcfl_mbo_vedai_weakaux_loop.py`（仓库根目录）

```python
"""DMM + DCFL + stage-decayed MBO on VEDAI; run from the essay repository."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT / 'code'
EXTENSION_ROOT = ROOT / 'add_new_code'
BASE_ROOT = CODE_ROOT
if not (BASE_ROOT / 'mmrotate/__init__.py').is_file():
    BASE_ROOT = CODE_ROOT / 'DMM-main'
os.environ['DMM_BASE_ROOT'] = str(BASE_ROOT)
sys.path.insert(0, str(BASE_ROOT))
sys.path.insert(0, str(EXTENSION_ROOT))
os.environ['PYTHONPATH'] = os.pathsep.join(
    (str(EXTENSION_ROOT), str(BASE_ROOT), os.environ.get('PYTHONPATH', '')))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location(
        'dcfl_mbo_vedai_weakaux_entry',
        EXTENSION_ROOT / 'dmm_mbo/vedai_dcfl_weakaux_loop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raise SystemExit(module.main())
```

---

## 四、文件清单

| 类型 | 路径 | 说明 |
|---|---|---|
| 修改 | `add_new_code/dmm_mbo/detector.py` | 6 处，见第一节 |
| 新建 | `add_new_code/dmm_mbo/configs/dmm_mbo_vedai_weakaux.py` | 验证组配置 |
| 新建 | `add_new_code/dmm_mbo/vedai_weakaux_loop.py` | 验证组 wrapper |
| 新建 | `train_mbo_vedai_weakaux_loop.py` | 验证组入口 |
| 新建 | `add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_vedai_weakaux.py` | 目标组配置 |
| 新建 | `add_new_code/dmm_mbo/vedai_dcfl_weakaux_loop.py` | 目标组 wrapper |
| 新建 | `train_dcfl_mbo_vedai_weakaux_loop.py` | 目标组入口 |

不允许改动的文件：

- `add_new_code/dmm_mbo/configs/dmm_mbo_vedai_groupb.py`
- `add_new_code/dmm_mbo/configs/dmm_mbo_vedai_groupc.py`
- `add_new_code/dmm_mbo/configs/dmm_mbo_vedai_groupd.py`
- `add_new_code/dmm_mbo/configs/dmm_mbo_vedai_gate.py`
- `add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py`
- `add_new_code/dmm_mbo/configs/dmm_mbo_train.py`
- `add_new_code/dmm_mbo/vedai_train_loop.py`
- `train_mbo_vedai_loop.py`
- `code/` 下的任何文件

---

## 五、验证（五项，全部通过才算完成）

工作目录：`E:\write_essay\essay`，Python：`C:\Users\feng1\anaconda3\envs\paper310\python.exe`

### 1. 语法

```bash
python -m py_compile \
  add_new_code/dmm_mbo/detector.py \
  add_new_code/dmm_mbo/configs/dmm_mbo_vedai_weakaux.py \
  add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_vedai_weakaux.py \
  add_new_code/dmm_mbo/vedai_weakaux_loop.py \
  add_new_code/dmm_mbo/vedai_dcfl_weakaux_loop.py \
  train_mbo_vedai_weakaux_loop.py \
  train_dcfl_mbo_vedai_weakaux_loop.py
```

### 2. 旧配置零改动

```bash
git diff --stat -- \
  add_new_code/dmm_mbo/configs/dmm_mbo_vedai_groupd.py \
  add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py \
  add_new_code/dmm_mbo/configs/dmm_mbo_train.py
```

输出必须为空。

### 3. 配置解析

```bash
python -c "
from mmengine.config import Config
c = Config.fromfile('add_new_code/dmm_mbo/configs/dmm_mbo_vedai_weakaux.py')
print('type      =', c.model['type'])
print('fork      =', c.model['mbo_fork'])
print('mbo_cfg   =', c.model['mbo_cfg'])
c2 = Config.fromfile('add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_vedai_weakaux.py')
print('head_init =', c2.model['bbox_head_init']['type'])
print('head_ref  =', c2.model['bbox_head_refine'][0]['type'])
print('refine_ok =', sorted(c2.model['bbox_head_refine'][0].keys()))
print('assigner0 =', c2.model['train_cfg']['init']['assigner']['type'])
print('assigner1 =', c2.model['train_cfg']['refine'][0]['assigner']['type'])
"
```

期望：

```
type      = MBODMMDetector
fork      = pre
mbo_cfg   = {... 'refine_aux_scale': 0.0 ...}
head_init = DCFLS2AHead
head_ref  = DCFLS2ARefineHead
refine_ok = ['anchor_generator', 'bbox_coder', 'feat_channels', 'frm_cfg', 'in_channels', 'loss_bbox', 'loss_cls', 'num_classes', 'stacked_convs', 'type']
assigner0 = C2FAssigner
assigner1 = C2FAssigner
```

`refine_ok` 这一行是硬门槛：必须同时出现 `frm_cfg`、`anchor_generator`、`bbox_coder`、`loss_cls`、`loss_bbox`。只打印出 `['type']` 说明 `bbox_head_refine` 被整体替换了，回到第三节检查写法。

### 4. 模型构建与辅助头类型

```bash
python -c "
import os, sys
sys.path.insert(0, 'code'); sys.path.insert(0, 'add_new_code')
os.environ['DMM_BASE_ROOT'] = os.path.abspath('code')
import mmrotate, dmm_mbo
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('mmrotate')
from mmdet.registry import MODELS
from dmm_mbo.dcfl.dcfl_heads import DCFLS2AHead, DCFLS2ARefineHead
c = Config.fromfile('add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_vedai_weakaux.py')
m = MODELS.build(c.model)
print('params = %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
print('aux_init =', type(m.mbo_init_heads['rgb']).__name__,
      isinstance(m.mbo_init_heads['rgb'], DCFLS2AHead))
print('aux_ref  =', type(m.mbo_refine_heads['rgb'][0]).__name__,
      isinstance(m.mbo_refine_heads['rgb'][0], DCFLS2ARefineHead))
print('refine_aux_scale =', m.mbo_cfg['refine_aux_scale'])
"
```

期望（已在本机实测，删掉 `refine_aux_scale` 后构建通过）：

```
params   = 106.76M
aux_init = DCFLS2AHead True
aux_ref  = DCFLS2ARefineHead True
refine_aux_scale = 0.0
```

`aux_init` 与 `aux_ref` 必须都是 `True`。这一条决定 `head_loss.prepare_loss` 会不会走进 `dcfl_targets` 分支——辅助分支若仍是 `S2AHead`，它会走 `MaxIoUAssigner` 那条老路，和主分支的 C2F 分配不一致。

`params = 106.76M` 与 DV 上 D 组同量级（辅助分支 18.79M + 主干 87.97M）；如果明显不同，说明某个 head 没被正确替换。

### 5. dry-run

```bash
python train_mbo_vedai_weakaux_loop.py --dry-run
python train_dcfl_mbo_vedai_weakaux_loop.py --dry-run
```

两条都必须打印各自独立的 `work_dirs` 路径，且不创建目录、不训练。

---

## 六、跑法

验证组：

```bash
python train_mbo_vedai_weakaux_loop.py
```

目标组（验证组出结果后再跑）：

```bash
python train_dcfl_mbo_vedai_weakaux_loop.py
```

两个入口都沿用引擎默认行为：一个 cycle = 12 epoch，连续 3 轮不涨才停。不要加 `--max-cycles`。

出结果后先在日志里核对监控键：

```bash
grep -m2 "mbo_aux_scale_refine_0" work_dirs/dmm_mbo_vedai_weakaux/run_*/cycle_1/*/*.log
```

必须打印 `0.0`。如果打印 `1.0`，说明改动 6 没生效，停下来检查。
