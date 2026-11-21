# VEDAI 两个创新点：代码改动提示词


---

## 0. 任务与边界

**目标**：在 VEDAI 上实现两个创新点。

| 编号 | 改动 | 作用 |
| -- | --- | --- |
| A | `mbo_fork` 开关（`'pre'` / `'post'`） | 让辅助支路可以从融合后分叉，用于"分叉位置"对照 |
| B | 零初始化精修再融合（`mbo_refine_fusion`） | 在 refine 阶段把单模态特征接回来，γ 初值 0 |

**数据集**：VEDAI，6 类（car / truck / tractor / camping_car / van / pickup），
train 817 / val 227 / test 118。

**训练方式：不冻结任何参数。** 与基线同一入口、同一 epoch 数、同一学习率，全参数参与训练。`mbo_gamma_init=0.0` 只保证初始化瞬间的前向与现状逐位相同，训练开始后 γ 会按正常梯度更新，**不保证训练后的指标不变**。

**允许新建的文件：**

- `add_new_code/dmm_mbo/configs/_base_/datasets/VEDAI.py`
- `add_new_code/dmm_mbo/configs/_base_/dmm_s2anet_vedai_rgbt.py`（复制，只改一行）
- `add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py`
- `add_new_code/dmm_mbo/vedai_train_loop.py`
- `train_mbo_vedai_loop.py`（仓库根目录）

**允许修改的文件：**

- `add_new_code/dmm_mbo/detector.py`

**不允许改：** `code/` 下任何文件、`head_loss.py`、`balancing.py`、`dcfl/`、`training/`、`configs/dmm_mbo_train.py`（DroneVehicle 那份）、`_base_/dmm_s2anet_rgbt.py`（共享基配置）。

`mbo_cfg` 里四个数（`temperature` / `aux_loss_weight` / `cost_class` / `cost_bbox`）**一个都不要动**，全部保持 1.0。本次只验证 A、B 两个结构改动，混入超参改动就说不清增益来源。

---

## 1. 新建 `_base_/datasets/VEDAI.py`

`_base_/datasets/` 目录下现在只有 `DroneVehicle.py`，需要为 VEDAI 补一份。

内容以 `code/configs/_base_/datasets/VEDAI.py` 为准，复制后**只改两处**：

```python
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = val_evaluator
```

（原文是 `DOTAMetric`。换成 `DOTAMetricPerClass` 后，日志和 xlsx 里会同时记录逐类 AP，写论文需要。注册入口在 `dmm_mbo.training.dota_per_class`，由配置里的 `custom_imports` 提供。）

**其余全部保持原样，特别是这两条不能动：**

1. `test_dataloader` 里的 `pipeline=val_pipeline`。**不要**改成 `test_pipeline`，那一份没有 `LoadAnnotations`，评估器会拿不到 GT，直接 `KeyError: 'bboxes'`。
2. `data_root = os.environ.get("DMM_VEDAI_ROOT", "datasets/vedai")`。数据实际在 `datasets/vedai`，环境变量名是 `DMM_VEDAI_ROOT`。

---

## 2. 新建 `_base_/dmm_s2anet_vedai_rgbt.py`（复制共享基配置）

**不能只靠子配置覆盖 `num_classes`。** 共享基配置 `_base_/dmm_s2anet_rgbt.py` 里 `bbox_head_refine` 是 **list**（第 105 行），而 MMEngine 合并配置时对 list 是整体替换：

```python
# mmengine/config/config.py:250-264
def _merge_a_into_b(a, b):
    if isinstance(a, dict):
        ...
    else:
        return a          # 非 dict（含 list）直接用子配置的值
```

子配置里写 `bbox_head_refine=dict(num_classes=6)` 会把整份 refine head 定义换成一个只有类别数的 dict，其余字段全部丢失，build 时报缺参数。`_base_/dmm_s2anet_1x_dronevehicle.py:10-11` 的注释说的就是这件事。

所以复制一份完整的模型基配置，只改类别数：

```bash
cp add_new_code/dmm_mbo/configs/_base_/dmm_s2anet_rgbt.py \
   add_new_code/dmm_mbo/configs/_base_/dmm_s2anet_vedai_rgbt.py
sed -i 's/^num_classes = 5$/num_classes = 6/' \
   add_new_code/dmm_mbo/configs/_base_/dmm_s2anet_vedai_rgbt.py
```

复制后**除第 10 行外一个字都不改**。`angle_version`、`data_preprocessor`、两个 backbone、`mtablock`、`fusblock`、`neck`、两个 head、`train_cfg` 全部保持一致。

改完确认第 10 行是 `num_classes = 6`，且文件里不再有 `num_classes = 5`。

---

## 3. 新建 `configs/dmm_mbo_vedai_train.py`

```python
"""DMM + MBO on VEDAI; from-scratch training, no pretrained checkpoint."""
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
    mbo_refine_fusion=True,
    mbo_gamma_init=0.0,
    mbo_cfg=dict(enabled=True, temperature=1., aux_loss_weight=1.,
                 cost_class=1., cost_bbox=1.))
randomness = dict(seed=int(os.environ.get('DMM_TRAIN_SEED', '20261121')))
custom_hooks = [dict(type='ProgressBarHook', bar_width=24, log_interval=10)]
val_evaluator = dict(type='DOTAMetricPerClass', metric='mAP', iou_thrs=[0.5])
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(max_keep_ckpts=-1))
work_dir = './work_dirs/dmm_mbo_vedai_loop'
load_from = None
resume = False
```

四处要点：

- **不要写 `pretrained`**。共享基配置写明 "Both backbones are trained from scratch, no pretrained checkpoint is loaded"，`backbone_vi/ir` 的 `pretrained` 都是 `None`。VEDAI 的 DMM 基线（`code/configs/dmm/dmm_s2anet_1x_vedai.py`）也是从头训。给 VEDAI 单独加载 VMamba 预训练会改变协议，与基线不可比。
- `backbone_vi=dict(use_checkpoint=True)` 是 dict 递归合并，只加这一个字段，不影响 `pretrained=None`。三支路 + batch=4 显存紧张时靠它兜。
- `mbo_fork='pre'` 与 `mbo_refine_fusion=True` 同时开是主模型配置；做分叉对照时只把 `mbo_fork` 改成 `'post'`，同时把 `mbo_refine_fusion` 关掉（理由见坑 5）。
- `work_dir` 必须与 DroneVehicle 的目录分开，否则循环训练会混写 `run_state.json`。

---

## 4. `detector.py` 改动 A：分叉位置开关

### A1. `__init__` 签名

现在：

```python
    def __init__(self, *args, mbo_cfg=None, mbo_auxiliary=True, **kwargs):
```

改成：

```python
    def __init__(self, *args, mbo_cfg=None, mbo_auxiliary=True,
                 mbo_fork='pre', mbo_refine_fusion=False,
                 mbo_gamma_init=0.0, **kwargs):
```

### A2. `__init__` 末尾（`self.mbo_auxiliary = mbo_auxiliary` 之后）

```python
        if mbo_fork not in ('pre', 'post'):
            raise ValueError("mbo_fork must be 'pre' or 'post'")
        self.mbo_fork = mbo_fork
        self.mbo_refine_fusion = bool(mbo_refine_fusion)
```

### A3. `_training_features`

现在（第 74~76 行）：

```python
        # Branch from these features, never from the DCFM output.
        return (self.neck(fused), self.mbo_necks['rgb'](rgb),
                self.mbo_necks['ir'](ir))
```

改成：

```python
        if self.mbo_fork == 'post':
            # Ablation: both auxiliary branches see the fused feature.
            branch_rgb = branch_ir = fused
        else:
            branch_rgb, branch_ir = rgb, ir
        return (self.neck(fused), self.mbo_necks['rgb'](branch_rgb),
                self.mbo_necks['ir'](branch_ir))
```

`'post'` 时两条辅助支路**故意**喂同一个 `fused`（DCFM 输出、FPN 之前），而不是 `self.neck(fused)`。原因是这样两支的 neck 仍各自独立，参数量与 `'pre'` 完全一致，对照才干净。

---

## 5. `detector.py` 改动 B：零初始化精修再融合

### B1. `__init__` 末尾追加（A2 之后）

```python
        if self.mbo_refine_fusion:
            if not self.mbo_auxiliary:
                raise ValueError('mbo_refine_fusion requires mbo_auxiliary=True')
            if self.mbo_fork == 'post':
                raise ValueError("mbo_refine_fusion needs mbo_fork='pre'")
            self.refine_align = nn.ModuleDict({
                name: nn.ModuleList([
                    deepcopy(head.feat_refine_module)
                    for head in self.bbox_head_refine])
                for name in ('rgb', 'ir')})
            self.refine_gamma = nn.ParameterList([
                nn.Parameter(torch.tensor(float(mbo_gamma_init)))
                for _ in range(self.num_refine_stages)])
```

三处说明：

- **不要从 `frm_cfg` build**，直接 `deepcopy(head.feat_refine_module)`。`S2ARefineHead.feat_refine_module` 就是一个 `AlignConv` 实例（`s2a_head.py:84` 用 `MODELS.build(frm_cfg)` 得到），deepcopy 后由 `init_weights()` 重新初始化，不依赖配置是否还在 `kwargs` 里。
- `self.num_refine_stages` 来自父类，当前配置是 **1**（`train_cfg.stage_loss_weights=[1.0]`），所以 `refine_gamma` 只有一个标量。用 `num_refine_stages` 循环写，以后加 stage 不用改。
- 文件顶部补 `import torch`（现在只有 `from torch import nn`）。

### B2. 新增 `_refine_features`（放在 `_training_features` 后面）

```python
    def _refine_features(self, stage, features, rois):
        """AlignConv(fused) + gamma * [AlignConv(rgb) + AlignConv(ir)]."""
        base = self.bbox_head_refine[stage].feature_refine(features[0], rois)
        if not self.mbo_refine_fusion:
            return base
        gamma = self.refine_gamma[stage]
        aux_rgb = self.refine_align['rgb'][stage](features[1], rois)
        aux_ir = self.refine_align['ir'][stage](features[2], rois)
        return [b + gamma * (r + i) for b, r, i in zip(base, aux_rgb, aux_ir)]
```

`AlignConv.forward(x, anchors)` 返回 `List[Tensor]`（每个 FPN 层一个），所以 `features[1]`、`features[2]`、`rois` 直接传进去即可，和 `feature_refine` 的调用方式完全一致。

### B3. `loss()` 的 refine 循环

现在（第 124~125 行）：

```python
            outputs = [head(head.feature_refine(x, roi))
                       for head, x, roi in zip(heads, features, rois)]
```

改成：

```python
            outputs = []
            for index, (head, x, roi) in enumerate(zip(heads, features, rois)):
                if index == 0:
                    x = self._refine_features(stage, features, roi)
                else:
                    x = head.feature_refine(x, roi)
                outputs.append(head(x))
```

`index == 0` 是融合分支，只有它加残差；两条辅助支路保持原逻辑。

### B4. 覆写 `predict` 与 `_forward`

父类 `predict`（`code/mmrotate/models/detectors/dmm_s2anet.py:171`）走 `extract_feat`，只有融合路径，拿不到 `features[1:]`，所以必须覆写。在 `remove_auxiliary_branches` 之前插入：

```python
    def predict(self, batch_inputs, batch_inputs2, batch_data_samples,
                rescale=True):
        if not self.mbo_refine_fusion:
            return super().predict(batch_inputs, batch_inputs2,
                                   batch_data_samples, rescale=rescale)
        features = self._training_features(batch_inputs, batch_inputs2)
        outs = self.bbox_head_init(features[0])
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for stage in range(self.num_refine_stages):
            head = self.bbox_head_refine[stage]
            outs = head(self._refine_features(stage, features, rois))
            if stage + 1 < self.num_refine_stages:
                rois = head.refine_bboxes(*outs, rois)
        batch_img_metas = [s.metainfo for s in batch_data_samples]
        predictions = self.bbox_head_refine[-1].predict_by_feat(
            *outs, rois=rois, batch_img_metas=batch_img_metas, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, predictions)

    def _forward(self, batch_inputs, batch_inputs2, batch_data_samples=None):
        if not self.mbo_refine_fusion:
            return super()._forward(batch_inputs, batch_inputs2)
        features = self._training_features(batch_inputs, batch_inputs2)
        outs = self.bbox_head_init(features[0])
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for stage in range(self.num_refine_stages):
            head = self.bbox_head_refine[stage]
            outs = head(self._refine_features(stage, features, rois))
            if stage + 1 < self.num_refine_stages:
                rois = head.refine_bboxes(*outs, rois)
        return outs
```

注意 `_forward` 的签名必须与父类一致（`batch_data_samples=None` 这个默认值不能省，mmengine 会按关键字调用）。两个方法开头都有短路，关闭 B 时完全走父类。

### B5. `init_weights` 追加

`AlignConv` 在 `nn.ModuleDict` / `nn.ModuleList` 里，父类遍历不到。在现有 `for branch in ('rgb', 'ir')` 块后面加：

```python
        if self.mbo_refine_fusion:
            for stage in range(self.num_refine_stages):
                for name in ('rgb', 'ir'):
                    self.refine_align[name][stage].init_weights()
```

### B6. `remove_auxiliary_branches` 保护

方法开头加：

```python
        if self.mbo_refine_fusion:
            raise RuntimeError('refine fusion needs the auxiliary branches')
```

开了 B 还调它，`predict` 会去访问已删除的 `mbo_necks` 并抛 `AttributeError`，错误信息很难查。

---

## 6. 运行入口（2 个新文件）

### 6.1 `add_new_code/dmm_mbo/vedai_train_loop.py`

复制 `baseline_train_loop.py`，把 profile 换成 VEDAI：

```python
"""Independent DMM + MBO on VEDAI; reuse the 12-epoch training engine."""
import importlib.util
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'training/train_loop.py'
    spec = importlib.util.spec_from_file_location('mbo_vedai_loop_engine', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.PROFILES['vedai_mbo'] = dict(
        label='VEDAI', variant='DMM + MBO + refine fusion',
        detail='6 classes; three branches; pre-fork; zero-init refine fusion',
        config='add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py',
        work_root='work_dirs/dmm_mbo_vedai_loop',
        metrics_xlsx='work_dirs/dmm_mbo_vedai_metrics.xlsx',
        default_data_root='datasets/vedai', env_var='DMM_VEDAI_ROOT',
        weights_dir='weights/dmm_mbo_vedai',
        classes='6 classes (car / truck / tractor / camping_car / van / pickup)')
    return engine.main('vedai_mbo')
```

### 6.2 `train_mbo_vedai_loop.py`（仓库根目录）

复制 `train_mbo_dronevehicle_loop.py`，只改最后指向的模块：

```python
    spec = importlib.util.spec_from_file_location(
        'mbo_vedai_entry',
        EXTENSION_ROOT / 'dmm_mbo/vedai_train_loop.py')
```

其余（`ROOT` / `BASE_ROOT` / `sys.path` / 编码设置）一字不改。

---

## 7. 八个坑

1. **`gamma_init` 不能非零。** γ=0 是唯一能保证"前向与现状逐位相同"的方式；0.01 都会让第一步的损失和梯度与基线不同。
2. **`num_classes` 不能靠子配置覆盖。** 基配置的 `bbox_head_refine` 是 list，子配置给 dict 会整段替换，见第 2 节。必须复制基配置改第 10 行。
3. **不要写 `pretrained`。** 基配置里两个 backbone 都是 `pretrained=None`（从头训），VEDAI 的 DMM 基线也是从头训。单独给 VEDAI 加载预训练会改变协议。
4. **`feat_refine_module` 是 `AlignConv` 单实例，不是多层容器。** 用 `deepcopy` 取，不要试图从 `frm_cfg` 重新 build。
5. **不要同时开 `mbo_fork='post'` 和 `mbo_refine_fusion=True`。** post 时 `features[1]` 和 `features[2]` 是同一个融合特征，残差变成"融合特征对自己"，白加参数。B1 里的 `raise` 就是拦这个。
6. **`num_refine_stages = 1`。** `train_cfg.stage_loss_weights=[1.0]` 决定只有一个 refine stage，所以只新增 2 个 AlignConv、约 **1.18 M** 参数（各约 0.59 M）。不要建长度为 5 的 ParameterList。
7. **VEDAI 的 test 只有 118 张。** 单次评估的抽样噪声比 DroneVehicle（8980 张）大得多。**所有分支组合和阈值都在 val（227 张）上选，test 只跑一次。**
8. **`batch_size` 与基线必须一致。** VEDAI 基配置是 `batch_size=4`（train / val / test 三处）。若显存不足要降到 2，**基线也必须用 2 重跑**，否则"有没有加模块"和"batch 大小"两个变量混在一起。

---

## 8. 静态验证（不需要训练）

```bash
# 1. 语法
python -m py_compile add_new_code/dmm_mbo/detector.py \
                      add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py \
                      add_new_code/dmm_mbo/vedai_train_loop.py \
                      train_mbo_vedai_loop.py

# 2. 基配置确实只改了一行
diff add_new_code/dmm_mbo/configs/_base_/dmm_s2anet_rgbt.py \
     add_new_code/dmm_mbo/configs/_base_/dmm_s2anet_vedai_rgbt.py

# 3. 配置能解析，参数到位
python -c "
from mmengine.config import Config
c = Config.fromfile('add_new_code/dmm_mbo/configs/dmm_mbo_vedai_train.py')
print('init  :', c.model.bbox_head_init.num_classes)
print('refine:', c.model.bbox_head_refine[0].num_classes)
print('anchor:', c.model.bbox_head_init.anchor_generator.strides)
print('pretrained:', c.model.backbone_vi.pretrained)
print('fork   :', c.model.mbo_fork)
print('fusion :', c.model.mbo_refine_fusion)
print('gamma  :', c.model.mbo_gamma_init)
print('mbo_cfg:', c.model.mbo_cfg)
print('eval   :', c.val_evaluator.type)
"

# 4. 改动范围
git status --porcelain
```

第 2 条应该只有一个 `num_classes` 的差异。第 3 条要满足：两个 head 都是 6 类、`pretrained=None`、`fork='pre'`、`fusion=True`、`gamma=0.0`、`mbo_cfg` 四个 1.0、`eval=DOTAMetricPerClass`。

再补四条 Python 层断言：

- 关闭 B（`mbo_refine_fusion=False`）时，`state_dict()` 里不含 `refine_align` / `refine_gamma`，参数量与改动前完全相同
- 开启 B 后，`model.refine_gamma[0].item() == 0.0`，参数量增加约 **1.18 M**
- `mbo_fork='post'` 且 `mbo_refine_fusion=True` 时，构造模型必须抛 `ValueError`
- 用同一份权重、同一 batch，分别以 `mbo_fork='pre', mbo_refine_fusion=False` 跑一次 `predict`，结果与改动前**逐位相同**（`torch.testing.assert_close(rtol=0, atol=0)`）

---

## 9. 交付自查

| 项 | 要求 |
| --- | --- |
| 已跟踪文件改动数 | 1（`detector.py`） |
| 新增文件数 | 5 |
| `code/` 是否被修改 | 否 |
| 共享基配置 `dmm_s2anet_rgbt.py` | 未改 |
| 复制出的 VEDAI 基配置 | 与共享版只有 `num_classes` 一行差异 |
| 两个 head 的类别数 | 6 |
| backbone 的 `pretrained` | `None`（从头训） |
| 训练时是否冻结参数 | **否**，全参数训练，与基线同协议 |
| `mbo_cfg` 四个数是否变化 | 否，全为 1.0 |
| `test_dataloader` 的 pipeline | `val_pipeline`（含 LoadAnnotations） |
| 关闭 A、B 时行为 | 与改动前逐位一致 |
| B 开启时推理 | 保留辅助分支 |
| 新增参数 | 约 1.18 M（仅 B） |
