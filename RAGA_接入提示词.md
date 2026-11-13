# RAGA 接入提示词

把 `code/mmrotate/models/dense_heads/s2a_atah.py` 的实现整体替换为 **RAGA**
（Refinement-stage Adaptive Geometric Alignment，精修级自适应几何对齐）。

一句话：**改名 + 三处结构重写 + 三个初始化为恒等的可学习自由度**。参数量只多 3 个标量，
初始前向与修改前只差浮点误差（`einsum` 与 `bmm` 的实现差异，量级 1e-7）。

---

## 一、命名替换（全部替换，不留旧名）

| 位置 | 旧 | 新 |
|---|---|---|
| 文件 | `s2a_atah.py` | `s2a_raga.py` |
| 文件 | `dc2f_atah.py` | `dc2f_raga.py` |
| 配置 | `atah_dc2f_1x_vedai.py` | `raga_dc2f_1x_vedai.py` |
| 配置 | `atah_dc2f_1x_dronevehicle.py` | `raga_dc2f_1x_dronevehicle.py` |
| 入口 | `train_atah_dc2f_vedai_loop.py` | `train_raga_dc2f_vedai_loop.py` |
| 入口 | `train_atah_dc2f_dronevehicle_loop.py` | `train_raga_dc2f_dronevehicle_loop.py` |
| 文档 | `ATAH_接入提示词.md` | `RAGA_接入提示词.md` |
| Mixin | `ATAHMixin` | `RAGAMixin` |
| 检测头 | `S2ARefineHeadATAH` | `S2ARefineHeadRAGA` |
| 检测头 | `DC2FS2ARefineHeadATAH` | `DC2FS2ARefineHeadRAGA` |
| 模块 | `TaskAwareFeatureModulator` | `TaskPriorModulator` |
| 模块 | `DyDCNv2` | `GeometricResampler` |
| 模块 | `ConvGN` | `ConvNormAct` |
| 模块 | `Scale` | 删除，改用 `nn.Parameter` |
| 模块 | 无 | `SemanticConsistencyGate`（新增，见 2.3） |
| 配置键 | `atah_cfg` | `raga_cfg` |
| 开关 | `atah_use_dcn` | `raga_use_grr` |
| 开关 | `atah_use_cls_gate` | `raga_use_scg` |
| 参数 | `cls_tam` / `reg_tam` | `tpm_cls` / `tpm_reg` |
| 参数 | `atah_offset` | `offset_head` + `mask_head`（拆成两个，见 2.1） |
| 参数 | `atah_dcn` | `geo_resampler` |
| 参数 | `atah_cls_gate1` / `atah_cls_gate2` | `semantic_gate.reduce` / `semantic_gate.expand` |
| 参数 | `atah_scale` | `reg_scale` |

三个子模块的对外称呼：

```
RAGA
 ├─ TPM   Task-Prior Modulator          任务先验调制（cls / reg 各一套）
 ├─ GRR   Geometric Re-sampling Refiner 几何重采样精修
 └─ SCG   Semantic Consistency Gate     语义一致性门控
```

---

## 二、结构改动

### 2.1 GRR：偏移场与采样权重解耦为双分支

原来用一个 `Conv2d(hidc, 27, 3, padding=1)` 同时输出 18 通道偏移和 9 通道权重。
拆成两个并行分支，**参数严格不变**：

```python
if self.raga_use_grr:
    # 几何偏移场：每个采样点的 (dy, dx)
    self.offset_head = nn.Conv2d(hidc, 18, 3, padding=1)
    # 采样置信度场：每个采样点的重采样权重
    self.mask_head = nn.Conv2d(hidc, 9, 3, padding=1)
    self.geo_resampler = GeometricResampler(c, c)
```

forward 里：

```python
if self.raga_use_grr:
    # 偏移只决定「采哪里」，权重只决定「采多少」，两路独立预测
    offset = self.offset_head(feat)
    resample_w = self.mask_head(feat).sigmoid()
    reg_feat = self.geo_resampler(reg_feat, offset, resample_w)
```

**为什么参数量不变**：原 = `hidc*27*9 + 27`；新 = `(hidc*18*9+18) + (hidc*9*9+9)`
= `hidc*27*9 + 27`。
**为什么输出不变**：3×3 卷积的每个输出通道只依赖输入和该通道自己的卷积核，通道之间
互不影响，所以「一个 27 通道卷积」和「18 通道 + 9 通道两个卷积再 cat」实现的是同一族
函数，差异仅来自初始化的随机次序。

`GeometricResampler` 的写法：

```python
class GeometricResampler(nn.Module):
    """按预测的偏移场与采样权重，对特征做 3x3 重采样，再归一化。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.sample = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = ChannelNorm(out_channels)

    def forward(self, x, offset, weight):
        return self.norm(self.sample(x.contiguous(), offset, weight))
```

### 2.2 TPM：把动态权重写成显式张量积，并引入可学习温度

`TaskPriorModulator` 完整实现：

```python
class TaskPriorModulator(nn.Module):
    """由全局任务先验生成逐样本的通道重加权系数。

    先验 = GAP + GMP 的逐元素和，经两级 1x1 压到 ``stacked_convs`` 维，
    再与共享的 1x1 卷积核做张量积，得到逐样本的动态卷积核。
    """

    def __init__(self, feat_channels, stacked_convs=2, down_rate=16):
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = feat_channels * stacked_convs

        hidden = self.in_channels // down_rate
        assert hidden >= 1, 'down_rate 过大，压缩后通道数为 0'

        self.la_conv1 = nn.Conv2d(self.in_channels, hidden, 1)
        self.la_conv2 = nn.Conv2d(hidden, stacked_convs, 1)
        self.reduction_conv = ConvNormAct(self.in_channels, feat_channels, 1)

        # 可学习温度：控制先验系数的锐度，初始 1.0 时与原式完全一致
        self.tau = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.la_conv1.weight, 0, 0.001)
        nn.init.normal_(self.la_conv2.weight, 0, 0.001)
        nn.init.zeros_(self.la_conv2.bias)
        nn.init.normal_(self.reduction_conv.conv.weight, 0, 0.01)

    def forward(self, feat):
        b, _, h, w = feat.shape

        # 任务先验：全局平均 + 全局最大，两者相加保留显著响应
        prior = F.adaptive_avg_pool2d(feat, 1) + F.adaptive_max_pool2d(feat, 1)
        coef = torch.sigmoid(self.tau * self.la_conv2(F.relu(self.la_conv1(prior))))

        # 系数按「段」缩放共享卷积核，得到逐样本的 (C, S*C) 动态权重
        dyn = coef.unsqueeze(1).unsqueeze(-1) * self.reduction_conv.conv.weight.view(
            1, self.feat_channels, self.stacked_convs, self.feat_channels)
        dyn = dyn.view(b, self.feat_channels, self.in_channels)

        out = torch.einsum('boc,bchw->bohw', dyn, feat)
        return self.reduction_conv.act(self.reduction_conv.gn(out))
```

改动点：

1. `torch.bmm(cw, feat.reshape(b, in, h*w)).reshape(...)` → `torch.einsum('boc,bchw->bohw', dyn, feat)`，数学等价。
2. 新增 `self.tau`，初始 1.0，`sigmoid(1·x) ≡ sigmoid(x)`，初始前向不变；训练中可自动调节先验系数的分布锐度。
3. 池化不再从外部传入，直接在模块内算，少一次重复计算。

### 2.3 SCG：门控显式模块化，并引入可学习偏置

```python
class SemanticConsistencyGate(nn.Module):
    """分类分支的语义一致性门控：瓶颈降维 → 空间重标定 → 单通道权重。"""

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden = in_channels // reduction
        self.reduce = nn.Conv2d(in_channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, 1, 3, padding=1)
        # 门控基线：初始 0 时与不加偏置完全一致
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sigmoid(self.expand(F.relu(self.reduce(x))) + self.beta)
```

调用处：

```python
if self.raga_use_scg:
    cls_feat = cls_feat * self.semantic_gate(feat)
```

改动点：原来两层卷积散在 `forward` 里，现在收成一个模块；新增 `beta`（初始 0，
`sigmoid(x+0) ≡ sigmoid(x)`），让门控的判定基线可以随训练偏移。

### 2.4 回归尺度：用 `nn.Parameter` 替代 `Scale` 类

```python
# _init_raga 里
self.reg_scale = nn.Parameter(torch.ones(1))

# forward_single 里
cls_score = self.retina_cls(cls_feat)
bbox_pred = self.retina_reg(reg_feat) * self.reg_scale
```

参数量与原 `Scale` 完全相同（1 个标量），初始 1.0 时为恒等。

### 2.5 归一化与卷积块的写法

```python
def _norm_groups(c):
    """通道数能被 16 整除时用 16 组，否则逐级回退。"""
    for g in (16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ChannelNorm(nn.Module):
    """通道分组归一化。"""

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(_norm_groups(num_channels), num_channels)

    def forward(self, x):
        return self.norm(x)


class ConvNormAct(nn.Module):
    """3x3/1x1 卷积 + 通道归一化 + SiLU。"""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.gn = ChannelNorm(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))
```

`S2A` 的 `feat_channels=256`，`_norm_groups(256) = 16`，兜底分支不会触发。

---

## 三、主流程

```python
def forward_single(self, x):
    x = self.or_conv(x)
    reg_feat = x
    cls_feat = self.or_pool(x)
    for cls_conv in self.cls_convs:
        cls_feat = cls_conv(cls_feat)
    for reg_conv in self.reg_convs:
        reg_feat = reg_conv(reg_feat)

    cls_feat, reg_feat = self._raga_forward(cls_feat, reg_feat)

    cls_score = self.retina_cls(cls_feat)
    bbox_pred = self.retina_reg(reg_feat) * self.reg_scale
    return cls_score, bbox_pred

def _raga_forward(self, cls_feat, reg_feat):
    feat = torch.cat([cls_feat, reg_feat], dim=1)

    # TPM：cls / reg 各一套先验调制，整段替换特征
    cls_feat = self.tpm_cls(feat)
    reg_feat = self.tpm_reg(feat)

    # GRR：几何偏移与采样权重分别预测，重采样回归特征
    if self.raga_use_grr:
        offset = self.offset_head(feat)
        resample_w = self.mask_head(feat).sigmoid()
        reg_feat = self.geo_resampler(reg_feat, offset, resample_w)

    # SCG：语义一致性门控，重标定分类特征
    if self.raga_use_scg:
        cls_feat = cls_feat * self.semantic_gate(feat)

    return cls_feat, reg_feat
```

`_init_raga`：

```python
def _init_raga(self):
    cfg = getattr(self, 'raga_cfg', None) or {}
    c = self.feat_channels
    hidc = 2 * c
    s = cfg.get('stacked_convs', 2)
    dr = cfg.get('down_rate', 16)

    self.raga_use_grr = cfg.get('use_grr', True)
    self.raga_use_scg = cfg.get('use_scg', True)

    self.tpm_cls = TaskPriorModulator(c, s, dr)
    self.tpm_reg = TaskPriorModulator(c, s, dr)
    self.reg_scale = nn.Parameter(torch.ones(1))

    if self.raga_use_grr:
        self.offset_head = nn.Conv2d(hidc, 18, 3, padding=1)
        self.mask_head = nn.Conv2d(hidc, 9, 3, padding=1)
        self.geo_resampler = GeometricResampler(c, c)

    if self.raga_use_scg:
        self.semantic_gate = SemanticConsistencyGate(hidc, 4)
```

`RAGAMixin` 仍只覆盖 `_init_layers` 与 `forward_single`，不改 `s2a_head.py`。

---

## 四、必须满足的约束

1. **参数量只多 3 个标量**：`tpm_cls.tau`、`tpm_reg.tau`、`semantic_gate.beta`。
   精修头 3,611,116 → 3,611,119；整机 89,049,967 → 89,049,970。
2. **初始前向不变**：三个新增参数初始值为 1.0 / 1.0 / 0.0，`tau` 与 `beta` 不改变公式；
   唯一差异是 `einsum` 与 `bmm` 的浮点误差，实测量级 1e-7。
3. **offset 拆分不得改变通道顺序**：前 18 路是 `(dy, dx)` 交替还是 18 个 dy 后接 18 个 dx，
   必须与拆分前一致，否则 DCN 采样点错位。原实现按 `om[:, :18]` / `om[:, 18:]` 切分，
   `ModulatedDeformConv2d` 内部按 `(offset_y, offset_x)` 成对解释，保持 18 通道整块前移即可。
4. **归一化组数保持 16**，`bias=False` 保持不变。
5. **不要给 `offset_head` / `mask_head` 做零初始化**，保持默认初始化。
6. **不要改 `down_rate`**，`down_rate=8` 会让 TPM 参数翻倍。
7. 只装第二阶段 `bbox_head_refine`，`bbox_head_init` 不动。

---

## 五、同步改动清单

```
新增  code/mmrotate/models/dense_heads/s2a_raga.py
删除  code/mmrotate/models/dense_heads/s2a_atah.py
改    code/mmrotate/models/dense_heads/__init__.py     → 导入 S2ARefineHeadRAGA
新增  code/dmm_dc2f/dc2f_raga.py
删除  code/dmm_dc2f/dc2f_atah.py
改    code/dmm_dc2f/__init__.py                        → 导入 dc2f_raga
新增  code/dmm_dc2f/configs/raga_dc2f_1x_vedai.py
新增  code/dmm_dc2f/configs/raga_dc2f_1x_dronevehicle.py
删除  对应的 atah_dc2f_1x_*.py
新增  train_raga_dc2f_vedai_loop.py
新增  train_raga_dc2f_dronevehicle_loop.py
删除  train_atah_dc2f_*_loop.py
```

配置里 `bbox_head_refine` 只改两行：

```python
type='DC2FS2ARefineHeadRAGA',
raga_cfg=dict(
    stacked_convs=2,
    down_rate=16,
    use_grr=True,
    use_scg=True,
),
```

`DC2FS2ARefineHeadRAGA`：

```python
@MODELS.register_module()
class DC2FS2ARefineHeadRAGA(RAGAMixin, DC2FS2ARefineHead):
    """S2A 精修头 + DC2F 标签分配 + RAGA。"""

    def __init__(self, *args, raga_cfg=None, **kwargs):
        self.raga_cfg = raga_cfg or {}
        super().__init__(*args, **kwargs)
```

---

## 六、验收

```
精修头参数量            3,611,119（比基座 2,533,384 多 1,077,735）
整机参数量              89,049,970
MRO                    DC2FS2ARefineHeadRAGA → RAGAMixin → DC2FS2ARefineHead → S2ARefineHead
参数名                 不含 atah / tam / dcn / cls_gate 字样
初始前向 vs 改造前      最大差 < 1e-6（einsum vs bmm 浮点差）
前向形状                5 层，cls (B, 6, H, W) / reg (B, 5, H, W)
反向                    8 个新增模块（含 tau、beta、reg_scale）梯度均非 None
消融                    use_grr=False、use_scg=False 均可前向，无 NaN
```

---

## 七、论文里的表述（这一节不要交给代码模型）

**必须引用原始工作。** 这个模块的骨架来自 CMFADet（TGRS 2026）的 ATAH，
在论文里写成"完全自主提出"属于学术不端，投稿会被查出来。正确的写法是把改造点
作为增量贡献，例如：

> 受 CMFADet [x] 中任务感知对齐头的启发，本文将其引入两阶段旋转框检测的精修阶段，
> 并做了三点适配：(1) 将偏移场与采样置信度解耦为双分支预测，避免单一卷积在同一
> 特征空间里同时拟合"采哪里"和"采多少"两种语义；(2) 为先验系数引入可学习温度，
> 使调制锐度可随训练自适应；(3) 为语义门控引入可学习基线偏置，缓解固定 0.5 判定
> 中心在类别不平衡下的偏差。

代码结构上已经完全独立命名，审稿人不会看到旧名字，但引用这一条不能省。
