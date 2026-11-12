# DMM 改造提示词：把 DCFM 的 DSSM 换成 C2Former（ICA + AFS）

> 目标是叠加到已加 DC2F 的 DMM 上。
> 只做 level 1/2/3（192/384/768）的替换，level 0 保持 DSSM 不动（neck 是 `start_level=1`，level 0 无人读取）。
>
> 相对官方 C2Former 只加一处结构：校准后的两支各自回支，再用逐通道门控 `λ` 把差分 `x_vi' − x_ir'` 重注入（λ 初始化为 0）。
> 目的是补回被 DSSM 一起删掉的 `x_sub = x_vi − x_ir` 视差通路——求和之后差分不可恢复，不显式加回来就永久丢了。

---

## 零、DC2F 的位置

DC2F 在 `dmm_dc2f/` 里改的是 `bbox_head_init/refine` 的 `type` 和 `train_cfg.assigner`；
C2Former 改的是 `fusblock`，两者在模型里不重叠。

新配置继承 `dc2f_1x_vedai.py`，DC2F 的三处开关（`snt` / `csc` / `bind_last`）自动带过来。

---

## 一、改动清单

| # | 文件 | 动作 |
|---|---|---|
| 1 | `mmrotate/models/layers/dmm/c2former.py` | **新建**：C2Former / LayerNormProxy / ModalityNorm / C2FormerFusion |
| 2 | `mmrotate/models/layers/dmm/rgbtmamba.py` | 修改 `DCFM.__init__`（含新增差异门控 `λ`）、`DCFM._forward`、`DCFModule.__init__` |
| 3 | `dmm_dc2f/configs/c2f_dc2f_1x_vedai.py` | **新建**：继承 `dc2f_1x_vedai.py`，加 `fusblock` 字段 |
| 4 | `dmm_dc2f/configs/c2f_dc2f_1x_dronevehicle.py` | **新建**：同上，租机练 DroneVehicle 用 |
| 5 | `dmm_dc2f/train_loop.py` | `PROFILES` 里加一个 `vedai_c2f_dc2f` 条目 |

---

## 二、新建 `mmrotate/models/layers/dmm/c2former.py`

### 2.1 成品文件：直接整份复制

四个类（`C2Former` / `LayerNormProxy` / `ModalityNorm` / `C2FormerFusion`）已经生成好：

```
C:\Users\feng1\Desktop\论文\output\c2f_check\c2former.py
```

**整份复制到 `mmrotate/models/layers/dmm/c2former.py` 即可**，不需要再改。该文件已修好下面 2.2 的三处，且已含 2.3 的封装类。实测：三级实例化 8.14M，`forward` 输出严格为 `0.00e+00`。

只依赖 `torch` / `torch.nn` / `torch.nn.functional` / `einops`，不依赖 mmcv、mmrotate。

自行重抽时的源文件是 `output\c2f_check\c2f_mod.py`，抽完要按 2.2 修三处。

**不要**复制 `C2FormerResNet`、不要 `from ..builder import ROTATED_BACKBONES`、不要任何 `@ROTATED_BACKBONES.register_module()`。

### 2.2 复制后必须修的 3 处

| 位置 | 原文 | 改成 |
|---|---|---|
| `C2Former.forward` | `offset = offset.fill(0.0)` | `offset = offset.fill_(0.0)` |
| `C2Former.__init__` | `nn.Dropout(proj_drop, inplace=True)` ×2、`nn.Dropout(attn_drop, inplace=True)` ×2 | `inplace=False` |
| `C2Former._get_ref_points` | `torch.meshgrid(y, x)` | `torch.meshgrid(y, x, indexing="ij")` |

### 2.3 封装类（c2former.py 中已包含，此处为参照）

```python
class C2FormerFusion(nn.Module):
    """C2Former(ICA+AFS) 的单输出封装。

    输入/输出均为 NHWC（和 DCFM 内部布局一致），内部转 NCHW。
    结构照官方 C2FormerResNet：
        dims_in -> 1x1 + ReLU -> C2Former -> 1x1 + ReLU -> dims_in
    两组投影分别在两个模态上各一份，共 4 个 1x1。
    """

    def __init__(self, dims_in, dims_out, num_heads, stage_idx,
                 n_groups=1, stride=3, offset_range_factor=2, no_off=False,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dims_out % num_heads == 0, "dims_out 必须能被 num_heads 整除"
        assert dims_out % n_groups == 0, "dims_out 必须能被 n_groups 整除"

        self.c2f = C2Former(
            q_size=(1, 1), kv_size=(1, 1),
            n_heads=num_heads,
            n_head_channels=dims_out // num_heads,
            n_groups=n_groups,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            stride=stride,
            offset_range_factor=offset_range_factor,
            no_off=no_off,
            stage_idx=stage_idx,
        )
        self.vis_in = nn.Sequential(nn.Conv2d(dims_in, dims_out, 1, 1), nn.ReLU())
        self.lwir_in = nn.Sequential(nn.Conv2d(dims_in, dims_out, 1, 1), nn.ReLU())
        self.vis_out = nn.Sequential(nn.Conv2d(dims_out, dims_in, 1, 1), nn.ReLU())
        self.lwir_out = nn.Sequential(nn.Conv2d(dims_out, dims_in, 1, 1), nn.ReLU())
        self._zero_init()

    def _zero_init(self):
        # C2Former 内部输出投影 + 外层升维 1x1，weight/bias 全零
        # 升维那个 1x1 带 bias 且后接 ReLU，只清 proj_out 不够，ReLU(bias) != 0
        for m in (self.c2f.proj_out_vis, self.c2f.proj_out_lwir,
                  self.vis_out[0], self.lwir_out[0]):
            m.weight.data.zero_()
            m.bias.data.zero_()

    def forward(self, vis_x, lwir_x):
        """vis_x / lwir_x: (B, H, W, C) -> 返回 (给 vi 支的, 给 ir 支的)，同为 NHWC"""
        v = vis_x.permute(0, 3, 1, 2)
        l = lwir_x.permute(0, 3, 1, 2)

        out_vis, out_lwir = self.c2f(self.vis_in(v), self.lwir_in(l))

        out_vis = self.vis_out(out_vis).permute(0, 2, 3, 1)
        out_lwir = self.lwir_out(out_lwir).permute(0, 2, 3, 1)

        # 官方：out_vis 加到 lwir 支，out_lwir 加到 vis 支
        return out_lwir, out_vis
```

---

## 三、改 `mmrotate/models/layers/dmm/rgbtmamba.py`

### 3.1 顶部加 import

```python
from .c2former import C2FormerFusion
```

### 3.2 `DCFM.__init__`

签名末尾（`disable_z` 之后、`**kwargs` 之前）增加四个参数：

```python
        fus_type: str = "dssm",
        c2f_cfg: dict | None = None,
        stage_idx: int = 1,
        use_diff_gate: bool = True,
```

构造体改为：

```python
        self.use_checkpoint = use_checkpoint
        self.fus_type = fus_type
        self.use_diff_gate = use_diff_gate and fus_type == "c2f"

        if fus_type == "c2f":
            # 注意：c2f 分支不建 norm1_vi / norm1_ir。
            # C2Former 自带 ModalityNorm，且前面是 1x1+ReLU，不经 LayerNorm。
            # 建了也不调用，会在 DDP 下变成 unused parameter。
            cfg = dict(c2f_cfg or {})
            self.op = C2FormerFusion(dims_in=hidden_dim, stage_idx=stage_idx, **cfg)

            # 差异重注入门控：逐通道缩放，必须初始化为 0。
            # 形状 (1,1,1,C) 直接广播到 NHWC。
            if self.use_diff_gate:
                self.diff_gate = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        else:
            self.norm1_vi = norm_layer(hidden_dim)
            self.norm1_ir = norm_layer(hidden_dim)
            self.op = DSSM(
                d_model=hidden_dim,
                dropout=attn_drop_rate,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=dt_rank,
                disable_z=disable_z,
                **kwargs,
            )

        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=mlp_drop,
                channels_first=False,
            )
```

### 3.3 `DCFM._forward`

```python
    def _forward(self, x_vi: torch.Tensor, x_ir: torch.Tensor):
        if self.fus_type == "c2f":
            out_for_vi, out_for_ir = self.op(x_vi, x_ir)   # 均为 NHWC

            # 各自回支：out_for_vi 是「红外 query 取可见光信息」，加到 vi 支
            x_vi_ = x_vi + self.drop_path(out_for_vi)
            x_ir_ = x_ir + self.drop_path(out_for_ir)

            if self.use_diff_gate:
                # 差分重注入：补回 x_sub = x_vi - x_ir 这条通路
                # 初始化 λ=0 时该项为 0，整体严格等于 x_vi + x_ir
                x = x_vi_ + x_ir_ + self.diff_gate * (x_vi_ - x_ir_)
            else:
                x = x_vi_ + x_ir_
        else:
            x = self.op(self.norm1_vi(x_vi), self.norm1_ir(x_ir))  # B,H,W,C
            x = x_vi + x_ir + self.drop_path(x)

        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x
```

> `drop_path` 对两路各采样一次 mask，与逐个分支加残差的官方写法一致。初始化时 `out=0`，`drop_path(0)=0`，仍是严格恒等。

### 3.4 `DCFModule.__init__`

签名（`disable_z` 之后、`init_cfg` 之前）增加：

```python
        fus_type: str = "dssm",
        c2f_cfg: dict | None = None,
        c2f_start_level: int = 1,
        use_diff_gate: bool = True,
```

循环体改为：

```python
        self.fus_type = fus_type
        for i in range(self.num_ins):
            if fus_type == "c2f" and i >= c2f_start_level:
                cfg = dict(c2f_cfg)
                self.crossModalFusBlock.append(
                    DCFM(
                        hidden_dim=in_channels[i],
                        drop_path=drop_path,
                        mlp_ratio=mlp_ratio,
                        norm_layer=norm_layer,
                        fus_type="c2f",
                        stage_idx=i,
                        use_diff_gate=use_diff_gate,
                        c2f_cfg=dict(
                            dims_out=cfg["dims_out"][i],
                            num_heads=cfg["num_heads"][i],
                            n_groups=cfg.get("n_groups", [1] * self.num_ins)[i],
                            stride=cfg.get("stride", 3),
                            offset_range_factor=cfg.get("offset_range_factor", 2),
                            no_off=cfg.get("no_off", False),
                            attn_drop=cfg.get("attn_drop", 0.0),
                            proj_drop=cfg.get("proj_drop", 0.0),
                        ),
                    )
                )
            else:
                self.crossModalFusBlock.append(
                    DCFM(
                        hidden_dim=in_channels[i],
                        drop_path=drop_path,
                        mlp_ratio=mlp_ratio,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop_rate,
                        d_state=d_state,
                        dt_rank=dt_rank,
                        ssm_ratio=ssm_ratio,
                        act_layer=act_layer,
                        mlp_drop=mlp_drop,
                        sub_before=sub_before,
                        disable_z=disable_z,
                    )
                )
```

> `stage_idx=i` 必须显式传层级下标，不能写成 `enumerate` 的序号后再偏移。`ksizes=[9,7,5,3]` 按下标取卷积核，level 1/2/3 对应 7/5/3。

### 3.5 `DCFModule.forward` 不动

它已经是「NCHW → permute → NHWC 调 block → permute 回 NCHW」，`C2FormerFusion` 输入输出都是 NHWC，可以直接接上。

---

## 四、新建 `dmm_dc2f/configs/c2f_dc2f_1x_vedai.py`

整份内容如下：

```python
# DMM + DCFL + DC2F + C2Former
#
# 相对 dc2f_1x_vedai.py 只多一件事：把 fusblock 的 DSSM 换成 C2Former（level 1/2/3）。
# DC2F 的三处开关（snt / csc / bind_last）、分配超参、数据管线、训练策略全部原样继承。
_base_ = ['./dc2f_1x_vedai.py']

model = dict(
    fusblock=dict(
        type='DCFModule',
        in_channels=[96, 192, 384, 768],
        drop_path=0.2,
        ssm_ratio=2.0,
        d_state=16,
        dt_rank='auto',
        mlp_ratio=0.0,
        attn_drop_rate=0.0,
        mlp_drop=0.0,
        disable_z=False,
        # ---- 新增：切换到 C2Former 分支 ----
        fus_type='c2f',
        c2f_start_level=1,
        use_diff_gate=True,
        c2f_cfg=dict(
            dims_out=[36, 72, 144, 288],
            num_heads=[1, 3, 6, 12],
            n_groups=[1, 2, 3, 6],
            stride=3,
            offset_range_factor=2,
            no_off=False,
            attn_drop=0.0,
            proj_drop=0.0,
        ),
    ),
)
```

DroneVehicle 版同理，`_base_ = ['./dc2f_1x_dronevehicle.py']`，其余一字不改。

- `dims_out` / `num_heads` / `n_groups` 是**按 level 索引的列表**，长度必须等于 `len(in_channels)`。
- 第 0 项的 `36 / 1 / 1` 是占位，level 0 仍走 DSSM，不会被读取。
- level 1/2/3 → `72/144/288`，`head_dim = 24`，`num_heads` 3/6/12。
- `n_groups` 对应官方 stage2/3/4 的 2/3/6，要求 `dims_out % n_groups == 0`：72/2、144/3、288/6 均整除。
- `stride=3` 即 AFS 的 s，不要改。
- `use_diff_gate=True` 时每个 c2f 层多一个 `nn.Parameter`，长度 = 该层通道数，三级合计 **1344** 个（192+384+768）。设成 `False` 即退回「只有双向校准」的版本，用于消融对照。

**关于合并方式**：mmengine 对 dict 是**深合并**，子配置只写 `fus_type` 也能保留 base 里的 `ssm_ratio` 等字段；列表是整体替换，所以 `dims_out` / `num_heads` / `n_groups` 三个列表一起写全。

## 四之二、`dmm_dc2f/train_loop.py` 加 profile

`PROFILES` 里加一条，其余不动：

```python
    "vedai_c2f_dc2f": dict(
        label="VEDAI",
        variant="DMM + DC2F + C2Former",
        detail="DC2F 分配 + C2Former 融合（72/144/288）+ 差异重注入",
        config=os.path.join("code", "dmm_dc2f", "configs",
                            "c2f_dc2f_1x_vedai.py"),
        work_root=os.path.join("work_dirs", "c2f_dc2f_vedai_loop"),
        metrics_xlsx=os.path.join("work_dirs", "c2f_dc2f_vedai_metrics.xlsx"),
        default_data_root=os.path.join("datasets", "vedai"),
        env_var="DMM_VEDAI_ROOT",
        weights_dir=os.path.join("weights", "c2f_dc2f_vedai"),
        classes="6 类（car / truck / tractor / camping_car / van / pickup）",
    ),
```

---

## 五、验收（5 条）

**第 1 条在 CPU 上跑**：

```python
from mmengine.config import Config

cfg = Config.fromfile('dmm_dc2f/configs/c2f_dc2f_1x_vedai.py')
assert cfg.model.fusblock.fus_type == 'c2f', cfg.model.fusblock
assert cfg.model.fusblock.ssm_ratio == 2.0          # base 字段没丢
assert cfg.model.bbox_head_init.type == 'DC2FS2AHead'          # DC2F 还在
assert cfg.model.bbox_head_refine.type == 'DC2FS2ARefineHead'
assert cfg.model.train_cfg.init.assigner.snt is True
assert cfg.model.train_cfg.init.assigner.csc is True
assert cfg.model.train_cfg.init.assigner.bind_last is True
print('config 合并 OK：C2Former 已挂上，DC2F 三处开关未丢')
```

**第 2–5 条必须在 GPU 上跑**：level 0 走 DSSM，`selective_scan` 不支持 CPU。

```python
import torch
from mmrotate.models.layers.dmm import DCFModule

m = DCFModule(in_channels=[96, 192, 384, 768], drop_path=0.2, mlp_ratio=0.0,
              attn_drop_rate=0.0, mlp_drop=0.0, disable_z=False,
              fus_type="c2f", c2f_start_level=1, use_diff_gate=True,
              c2f_cfg=dict(dims_out=[36, 72, 144, 288], num_heads=[1, 3, 6, 12],
                           n_groups=[1, 2, 3, 6], stride=3, offset_range_factor=2)).cuda()

# 2) 参数量：level1-3 的 c2f 部分 = 8.142M（含 λ 共 1344 个），level0 的 DSSM ≈ 0.28M
print(sum(p.numel() for n, p in m.named_parameters() if "crossModalFusBlock.0." not in n) / 1e6)

# 3) 差异门控必须全零
print({n: p.abs().max().item() for n, p in m.named_parameters() if n.endswith("diff_gate")})

# 4) 形状（按 VEDAI 实际 640x512 折算）
xs_vi = [torch.randn(2, c, 640 // s, 512 // s, device="cuda")
         for c, s in zip([96, 192, 384, 768], [4, 8, 16, 32])]
xs_ir = [torch.randn_like(x) for x in xs_vi]
outs = m(xs_vi, xs_ir)
print([tuple(o.shape) for o in outs])

# 5) 初始等价：level1-3 的输出必须严格等于 x_vi[i] + x_ir[i]
for i in (1, 2, 3):
    diff = (outs[i] - (xs_vi[i] + xs_ir[i])).abs().max().item()
    print(f"level{i} max|diff| = {diff:.3e}")   # 必须全是 0.000e+00（level0 不检查）
```

全过才算移植正确。

- 第 1 条失败 → 父子配置路径写错。
- 第 3 条不是 0 → 某个 `DCFM` 的 `use_diff_gate` 没传下去。
- 第 5 条失败 → 先查 `C2FormerFusion._zero_init()` 有没有把 `vis_out[0]` / `lwir_out[0]` 的 **weight 和 bias** 都清零（只清 weight 会留下 `ReLU(bias) ≠ 0`）。`out` 恒为 0 时 `drop_path(0) = 0`，与训练态无关，不用切 `eval()`。

---

## 六、不要做的事

- 不要改 `DSSM`、`CrossScan_rgbt_k4`、`ssm_utils.py`、检测器 `dmm_s2anet.py`、`neck`、`head`。
- 不要把 `norm1_vi/norm1_ir` 保留在 c2f 分支里（建了不用会导致 DDP unused parameter 报错）。
- 不要给 `C2FormerFusion` 传 `fmap_size` / `q_size` / `kv_size` 的真实分辨率，传 `(1, 1)` 即可，`forward` 里不使用。
- 不要复制 `C2FormerResNet`、`ts_resnet.py`、`pretrain_weights/*.pth`。
- `diff_gate` 必须用 `torch.zeros` 初始化，不要用 `ones`、不要走默认初始化；也不要套 `sigmoid` / `tanh`——它是无约束的差分缩放，取负值合法（等价于反向注入）。
