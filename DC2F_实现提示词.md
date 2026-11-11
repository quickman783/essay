# DC2F：三处代码改动

## 1. 阈值按目标形状归一化

**原来**，`C2FAssigner.assign()` 里：

```python
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0
```

**改成**：

```python
        # ===== DC2F / SNT =====
        if self.snt:
            thr = self.sim_upper(gt_bboxes).clamp(max=self.thr_ratio)   # τ_i = min(0.8, s_max_i)
            pass_thr = (overlaps >= thr[:, None]).any(dim=0)            # (N,)
            assigned_gt_inds[(max_overlaps >= 0) & (~pass_thr)] = 0
        else:
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0
```

**`C2FAssigner` 里加方法**：

```python
    def sim_upper(self, gt_bboxes):
        """每个 GT 在当前锚框尺寸下能达到的 gjsd 相似度上界，返回 (G,)。"""
        a2 = float(self.anchor_size) ** 2
        w = gt_bboxes[:, 2].clamp(min=1e-3)
        h = gt_bboxes[:, 3].clamp(min=1e-3)
        m1 = 1.0 / (2.0 / a2 + 2.0 / w.square())
        m2 = 1.0 / (2.0 / a2 + 2.0 / h.square())
        jsd = 0.5 * ((a2 * w * h / 16.0) / (m1 * m2)).log().clamp(min=0)
        return 1.0 / (1.0 + jsd)
```

自检（`a=32`）：`32×32 → 1.000`、`20×20 → 0.904`、`9×20 → 0.725`。

`clamp(max=0.8)` 是关键：形状上界本来就不低于 0.8 的目标，阈值仍是 0.8，行为与原版完全一致；只有上界低于 0.8 的目标，阈值才降到它自己的上界。

**`__init__` 加**：`snt: bool = False`、`thr_ratio: float = 0.8`、`anchor_size: float = 32.0`。

---

## 2. 跨级先验：只救零正样本的目标

**先验缓冲**，按图存：

```python
_PRIOR = {}

def write_prior(img_id, stats): _PRIOR[int(img_id)] = list(stats)
def read_prior(img_id):        return _PRIOR.get(int(img_id))
```

必须按图分：`dcfl_loss_by_feat` 是逐图调 `assign()` 的，只存一个 list 会让第 4 张图读到第 1 张图的先验。

**`dcfl_loss_by_feat` 加参数** `stage: str = None`，两个子类分别传 `'init'` 和 `'refine'`。在 `assign_result = head.assigner.assign(...)` 之后：

```python
        # ===== DC2F / CSC =====
        if stage == 'init' and getattr(head.assigner, 'csc', False):
            inds = assign_result.gt_inds
            write_prior(img_id, [int((inds == gi + 1).sum())
                                 for gi in range(len(gt_instances.bboxes))])
```

**`assign()` 里替换 `inside_flag`**（DGMM 那段）：

```python
            # ===== DC2F / CSC =====
            if self.csc:
                thr_vec = torch.full((num_gts,), self.gauss_thr,
                                     device=gt_bboxes.device, dtype=torch.float32)
                prior = read_prior(self._cur_img_id)
                if prior is not None and len(prior) == num_gts:
                    zero = torch.tensor([n == 0 for n in prior],
                                        device=gt_bboxes.device, dtype=torch.bool)
                    thr_vec = torch.where(zero, torch.full_like(thr_vec, self.csc_free_thr),
                                          thr_vec)
                inside_flag = gaussian >= torch.exp(-thr_vec)[:, None]
            else:
                inside_flag = gaussian >= torch.exp(
                    torch.tensor([-self.gauss_thr])).to(device1)
```

只对初始阶段一个正样本都没分到的目标，把高斯掩码放宽到 `csc_free_thr`（默认 3.0，等于几乎不删）。其余目标的掩码与原版逐位相同。

`assign()` 拿不到 `img_id`，加一个 `img_id=None` 的关键字参数（不动前两个位置参数），调用处传 `img_id=img_id`，内部先 `self._cur_img_id = img_id`。

**`__init__` 加**：`csc: bool = False`、`csc_free_thr: float = 3.0`。

---

## 3. 归属绑定：把遍历顺序显式化

**原来**：

```python
        for i in range(num_gts):
            for j in range(k):
                max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                assigned_gt_inds[max_overlap_inds] = i + 1
```

一个锚框同时进了多个 GT 的候选集时，最终归属取决于循环顺序（编号大的 GT 覆盖编号小的）。**改成显式的"编号最大者优先"**：

```python
        # ===== DC2F / BIND =====
        if self.bind_last:
            kth = gt_max_overlaps[:, -1]                                    # (G,) 第 k 名
            in_topk = overlaps >= kth[:, None]                              # (G, N)
            gi = torch.arange(num_gts, device=overlaps.device)[:, None].expand_as(in_topk)
            last = torch.where(in_topk, gi, -1).max(dim=0).values           # (N,)
            sel = last >= 0
            assigned_gt_inds[sel] = last[sel] + 1
        else:
            for i in range(num_gts):
                for j in range(k):
                    max_overlap_inds = overlaps[i, :] == gt_max_overlaps[i, j]
                    assigned_gt_inds[max_overlap_inds] = i + 1
```

`>= kth` 与原来逐 j 比较 `== gt_max_overlaps[i, j]` 得到的集合相同（第 k 名处并列的锚框两边都会纳入），所以输出与循环版**逐位相同**，只是把隐式的循环覆盖写成了显式的 argmax。

**`__init__` 加**：`bind_last: bool = False`。

---

## 开关汇总

`C2FAssigner.__init__` 新增六个带默认值的关键字参数：

```python
snt: bool = False, thr_ratio: float = 0.8, anchor_size: float = 32.0,
csc: bool = False, csc_free_thr: float = 3.0,
bind_last: bool = False,
```

全为 `False` 时 `assign()` 的输出与改动前逐位相同（`torch.equal` 验证），所以每处都写成 `if` 分支、`else` 保留原语句原文。

三处改动对分配结果的影响：BIND 逐位相同；SNT 只改"上界达不到 0.8 的目标"被判背景还是忽略；CSC 只改零正样本目标的 DGMM 掩码。预测指标与 DCFL 持平。
