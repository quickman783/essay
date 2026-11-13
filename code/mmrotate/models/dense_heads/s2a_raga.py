"""RAGA (Refinement-stage Adaptive Geometric Alignment) 的精修头实现。

挂在 S2A-Net 第二阶段（``S2ARefineHead``）上，只覆盖 ``_init_layers`` 与
``forward_single``，不改 ``s2a_head.py``。三个子模块：

    TPM  Task-Prior Modulator         任务先验调制（cls / reg 各一套）
    GRR  Geometric Re-sampling Refiner 几何重采样精修（偏移场与采样权重解耦）
    SCG  Semantic Consistency Gate     语义一致性门控

相对原始骨架（CMFADet TGRS 2026 的 ATAH）的改动点：
    1. 偏移场与采样置信度拆成两个并行分支，避免单一卷积在同一特征空间里
       同时拟合「采哪里」和「采多少」；
    2. 先验系数引入可学习温度 ``tau``，调制锐度随训练自适应；
    3. 语义门控引入可学习基线 ``beta``，缓解固定 0.5 判定中心在类别不平衡下的偏差。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d

from mmrotate.registry import MODELS
from mmrotate.models.dense_heads.s2a_head import S2ARefineHead


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
        coef = torch.sigmoid(
            self.tau * self.la_conv2(F.relu(self.la_conv1(prior))))

        # 系数按「段」缩放共享卷积核，得到逐样本的 (C, S*C) 动态权重
        # coef 是 (b, S, 1, 1)，必须 reshape 成 (b, 1, S, 1) 才能和权重的段维对齐
        dyn = coef.view(b, 1, self.stacked_convs, 1) * \
            self.reduction_conv.conv.weight.view(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        dyn = dyn.view(b, self.feat_channels, self.in_channels)

        out = torch.einsum('boc,bchw->bohw', dyn, feat)
        return self.reduction_conv.act(self.reduction_conv.gn(out))


class GeometricResampler(nn.Module):
    """按预测的偏移场与采样权重，对特征做 3x3 重采样，再归一化。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.sample = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = ChannelNorm(out_channels)

    def forward(self, x, offset, weight):
        return self.norm(self.sample(x.contiguous(), offset, weight))


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


class RAGAMixin:
    """挂到 ``S2ARefineHead`` 上的 RAGA 分支。

    子类必须在 ``super().__init__()`` 之前设置 ``self.raga_cfg``，
    因为 ``_init_layers`` 会在基类的 ``__init__`` 里被调用。
    """

    def _init_layers(self):
        super()._init_layers()
        self._init_raga()

    def _init_raga(self):
        cfg = getattr(self, 'raga_cfg', None) or {}
        c = self.feat_channels          # or_conv 之后 / cls_convs 输出的通道数
        hidc = 2 * c                    # cls_feat 与 reg_feat 拼接后的通道数
        s = cfg.get('stacked_convs', 2)
        dr = cfg.get('down_rate', 16)

        self.raga_use_grr = cfg.get('use_grr', True)
        self.raga_use_scg = cfg.get('use_scg', True)

        self.tpm_cls = TaskPriorModulator(c, s, dr)
        self.tpm_reg = TaskPriorModulator(c, s, dr)
        self.reg_scale = nn.Parameter(torch.ones(1))

        if self.raga_use_grr:
            # 几何偏移场：每个采样点的 (dy, dx)，18 = 2 * 3 * 3
            self.offset_head = nn.Conv2d(hidc, 18, 3, padding=1)
            # 采样置信度场：每个采样点的重采样权重，9 = 3 * 3
            self.mask_head = nn.Conv2d(hidc, 9, 3, padding=1)
            self.geo_resampler = GeometricResampler(c, c)

        if self.raga_use_scg:
            self.semantic_gate = SemanticConsistencyGate(hidc, 4)

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


@MODELS.register_module()
class S2ARefineHeadRAGA(RAGAMixin, S2ARefineHead):
    """S2A-Net 第二阶段 + RAGA（不含 DC2F，纯 RAGA 对照组可用）。"""

    def __init__(self, *args, raga_cfg=None, **kwargs):
        self.raga_cfg = raga_cfg or {}
        super().__init__(*args, **kwargs)
