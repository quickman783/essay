"""GauCho 的回归损失：在高斯参数上算 GWD / KLD / BCD。

和 ``mmrotate.models.losses.GDLoss_v1`` 的唯一区别是：``pred`` 传入的是
网络解码出来的 Cholesky 高斯参数 ``(cx, cy, a, b, c)``，而不是 ``(cx, cy, w, h, θ)``。
其余（mask、reduction、权重处理）与官方版本逐行一致，便于和基线做公平对照。

GauCho 原文默认用 ``gwd``（Gaussian Wasserstein Distance），``kld`` 也可以试。
"""

from copy import deepcopy

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss

from mmrotate.models.losses.gaussian_dist_loss_v1 import (
    bcd_loss, gwd_loss, kld_loss, xy_wh_r_2_xy_sigma)
from mmrotate.registry import MODELS

from .transforms import gaussian_prediction_2_xy_sigma

__all__ = ['GauchoGDLoss']


@MODELS.register_module()
class GauchoGDLoss(nn.Module):
    """高斯距离损失，预测值以 Cholesky 参数形式给出。

    Args:
        loss_type (str): ``gwd`` / ``kld`` / ``bcd``。
        fun (str): 距离的后处理，``sqrt`` / ``log1p`` / ``''``。
        tau (float): 距离的缩放系数。
        reduction (str): ``none`` / ``sum`` / ``mean``。
        loss_weight (float): 损失权重。
        gaussian_prediction (bool): True 表示 ``pred`` 是 (x, y, a, b, c)；
            False 表示 ``pred`` 也是 (x, y, w, h, θ)，此时等价于官方 GDLoss_v1。
    """

    BAG_GD_LOSS = {'kld': kld_loss, 'bcd': bcd_loss, 'gwd': gwd_loss}

    def __init__(self,
                 loss_type: str,
                 fun: str = 'sqrt',
                 tau: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 gaussian_prediction: bool = True,
                 **kwargs):
        super().__init__()
        assert reduction in ('none', 'sum', 'mean')
        assert fun in ('log1p', 'sqrt', '')
        assert loss_type in self.BAG_GD_LOSS, \
            'loss_type 只能是 {0}'.format(sorted(self.BAG_GD_LOSS))
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = xy_wh_r_2_xy_sigma
        self.fun = fun
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.gaussian_prediction = gaussian_prediction
        self.kwargs = kwargs

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor = None,
                avg_factor: int = None,
                reduction_override: str = None,
                **kwargs) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()

        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        mask = (weight > 0).detach()
        pred = pred[mask]
        target = target[mask]
        kept_weight = weight[mask]

        if self.gaussian_prediction:
            pred = gaussian_prediction_2_xy_sigma(pred)
        else:
            pred = self.preprocess(pred)
        # 目标始终是原始 gt 框 (cx, cy, w, h, θ)
        target = self.preprocess(target)

        # gwd_loss / kld_loss / bcd_loss 返回的是逐框损失，这里做标准归约，
        # 与 mmrotate 里带 @weighted_loss 装饰器的 GDLoss 行为一致。
        # avg_factor 至少取 1：整批没有正样本时 0/0 会得到 NaN，
        # 而 NaN 会顺着梯度把整份权重毁掉。
        losses = self.loss(pred, target, fun=self.fun, tau=self.tau, **_kwargs)
        if losses.numel() == 0:
            return pred.sum() * 0.0
        loss = weight_reduce_loss(
            losses, kept_weight, reduction, max(float(avg_factor or 0), 1.0))
        return loss * self.loss_weight
