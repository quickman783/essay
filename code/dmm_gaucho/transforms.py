"""GauCho 用到的几何 / 高斯工具函数。

``hbb2obb`` 在 mmrotate 1.0 里被删掉了（0.x 时代在 ``mmrotate.core.bbox.transforms``），
这里按原实现补回来，行为与 0.x 一致，只依赖 torch。

``gaussian_prediction_2_xy_sigma`` 把网络输出的 (x, y, a, b, c) 还原成均值 + 协方差：
协方差阵按 ``[[a, c], [c, b]]`` 排布，正好等于 Cholesky 因子 ``L = [[α, 0], [γ, β]]``
的 ``L @ L.T = [[α², αγ], [αγ, γ² + β²]]``。
"""

import numpy as np
import torch

# 角度归一化在 mmrotate 1.0 里还在，直接复用，避免两份实现出现偏差
from mmrotate.structures.bbox import norm_angle

__all__ = ['hbb2obb', 'gaussian_prediction_2_xy_sigma', 'norm_angle']


def hbb2obb(hbboxes, version='oc'):
    """水平框 [x1, y1, x2, y2] 转定向框 [cx, cy, w, h, angle]。

    Args:
        hbboxes (Tensor): 形状 (..., 4)。
        version (str): 角度定义，支持 ``oc`` / ``le135`` / ``le90``。

    Returns:
        Tensor: 形状 (..., 5)。
    """
    if version == 'oc':
        x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
        y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
        w = hbboxes[..., 2] - hbboxes[..., 0]
        h = hbboxes[..., 3] - hbboxes[..., 1]
        theta = x.new_zeros(*x.shape)
        # oc 的角度范围是 [0, pi/2)，横框对应 theta = pi/2 且宽高互换
        return torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)

    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)

    if version == 'le135':
        obboxes2 = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    elif version == 'le90':
        obboxes2 = torch.stack([x, y, h, w, theta - np.pi / 2], dim=-1)
    else:
        raise NotImplementedError(
            'angle version {0} is not supported'.format(version))
    return torch.where((w >= h)[..., None], obboxes1, obboxes2)


def gaussian_prediction_2_xy_sigma(xyabc):
    """把网络输出的高斯参数 (x, y, a, b, c) 转成 (中心, 协方差矩阵)。

    Args:
        xyabc (Tensor): 形状 (N, 5)，其中协方差为 ``[[a, c], [c, b]]``。

    Returns:
        tuple(Tensor, Tensor): ``xy`` 形状 (N, 2)，``sigma`` 形状 (N, 2, 2)。
    """
    xy = xyabc[..., :2]
    sigma = torch.stack(
        (xyabc[..., 2], xyabc[..., 4], xyabc[..., 4], xyabc[..., 3]),
        dim=-1).reshape(xyabc.shape[:-1] + (2, 2))
    return xy, sigma
