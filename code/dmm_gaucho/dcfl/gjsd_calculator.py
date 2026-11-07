"""DCFL 的粗筛选指标：把定向框看成二维高斯，用高斯 JS 散度算相似度。

对应原仓库 ``mmrotate/core/bbox/iou_calculators/rotate_metric_calculator.py``，
接口改写为 mmrotate 1.0 的 ``TASK_UTILS`` 形式。

与 ``RBboxOverlaps2D`` 的区别：多支持 ``mode='gjsd'``。
gjsd 越大表示两个高斯越像，这里换成相似度 ``1 / (1 + gjsd)``，
数值范围和 IoU 一致（0~1），所以后面 ``< 0.8`` 这类阈值可以直接沿用。

原实现在 CUDA 上会把中间结果转成 fp16 提速；这里保持 fp32，
因为 fp16 下 ``det()`` 容易下溢，而两者只差浮点误差。
"""

import torch
from mmcv.ops import box_iou_rotated

from mmrotate.registry import TASK_UTILS

__all__ = ['RBboxMetrics2D', 'xy_wh_r_2_xy_sigma', 'get_gjsd']

_EPS = 1e-3


def _inv_2x2(matrix):
    """2x2 矩阵求逆（伴随矩阵 / 行列式），与原始实现一致。"""
    inv = torch.stack((matrix[..., 1, 1], -matrix[..., 0, 1],
                       -matrix[..., 1, 0], matrix[..., 0, 0]),
                      dim=-1).reshape(*matrix.shape)
    det = matrix.det()
    while det.dim() < inv.dim():
        det = det.unsqueeze(-1)
    return inv / det


def xy_wh_r_2_xy_sigma(xywhr):
    """定向框 (cx, cy, w, h, θ) 转二维高斯 (均值, 协方差)。"""
    shape = xywhr.shape
    assert shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    rot = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    scale = 0.5 * torch.diag_embed(wh)
    sigma = rot.bmm(scale.square()).bmm(
        rot.permute(0, 2, 1)).reshape(shape[:-1] + (2, 2))
    return xy, sigma


def get_gjsd(pred, target, alpha=0.5):
    """高斯 Jensen-Shannon 散度，返回形状 (num_pred, num_target)。"""
    xy_p, sigma_p = pred
    xy_t, sigma_t = target

    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)
    inv_p = _inv_2x2(sigma_p)
    inv_t = _inv_2x2(sigma_t)

    # 广播成 (num_pred, num_target, ...)
    xy_p = xy_p[..., :, None, :2]
    xy_t = xy_t[..., None, :, :2]
    sigma_p = sigma_p[..., :, None, :2, :2]
    inv_p = inv_p[..., :, None, :2, :2]
    sigma_t = sigma_t[..., None, :, :2, :2]
    inv_t = inv_t[..., None, :, :2, :2]

    # 混合分布的精度矩阵与其逆
    mixed_inv = (1 - alpha) * inv_p + alpha * inv_t
    sigma_mixed = _inv_2x2(mixed_inv)

    xy_p = xy_p.unsqueeze(-1)
    xy_t = xy_t.unsqueeze(-1)
    mean_mixed = sigma_mixed.matmul(
        (1 - alpha) * inv_p.matmul(xy_p) + alpha * inv_t.matmul(xy_t))

    first_part = (
        (1 - alpha) * xy_p.permute(0, 1, 3, 2).matmul(inv_p).matmul(xy_p)
        + alpha * xy_t.permute(0, 1, 3, 2).matmul(inv_t).matmul(xy_t)
        - mean_mixed.permute(0, 1, 3, 2).matmul(mixed_inv).matmul(mean_mixed))
    second_part = (
        (sigma_p.det() ** (1 - alpha)) * (sigma_t.det() ** alpha)
        / sigma_mixed.det()).log()

    return 0.5 * (first_part.squeeze(-1).squeeze(-1) + second_part)


def rbbox_metrics(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """计算两组定向框之间的度量，返回 (num_bboxes1, num_bboxes2)。"""
    assert mode in ('iou', 'iof', 'gjsd', 'center_distance2')
    if mode != 'center_distance2':
        assert bboxes1.size(-1) == 5 or bboxes1.size(0) == 0
        assert bboxes2.size(-1) == 5 or bboxes2.size(0) == 0

    rows, cols = bboxes1.size(0), bboxes2.size(0)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if mode in ('iou', 'iof'):
        # 原始实现也先 clamp，避免极小的框让 CUDA 算子算出异常值
        clamped1 = bboxes1.detach().clone()
        clamped2 = bboxes2.detach().clone()
        clamped1[:, 2:4].clamp_(min=_EPS)
        clamped2[:, 2:4].clamp_(min=_EPS)
        return box_iou_rotated(clamped1, clamped2, mode, is_aligned)

    if mode == 'gjsd':
        g1 = xy_wh_r_2_xy_sigma(bboxes1)
        g2 = xy_wh_r_2_xy_sigma(bboxes2)
        return 1.0 / (1.0 + get_gjsd(g1, g2))

    center1 = bboxes1[..., :, None, :2]
    center2 = bboxes2[..., None, :, :2]
    whs = center1[..., :2] - center2[..., :2]
    return whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + 1e-6


@TASK_UTILS.register_module()
class RBboxMetrics2D:
    """定向框度量计算器，比 ``RBboxOverlaps2D`` 多一个 ``gjsd`` 模式。"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False,
                 **kwargs):
        return rbbox_metrics(bboxes1, bboxes2, mode=mode, is_aligned=is_aligned)

    def __repr__(self):
        return self.__class__.__name__ + '()'
