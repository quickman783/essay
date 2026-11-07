"""GauCho: Gaussian-Cholesky 定向框编解码。

数学部分与 GauCho (CVPR 2025) 的 mmrotate-0.x 实现保持一致，接口改写为
mmrotate 1.0 的 ``TASK_UTILS`` 形式，可以当作 ``DeltaXYWHTRBBoxCoder`` 的替代品。

与原来 ``DeltaXYWHTRBBoxCoder`` 的区别只在**参数化方式**：

    原实现：直接回归 (dx, dy, dlogw, dlogh, dθ)，θ 有周期性和边界不连续
    GauCho：回归 (dx, dy, dα, dβ, dγ)，解码成协方差阵
            Σ = [[a, c], [c, b]]，其中 a = α², b = γ² + β², c = αγ
            （即 Cholesky 因子 L = [[α, 0], [γ, β]] 的 L @ L.T）

训练时损失直接作用在 (cx, cy, a, b, c) 上（见 ``gaucho_loss.GauchoGDLoss``），
推理时再还原成 (cx, cy, w, h, θ)，对外接口与原来的 coder 完全一致。
"""

import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from mmdet.structures.bbox import get_box_tensor
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes

from .transforms import hbb2obb, norm_angle

__all__ = ['GauchoAnchorOBBDecoder']

# exp() 的自变量上下界，防止协方差指数级放大后 det() 溢出
_EXP_CLAMP = 10.0


@TASK_UTILS.register_module()
class GauchoAnchorOBBDecoder(BaseBBoxCoder):
    """把定向框表示成二维高斯，用 Cholesky 因子参数化后回归。

    Args:
        target_means (Sequence[float]): delta 的均值。默认全 0。
        target_stds (Sequence[float]): delta 的标准差。默认全 1。
        angle_version (str): 角度定义，``oc`` / ``le135`` / ``le90``。
        norm_factor (float, optional): 角度归一化系数，仅为与原 coder 参数对齐。
        edge_swap (bool): 是否交换宽高，仅为与原 coder 参数对齐。
        proj_xy (bool): 中心偏移是否按 anchor 角度投影。
        add_ctr_clamp (bool): 是否限制中心偏移。
        ctr_clamp (int): 中心偏移上限。
        horizontal (bool): anchor 是否为水平框（S2A-Net 用的是定向锚框，保持 False）。
        use_box_type (bool): 输出是否包装成 ``RotatedBoxes``。必须保持 True：
            推理阶段 mmdet 的 ``scale_boxes`` 只认 BaseBoxes，裸的 5 列 Tensor
            会被当成 4 列水平框去乘 scale_factor 而报维度错误。
            ``DeltaXYWHTRBBoxCoder`` 默认也是 True。
    """

    # 与 DeltaXYWHTRBBoxCoder 一致：回归分支输出 5 个通道
    encode_size = 5

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_version='le135',
                 norm_factor=None,
                 edge_swap=False,
                 proj_xy=True,
                 add_ctr_clamp=False,
                 ctr_clamp=32,
                 horizontal=False,
                 use_box_type=True,
                 **kwargs):
        super().__init__(use_box_type=use_box_type)
        self.means = target_means
        self.stds = target_stds
        self.angle_version = angle_version
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.proj_xy = proj_xy
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.horizontal = horizontal

    def encode(self, bboxes, gt_bboxes):
        """把 gt 编码成相对 anchor 的 delta。

        注意：GauCho 的训练走 ``reg_decoded_bbox=True`` 分支，目标直接用原始
        gt 框，所以这个函数只在把 coder 拿去配别的 head 时才会用到，
        α/β/γ 三个通道沿用原实现的占位写法（与官方代码一致）。
        """
        bboxes = get_box_tensor(bboxes)
        gt_bboxes = get_box_tensor(gt_bboxes)
        if bboxes.size(-1) == 4:
            bboxes = hbb2obb(bboxes, self.angle_version)

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 5
        assert gt_bboxes.size(-1) == 5
        if self.angle_version not in ('oc', 'le135', 'le90'):
            raise NotImplementedError
        return bbox2delta(bboxes, gt_bboxes, self.means, self.stds)

    def decode(self,
               bboxes,
               pred_bboxes: Tensor,
               max_shape=None,
               wh_ratio_clip: float = 16 / 1000,
               to_obb: bool = True):
        """把 delta 解码成框。

        Args:
            bboxes (Tensor): anchor，形状 (N, 5)。
            pred_bboxes (Tensor): 网络输出，形状 (N, 5)。
            max_shape (Sequence[int], optional): 图像尺寸，用于裁剪中心点。
            wh_ratio_clip (float): 宽高比裁剪阈值。
            to_obb (bool): True 时返回 (cx, cy, w, h, θ)；
                False 时返回高斯参数 (cx, cy, a, b, c)，训练算 GauCho 损失用。
        """
        bboxes = get_box_tensor(bboxes)
        assert pred_bboxes.size(0) == bboxes.size(0)
        if bboxes.size(-1) == 4:
            bboxes = hbb2obb(bboxes, self.angle_version)
        if self.angle_version not in ('oc', 'le135', 'le90'):
            raise NotImplementedError
        decoded = delta2bbox(
            bboxes, pred_bboxes, self.means, self.stds, max_shape,
            wh_ratio_clip, self.add_ctr_clamp, self.ctr_clamp,
            self.angle_version, self.norm_factor, self.edge_swap,
            self.proj_xy, to_obb, self.horizontal)

        # 推理阶段必须包成 RotatedBoxes，否则 mmdet 的 scale_boxes 会当成
        # 4 列水平框处理；训练阶段（to_obb=False）返回的高斯参数保持裸 Tensor
        if to_obb and self.use_box_type and decoded.size(-1) == 5:
            return RotatedBoxes(decoded)
        return decoded


def bbox2delta(proposals: Tensor,
               gts: Tensor,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.)) -> Tensor:
    """anchor -> gt 的 delta（仅中心与角度可用，α/β/γ 为占位值）。"""
    assert proposals.size() == gts.size()
    proposals = proposals.float()
    gts = gts.float()
    px, py, pw, ph, pa = proposals.unbind(dim=-1)
    gx, gy, _, _, _ = gts.unbind(dim=-1)

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    # 沿用官方实现：α/β/γ 三个通道直接用 anchor 的尺寸和角度占位
    dw = pw
    dh = ph
    da = pa

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    return deltas.sub_(means).div_(stds)


def delta2bbox(rois: Tensor,
               deltas: Tensor,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip: float = 16 / 1000,
               add_ctr_clamp: bool = False,
               ctr_clamp: int = 32,
               angle_version: str = 'le135',
               norm_factor=None,
               edge_swap: bool = False,
               proj_xy: bool = False,
               to_obb: bool = True,
               horizontal: bool = False) -> Tensor:
    """delta -> 框。``to_obb=False`` 时返回 Cholesky 高斯参数 (cx, cy, a, b, c)。"""
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dalpha = denorm_deltas[:, 2::5]
    dbeta = denorm_deltas[:, 3::5]
    dgamma = denorm_deltas[:, 4::5]

    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    pw = rois[:, 2].unsqueeze(1).expand_as(dalpha)
    ph = rois[:, 3].unsqueeze(1).expand_as(dbeta)
    pa = rois[:, 4].unsqueeze(1).expand_as(dgamma)

    # 半宽半高，也是一维高斯的方差
    eig_w = 0.25 * pw.square()
    eig_h = 0.25 * ph.square()
    sqrt_eig_w = 0.5 * pw
    sqrt_eig_h = 0.5 * ph

    if horizontal:
        galpha = sqrt_eig_w * dalpha.exp()
        gbeta = sqrt_eig_h * dbeta.exp()
        ggamma = torch.max(
            torch.min(sqrt_eig_w, sqrt_eig_h),
            torch.abs(sqrt_eig_w - sqrt_eig_h)) * dgamma
    else:
        # anchor 自身转成高斯后，取其 Cholesky 因子作为基准
        p_cova = torch.cos(pa).square() * eig_w + torch.sin(pa).square() * eig_h
        p_covc = 0.5 * torch.sin(2 * pa) * (eig_w - eig_h)
        palpha = torch.sqrt(p_cova)
        pgamma = p_covc / palpha
        pbeta = (sqrt_eig_h * sqrt_eig_w).clamp(1.0) / palpha

        # exp() 无上界，dalpha 一旦被梯度推大就会指数放大协方差，
        # 继而让 det() 溢出成 inf，算出 inf - inf = NaN。
        # 夹到 ±_EXP_CLAMP 后 exp 的范围是 [4.5e-5, 22026]，足够覆盖
        # 真实的宽高变化（扩大 22026 倍），又不会溢出。
        dalpha = dalpha.clamp(min=-_EXP_CLAMP, max=_EXP_CLAMP)
        dbeta = dbeta.clamp(min=-_EXP_CLAMP, max=_EXP_CLAMP)
        galpha = palpha * dalpha.exp()
        gbeta = pbeta * dbeta.exp()
        ggamma = pgamma + sqrt_eig_h.clamp(1.0) * dgamma

    if proj_xy:
        gx = dx * pw * torch.cos(pa) - dy * ph * torch.sin(pa) + px
        gy = dx * pw * torch.sin(pa) + dy * ph * torch.cos(pa) + py
    else:
        gx = px + pw * dx
        gy = py + ph * dy

    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)

    # 由 Cholesky 因子还原协方差：Σ = [[a, c], [c, b]]
    ga = galpha ** 2
    gb = gbeta ** 2 + ggamma ** 2
    gc = galpha * ggamma

    if to_obb:
        # 特征值分解，最大特征值对应宽
        delta = torch.sqrt(4 * torch.abs(gc).square() + (ga - gb).square())
        eig2 = 0.5 * (ga + gb + delta)
        # eig1 用特征值之积（= 行列式）除以 eig2 求，而不是 0.5(ga+gb-delta)：
        # 后者在 ga 远大于 gb 时是两个大数相减，fp32 下会直接把 gh 算成 0。
        eig1 = (ga * gb - gc.square()).clamp(min=0) / eig2.clamp(min=1e-12)
        gw = 2 * torch.sqrt(eig2)
        gh = 2 * torch.sqrt(eig1)
        # 用 atan2 而不是 atan(gc/(eig2-gb))：后者在协方差接近圆形
        # （ga==gb 且 gc==0）时是 atan(0/0)=NaN。S2A-Net 的锚框是正方形，
        # 初始化时恰好落在这个退化点上，训练中也会反复遇到，
        # 一旦产生 NaN 就顺着梯度毁掉全部权重。
        # atan2(0, 0) 定义为 0，且整体连续、无除零。
        gt = 0.5 * torch.atan2(2 * gc, ga - gb)
        gt = norm_angle(gt, angle_version)
        return torch.stack([gx, gy, gw, gh, gt], dim=-1).view(deltas.size())

    # 训练分支：直接给高斯参数，避免绕回 (w, h, θ) 再转回协方差的精度损失
    return torch.stack([gx, gy, ga, gb, gc], dim=-1).view(deltas.size())
