"""GauCho 版的 S2A-Net 两个检测头。

只改**回归损失的入口**，其余（ORConv、AlignConv、锚框、分配器、推理流程）
全部沿用原来的 ``S2AHead`` / ``S2ARefineHead``：

    RotatedRetinaHead.loss_by_feat_single 里原本是
        bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)   # -> (cx,cy,w,h,θ)
    这里改成
        bbox_pred = self.bbox_coder.decode(anchors, bbox_pred, to_obb=False)  # -> (cx,cy,a,b,c)

配合 ``reg_decoded_bbox=True``（目标直接用原始 gt 框）和 ``GauchoGDLoss``，
就得到 GauCho 的回归分支。推理时仍走原来的 ``predict_by_feat``，
``decode`` 默认 ``to_obb=True``，输出的还是 (cx, cy, w, h, θ)，行为不变。
"""

from mmdet.structures.bbox import get_box_tensor
from torch import Tensor

from mmrotate.models.dense_heads.s2a_head import S2AHead, S2ARefineHead
from mmrotate.registry import MODELS

__all__ = ['GauchoS2AHead', 'GauchoS2ARefineHead']


class GauchoRegMixin:
    """把 ``loss_by_feat_single`` 的回归分支换成高斯参数解码。"""

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """计算单个尺度层级的损失。

        与 ``RotatedRetinaHead.loss_by_feat_single`` 的唯一差别在
        ``to_obb=False``：解码出高斯参数 (cx, cy, a, b, c) 后再算损失。
        """
        # 分类损失
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)

        # 回归损失
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_coder.encode_size)

        anchors = anchors.reshape(-1, anchors.size(-1))
        bbox_pred = self.bbox_coder.decode(anchors, bbox_pred, to_obb=False)
        bbox_pred = get_box_tensor(bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls, loss_bbox


@MODELS.register_module()
class GauchoS2AHead(GauchoRegMixin, S2AHead):
    """S2A-Net 的第一阶段（ODM），回归分支换成 GauCho。"""


@MODELS.register_module()
class GauchoS2ARefineHead(GauchoRegMixin, S2ARefineHead):
    """S2A-Net 的第二阶段（AlignConv 精修），回归分支换成 GauCho。

    ``loss_by_feat`` / ``get_anchors`` / ``predict_by_feat`` 全部沿用原实现：
    ``S2ARefineHead.loss_by_feat`` 里 ``super(RotatedRetinaHead, self)`` 最终会
    走到 ``AnchorHead.loss_by_feat``，它再按 MRO 调用到本类的
    ``loss_by_feat_single``，所以只需要这一个方法被换掉。
    """
