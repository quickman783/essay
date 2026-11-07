"""把 DCFL 的标签分配接到**基线** S2A-Net 的两个检测头上（不含 GauCho）。

和 ``dmm_gaucho.dcfl.dcfl_heads`` 的唯一区别就是父类：
    GauCho 版用 GauchoS2AHead / GauchoS2ARefineHead（高斯回归 + GWD）
    本模块用 S2AHead / S2ARefineHead（原样，L1 回归 + SmoothL1）

损失计算本身直接复用 ``dcfl_loss_by_feat``，一行没改，保证两组实验的
分配过程完全一致，差异只来自回归分支。

``dcfl_loss_by_feat`` 里 ``reg_decoded_bbox`` 的两个分支都已经处理：
    False（本对照组走这条）  目标由 bbox_coder.encode 编码，损失作用在 delta 上
    True （GauCho 走这条）    目标直接用 gt 框，损失作用在解码后的高斯参数上
"""

from typing import List, Optional

import torch
from mmdet.utils import InstanceList, OptInstanceList

from mmrotate.models.dense_heads.s2a_head import S2AHead, S2ARefineHead
from mmrotate.registry import MODELS

# 与 GauCho 版共用同一份分配+损失流程
from dmm_gaucho.dcfl.dcfl_heads import dcfl_loss_by_feat

__all__ = ['DCFLS2AHead', 'DCFLS2ARefineHead']


@MODELS.register_module()
class DCFLS2AHead(S2AHead):
    """S2A-Net 第一阶段 + 基线回归 + DCFL 标签分配。"""

    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        return dcfl_loss_by_feat(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore)


@MODELS.register_module()
class DCFLS2ARefineHead(S2ARefineHead):
    """S2A-Net 第二阶段（AlignConv 精修）+ 基线回归 + DCFL 标签分配。"""

    def loss_by_feat(self,
                     cls_scores: List[torch.Tensor],
                     bbox_preds: List[torch.Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     rois: Optional[List[torch.Tensor]] = None) -> dict:
        # 精修阶段把上一阶段输出的框当锚框，沿用原实现的约定
        assert rois is not None
        self.bboxes_as_anchors = rois
        return dcfl_loss_by_feat(
            self, cls_scores, bbox_preds, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore)
