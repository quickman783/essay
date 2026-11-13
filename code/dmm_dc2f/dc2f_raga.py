"""DC2F + RAGA 的第二阶段（精修）检测头。

MRO 顺序 ``RAGAMixin -> DC2FS2ARefineHead``：

    RAGAMixin            覆盖 ``_init_layers`` / ``forward_single``
    DC2FS2ARefineHead     提供 ``loss_by_feat``（DC2F 三处改进的标签分配）

只装第二阶段：第一阶段的 ``bbox_head_init`` 输出会被精修覆盖，装在那里等于白加参数。
"""

from mmrotate.models.dense_heads.s2a_raga import RAGAMixin
from mmrotate.registry import MODELS

from .dc2f_heads import DC2FS2ARefineHead

__all__ = ['DC2FS2ARefineHeadRAGA']


@MODELS.register_module()
class DC2FS2ARefineHeadRAGA(RAGAMixin, DC2FS2ARefineHead):
    """S2A 精修头 + DC2F 标签分配 + RAGA。"""

    def __init__(self, *args, raga_cfg=None, **kwargs):
        self.raga_cfg = raga_cfg or {}
        super().__init__(*args, **kwargs)
