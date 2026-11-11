"""CSC 用的跨阶段先验缓冲（按图存）。

初始阶段（bbox_head_init）写，精修阶段（bbox_head_refine）读，
内容是「每个 GT 在这一阶段被选出了几个正样本」。

**必须按图分开存**：``dc2f_loss_by_feat`` 是逐图调用 ``assign()`` 的，
如果只存一个 list，第 4 张图会读到第 1 张图的先验。

key 用 ``img_id``：它是一个 0 起的整数下标，在同一次 ``loss_by_feat``
调用里唯一。用文件名更稳，但 ``assign()`` 里拿不到。
"""

__all__ = ['write_prior', 'read_prior', 'clear_prior']

_PRIOR = {}


def write_prior(img_id, stats) -> None:
    """记下某张图每个 GT 的正样本数。"""
    if img_id is None:
        return
    _PRIOR[int(img_id)] = [int(v) for v in stats]


def read_prior(img_id):
    """取某张图的先验；没有就返回 None。"""
    if img_id is None:
        return None
    return _PRIOR.get(int(img_id))


def clear_prior() -> None:
    """清空缓冲。每轮训练开始时调一次，避免上一轮的脏数据。"""
    _PRIOR.clear()
