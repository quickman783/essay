"""DC2F 的三处改动：SNT / CSC / BIND。

三处都作用在标签分配上，**不动网络结构、不增加参数、不改回归分支**：

    SNT  Shape-Normalized Threshold
         粗筛阈值不再是固定的 0.8，而是 min(0.8, 该 GT 形状能达到的 gjsd 上界)。
         形状上界本来就不低于 0.8 的目标阈值仍是 0.8，行为与 DCFL 一致；
         只有上界低于 0.8 的目标（小目标 / 极端长宽比）阈值才降低。

    CSC  Cross-Stage Constraint
         初始阶段一个正样本都没分到的目标，精修阶段把 dgmm 高斯掩码几乎放开
         （阈值 ``csc_free_thr``，默认 3.0），避免难目标被二次筛掉。
         其余目标的掩码与 DCFL 逐位相同。

    BIND Best-owner assIgnNment by gjsD
         把 DCFL 里「按 GT 编号循环、后者覆盖前者」的隐式归属写成显式的
         「编号最大者优先」。数学上与循环版逐位相同，只是不再依赖循环顺序。

**本包不修改 ``dmm_gaucho.dcfl`` 里的任何文件**：分配器是 ``C2FAssigner``
的子类，只覆写 ``assign()``；三个开关全为 False 时，输出与 DCFL 逐位相同。
"""

from . import prior  # noqa: F401
from . import dc2f_assigner  # noqa: F401
from . import dc2f_heads  # noqa: F401
from . import dc2f_raga  # noqa: F401

__all__ = ['prior', 'dc2f_assigner', 'dc2f_heads', 'dc2f_raga']
