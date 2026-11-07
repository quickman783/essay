"""DCFL（Dynamic Coarse-to-Fine 标签分配）分支。

    gjsd_calculator.py  高斯 JS 散度度量（粗筛用）
    c2f_assigner.py     C2FAssigner 粗到细分配器
    dcfl_heads.py       把它接到 GauCho 版 S2A-Net 两个头上的损失入口
"""

from . import gjsd_calculator  # noqa: F401
from . import c2f_assigner  # noqa: F401
from . import dcfl_heads  # noqa: F401

__all__ = ['gjsd_calculator', 'c2f_assigner', 'dcfl_heads']
