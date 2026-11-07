"""DMM + DCFL 对照组（**不含 GauCho**）。

消融实验的另一半：

    基线            L1 回归 + MaxIoU 分配        train_vedai_loop.py
    GauCho          高斯回归 + MaxIoU 分配        train_gaucho_vedai_loop.py
    DCFL（本包）    L1 回归 + DCFL 分配          train_dmm_dcfl_vedai_loop.py
    GauCho + DCFL   高斯回归 + DCFL 分配          train_gaucho_dcfl_vedai_loop.py

这样四组两两对照，就能把「高斯回归」和「DCFL 分配」各自的贡献拆开算。

DCFL 的分配器和 gjsd 度量**直接复用** ``dmm_gaucho.dcfl`` 里的那一份，
不复制代码 —— 对照组和实验组的标签分配逻辑必须逐字节相同，
否则差异里会混进实现细节的影响。本包只提供「基线检测头 + DCFL 分配」的粘合层。
"""

from dmm_gaucho.dcfl import c2f_assigner, gjsd_calculator  # noqa: F401
from . import dcfl_heads  # noqa: F401

__all__ = ['c2f_assigner', 'gjsd_calculator', 'dcfl_heads']
