"""GauCho 版 DMM：用 Cholesky 分解后的高斯参数回归定向框。

这是一个**旁路实现**，不改动 ``mmrotate`` 里的任何原有代码：

    dmm_gaucho/
    ├── transforms.py     mmrotate 1.0 里删掉的 hbb2obb 等几何工具
    ├── gaucho_coder.py   GauchoAnchorOBBDecoder（编解码）
    ├── gaucho_loss.py    GauchoGDLoss（在高斯参数上算 GWD/KLD/BCD）
    ├── gaucho_heads.py   GauchoS2AHead / GauchoS2ARefineHead（损失入口）
    ├── configs/          VEDAI / DroneVehicle 的训练配置
    └── train_loop.py     循环训练引擎（学习率续训、指标表格、权重备份）

配置里通过::

    custom_imports = dict(imports=['dmm_gaucho'], allow_failed_imports=False)

触发注册，所以训练/评估脚本都不需要额外 import 本包。
"""

from . import transforms  # noqa: F401
from . import gaucho_coder  # noqa: F401
from . import gaucho_loss  # noqa: F401
from . import gaucho_heads  # noqa: F401
from . import dcfl  # noqa: F401

__all__ = ['transforms', 'gaucho_coder', 'gaucho_loss', 'gaucho_heads', 'dcfl']
