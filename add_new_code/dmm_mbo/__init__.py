"""Instance-aware MBO for DMM + DCFL."""
from . import dcfl  # noqa: F401
from .detector import MBODMMDetector
__all__ = ["MBODMMDetector"]
