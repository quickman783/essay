import os, sys
from pathlib import Path
ROOT = Path(r"e:/论文/essay")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "add_new_code"))
os.chdir(ROOT)
os.environ["DMM_BASE_ROOT"] = str(ROOT / "code")

import mmrotate
import dmm_mbo
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope("mmrotate")
from mmdet.registry import MODELS

cases = [
    ("groupd",     "add_new_code/dmm_mbo/configs/dmm_mbo_vedai_groupd.py"),
    ("vedai_gate", "add_new_code/dmm_mbo/configs/dmm_mbo_vedai_gate.py"),
    ("dv_gate",    "add_new_code/dmm_mbo/configs/dmm_mbo_dronevehicle_gate.py"),
]
for tag, path in cases:
    cfg = Config.fromfile(path)
    model = MODELS.build(cfg.model)
    gate = getattr(model, "mbo_gate", None)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    aux = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith("mbo_")) / 1e6
    print("%-12s gate=%-9s total=%.2fM aux=%.2fM work_dir=%s" % (
        tag, ("%.6f" % gate), total, aux, cfg.work_dir))
print("GATE_CHECK_DONE")