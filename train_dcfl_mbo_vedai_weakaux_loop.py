"""DMM + DCFL + stage-decayed MBO on VEDAI; run from the essay repository."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT / 'code'
EXTENSION_ROOT = ROOT / 'add_new_code'
BASE_ROOT = CODE_ROOT
if not (BASE_ROOT / 'mmrotate/__init__.py').is_file():
    BASE_ROOT = CODE_ROOT / 'DMM-main'
os.environ['DMM_BASE_ROOT'] = str(BASE_ROOT)
sys.path.insert(0, str(BASE_ROOT))
sys.path.insert(0, str(EXTENSION_ROOT))
os.environ['PYTHONPATH'] = os.pathsep.join(
    (str(EXTENSION_ROOT), str(BASE_ROOT), os.environ.get('PYTHONPATH', '')))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location(
        'dcfl_mbo_vedai_weakaux_entry',
        EXTENSION_ROOT / 'dmm_mbo/vedai_dcfl_weakaux_loop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raise SystemExit(module.main())
