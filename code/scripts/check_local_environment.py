"""Record installed DMM dependencies and probe native operations without changing the environment."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ['torch', 'torchvision', 'numpy', 'opencv-python', 'Pillow', 'mmcv',
            'mmcv-lite', 'mmengine', 'mmdet', 'timm', 'einops', 'fvcore',
            'triton', 'selective-scan']
PROBES = {
    'torch_cuda': 'import torch,json; print(json.dumps(dict(version=torch.__version__,cuda_build=torch.version.cuda,available=torch.cuda.is_available(),devices=torch.cuda.device_count())))',
    'mmcv_ops': 'from mmcv.ops import nms_rotated,box_iou_rotated; print("OK")',
    'selective_scan_oflex': 'import selective_scan_cuda_oflex; print("OK")',
    'triton': 'import triton; print(triton.__version__)',
    'model_import': 'import sys; sys.path.insert(0,' + repr(str(ROOT / 'code/DMM-main')) + '); from mmrotate.utils import register_all_modules; register_all_modules(); print("OK")',
}


def inspect(interpreter):
    version_probe = ('import json,sys;from importlib.metadata import version,PackageNotFoundError;'
                     'out={"python":sys.version,"executable":sys.executable};'
                     '\nfor name in ' + repr(PACKAGES) + ':\n try: out[name]=version(name)'
                     '\n except PackageNotFoundError: out[name]=None\nprint(json.dumps(out))')
    result = subprocess.run([str(interpreter), '-c', version_probe], capture_output=True, text=True, timeout=60)
    output = json.loads(result.stdout)
    output['probes'] = {}
    for name, code in PROBES.items():
        try:
            probe = subprocess.run([str(interpreter), '-c', code], capture_output=True, text=True, timeout=60)
            output['probes'][name] = dict(ok=probe.returncode == 0, stdout=probe.stdout.strip(), stderr=probe.stderr.strip())
        except subprocess.TimeoutExpired:
            output['probes'][name] = dict(ok=False, stderr='Timed out after 60 seconds')
    return output


if __name__ == '__main__':
    interpreters = [Path(sys.executable)] + sorted((Path.home() / '.conda/envs').glob('*/python.exe'))
    report = dict(platform=platform.platform(), environments=[inspect(p) for p in interpreters])
    target = ROOT / 'results/environment_audit.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(target))
