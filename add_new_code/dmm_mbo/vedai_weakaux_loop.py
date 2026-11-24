"""Independent DMM + stage-decayed MBO on VEDAI; reuse the 12-epoch engine."""
import importlib.util
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'training/train_loop.py'
    spec = importlib.util.spec_from_file_location(
        'mbo_vedai_weakaux_engine', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    # 引擎的 --pretrained 恒为 True，会检查 code/weights/vssm/*.pth
    # （本地不存在，直接 SystemExit）。它不参与建模，压回 False 绕开。
    _engine_parse_cli = engine.parse_cli

    def _vedai_parse_cli():
        options = _engine_parse_cli()
        options['pretrained'] = False
        return options

    engine.parse_cli = _vedai_parse_cli

    engine.PROFILES['vedai_mbo_weakaux'] = dict(
        label='VEDAI', variant='DMM + MBO (stage-decayed aux)',
        detail='6 classes; three branches; pre-fork; refine aux scale 0',
        config='add_new_code/dmm_mbo/configs/dmm_mbo_vedai_weakaux.py',
        work_root='work_dirs/dmm_mbo_vedai_weakaux',
        metrics_xlsx='work_dirs/dmm_mbo_vedai_weakaux_metrics.xlsx',
        default_data_root='datasets/vedai', env_var='DMM_VEDAI_ROOT',
        weights_dir='weights/dmm_mbo_vedai_weakaux',
        classes='6 classes (car / truck / tractor / camping_car / van / pickup)')
    return engine.main('vedai_mbo_weakaux')
