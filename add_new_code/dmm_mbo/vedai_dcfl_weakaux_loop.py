"""Independent DMM + DCFL + stage-decayed MBO on VEDAI."""
import importlib.util
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'training/train_loop.py'
    spec = importlib.util.spec_from_file_location(
        'dcfl_mbo_vedai_weakaux_engine', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    _engine_parse_cli = engine.parse_cli

    def _vedai_parse_cli():
        options = _engine_parse_cli()
        options['pretrained'] = False
        return options

    engine.parse_cli = _vedai_parse_cli

    engine.PROFILES['vedai_dcfl_mbo_weakaux'] = dict(
        label='VEDAI', variant='DMM + DCFL + MBO (stage-decayed aux)',
        detail='6 classes; C2FAssigner; three branches; refine aux scale 0',
        config='add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_vedai_weakaux.py',
        work_root='work_dirs/dmm_dcfl_mbo_vedai_weakaux',
        metrics_xlsx='work_dirs/dmm_dcfl_mbo_vedai_weakaux_metrics.xlsx',
        default_data_root='datasets/vedai', env_var='DMM_VEDAI_ROOT',
        weights_dir='weights/dmm_dcfl_mbo_vedai_weakaux',
        classes='6 classes (car / truck / tractor / camping_car / van / pickup)')
    return engine.main('vedai_dcfl_mbo_weakaux')
