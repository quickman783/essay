"""MBO experiment profile using the existing collaborator's 12-epoch loop."""
import importlib.util
from pathlib import Path


PROFILES = {
    'dronevehicle_dcfl_mbo': dict(
        label='DroneVehicle', variant='DMM + DCFL + instance-aware MBO',
        detail='Original C2FAssigner/S2A/le135/Focal+SmoothL1; RGB/IR auxiliary branches; FP32',
        pretrained_stem='dmm_dcfl_mbo_fast',
        config='add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_train.py',
        work_root='work_dirs/dmm_dcfl_mbo_fast_pretrained_dronevehicle_loop',
        metrics_xlsx='work_dirs/dmm_dcfl_mbo_fast_pretrained_dronevehicle_metrics.xlsx',
        default_data_root='datasets/prepared/dronevehicle',
        env_var='DMM_DV_ROOT',
        weights_dir='weights/dmm_dcfl_mbo_fast_pretrained_dronevehicle',
        classes='5 classes (car / truck / bus / van / freight-car)',
    ),
}


def load_engine():
    source = Path(__file__).resolve().parent / 'training' / 'train_loop.py'
    spec = importlib.util.spec_from_file_location('mbo_author_loop_engine', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.PROFILES.update(PROFILES)
    return engine


def main(dataset='dronevehicle_dcfl_mbo'):
    import sys
    engine = load_engine()
    if '--fast-assign' in sys.argv:
        sys.argv.remove('--fast-assign')  # Accepted for existing launch commands.
    return engine.main(dataset)
