"""Independent DMM + MBO ablation; reuse the 12-epoch training engine."""
import importlib.util
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'training/train_loop.py'
    spec = importlib.util.spec_from_file_location('mbo_baseline_loop_engine', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.PROFILES['dronevehicle_mbo'] = dict(
        label='DroneVehicle', variant='DMM + MBO (no DCFL)',
        detail='Original S2A/MaxIoUAssigner/le135/Focal+SmoothL1; three branches; FP32',
        config='add_new_code/dmm_mbo/configs/dmm_mbo_train.py',
        work_root='work_dirs/dmm_mbo_pretrained_dronevehicle_loop',
        metrics_xlsx='work_dirs/dmm_mbo_pretrained_dronevehicle_metrics.xlsx',
        default_data_root='datasets/prepared/dronevehicle', env_var='DMM_DV_ROOT',
        weights_dir='weights/dmm_mbo_pretrained_dronevehicle',
        classes='5 classes (car / truck / bus / van / freight-car)')
    return engine.main('dronevehicle_mbo')
