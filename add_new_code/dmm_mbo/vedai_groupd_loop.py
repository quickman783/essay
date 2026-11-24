"""Independent DMM + MBO run on VEDAI, group D: pre-fork, no RMR."""
import importlib.util
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'training/train_loop.py'
    spec = importlib.util.spec_from_file_location('mbo_vedai_groupd_engine', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    # 引擎的 --pretrained 是 action='store_true' + default=True（恒为 True，命令行关不掉），
    # 它只做两件事：检查 code/weights/vssm/*.pth 是否存在（不存在就 SystemExit）、
    # dry-run 时打印一行。权重实际在仓库根 weights/vssm/，所以这里压回 False 绕开检查。
    # 注意：它不参与建模。VEDAI 到底加载不加载 ImageNet，只由配置里的
    # model.backbone_vi/ir.pretrained 决定（configs/_base_/dmm_s2anet_vedai_rgbt.py
    # 现已与 code/configs/dmm/_base_/dmm_s2anet_rgbt.py 一致，加载 VMamba-Tiny 权重）。
    _engine_parse_cli = engine.parse_cli

    def _vedai_parse_cli():
        options = _engine_parse_cli()
        options['pretrained'] = False
        return options

    engine.parse_cli = _vedai_parse_cli

    engine.PROFILES['vedai_mbo_groupd'] = dict(
        label='VEDAI-D', variant='DMM + MBO (pre-fork, no RMR)',
        detail='6 classes; three branches; pre-fork; per-GT reweighting; no RMR',
        config='add_new_code/dmm_mbo/configs/dmm_mbo_vedai_groupd.py',
        work_root='work_dirs/dmm_mbo_vedai_groupd',
        metrics_xlsx='work_dirs/dmm_mbo_vedai_groupd_metrics.xlsx',
        default_data_root='datasets/vedai', env_var='DMM_VEDAI_ROOT',
        weights_dir='weights/dmm_mbo_vedai_groupd',
        classes='6 classes (car / truck / tractor / camping_car / van / pickup)')
    return engine.main('vedai_mbo_groupd')
