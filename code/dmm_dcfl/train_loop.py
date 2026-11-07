"""DMM + DCFL 对照组（不含 GauCho）的循环训练引擎。

训练机制的全部实现都在 ``dmm_gaucho.train_loop`` 里，这里只替换输出配置，
不复制代码 —— 对照组和实验组的训练/评估/记账流程必须完全一致，
否则表格里的差异会混进管道实现的影响。

只替换四类路径，保证不和正在跑的实验组互相覆盖：
    权重     work_dirs/dmm_dcfl_<数据集>_loop/run_<时间戳>/cycle_N/
    指标表   work_dirs/dmm_dcfl_<数据集>_metrics.xlsx
    状态     work_dirs/dmm_dcfl_<数据集>_loop/run_state.json
    起始权重 weights/dmm_dcfl_<数据集>/
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

PROFILES = {
    "vedai_dcfl": dict(
        label="VEDAI",
        variant="DMM + DCFL 对照组（基线回归，不含 GauCho）",
        config=os.path.join("code", "dmm_dcfl", "configs",
                            "dmm_dcfl_1x_vedai.py"),
        work_root=os.path.join("work_dirs", "dmm_dcfl_vedai_loop"),
        metrics_xlsx=os.path.join("work_dirs", "dmm_dcfl_vedai_metrics.xlsx"),
        default_data_root=os.path.join("datasets", "vedai"),
        env_var="DMM_VEDAI_ROOT",
        weights_dir=os.path.join("weights", "dmm_dcfl_vedai"),
        classes="6 类（car / truck / tractor / camping_car / van / pickup）",
    ),
    "dronevehicle_dcfl": dict(
        label="DroneVehicle",
        variant="DMM + DCFL 对照组（基线回归，不含 GauCho）",
        config=os.path.join("code", "dmm_dcfl", "configs",
                            "dmm_dcfl_1x_dronevehicle.py"),
        work_root=os.path.join("work_dirs", "dmm_dcfl_dronevehicle_loop"),
        metrics_xlsx=os.path.join("work_dirs",
                                  "dmm_dcfl_dronevehicle_metrics.xlsx"),
        default_data_root=os.path.join("data", "dv"),
        env_var="DMM_DV_ROOT",
        weights_dir=os.path.join("weights", "dmm_dcfl_dronevehicle"),
        classes="5 类（car / truck / bus / van / freight-car）",
    ),
}


def main(dataset):
    """循环训练入口。

    Args:
        dataset (str): ``vedai_dcfl`` 或 ``dronevehicle_dcfl``。
    """
    if dataset not in PROFILES:
        raise SystemExit(
            "未知数据集：{0}（可选 {1}）".format(dataset, sorted(PROFILES)))

    import dmm_gaucho.train_loop as engine

    # 换掉引擎里的数据集配置表；其余逻辑（学习率续训、best 副本、
    # 指标表格、权重不覆盖）原样复用
    engine.PROFILES.clear()
    engine.PROFILES.update(PROFILES)
    return engine.main(dataset)
