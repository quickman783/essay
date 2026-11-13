"""DC2F 的循环训练配置表。

训练机制（进度条、学习率跨次保留、指标表格、权重只增不删）全部复用
``dmm_gaucho.train_loop``，这里只提供 DC2F 自己的路径，保证不和
DCFL / GauCho 那些正在跑或已跑完的实验互相覆盖。

本文件同时管两套方案：
    vedai_dc2f / dronevehicle_dc2f           DMM + DC2F
    vedai_raga_dc2f / dronevehicle_raga_dc2f DMM + DC2F + RAGA（只换精修头）
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

PROFILES = {
    "vedai_dc2f": dict(
        label="VEDAI",
        variant="DMM + DC2F",
        detail="基线 L1 回归 + SmoothL1 损失，DC2F 分配（SNT+CSC+BIND）",
        config=os.path.join("code", "dmm_dc2f", "configs",
                            "dc2f_1x_vedai.py"),
        work_root=os.path.join("work_dirs", "dc2f_vedai_loop"),
        metrics_xlsx=os.path.join("work_dirs", "dc2f_vedai_metrics.xlsx"),
        default_data_root=os.path.join("datasets", "vedai"),
        env_var="DMM_VEDAI_ROOT",
        weights_dir=os.path.join("weights", "dc2f_vedai"),
        classes="6 类（car / truck / tractor / camping_car / van / pickup）",
    ),
    "dronevehicle_dc2f": dict(
        label="DroneVehicle",
        variant="DMM + DC2F",
        detail="基线 L1 回归 + SmoothL1 损失，DC2F 分配（SNT+CSC+BIND）",
        config=os.path.join("code", "dmm_dc2f", "configs",
                            "dc2f_1x_dronevehicle.py"),
        work_root=os.path.join("work_dirs", "dc2f_dronevehicle_loop"),
        metrics_xlsx=os.path.join("work_dirs", "dc2f_dronevehicle_metrics.xlsx"),
        default_data_root=os.path.join("data", "dv"),
        env_var="DMM_DV_ROOT",
        weights_dir=os.path.join("weights", "dc2f_dronevehicle"),
        classes="5 类（car / truck / bus / van / freight-car）",
    ),
    "vedai_raga_dc2f": dict(
        label="VEDAI",
        variant="DMM + DC2F + RAGA",
        detail="DC2F 分配 + RAGA 精修头（TPM + GRR + SCG）",
        config=os.path.join("code", "dmm_dc2f", "configs",
                            "raga_dc2f_1x_vedai.py"),
        work_root=os.path.join("work_dirs", "raga_dc2f_vedai_loop"),
        metrics_xlsx=os.path.join("work_dirs", "raga_dc2f_vedai_metrics.xlsx"),
        default_data_root=os.path.join("datasets", "vedai"),
        env_var="DMM_VEDAI_ROOT",
        weights_dir=os.path.join("weights", "raga_dc2f_vedai"),
        classes="6 类（car / truck / tractor / camping_car / van / pickup）",
    ),
    "dronevehicle_raga_dc2f": dict(
        label="DroneVehicle",
        variant="DMM + DC2F + RAGA",
        detail="DC2F 分配 + RAGA 精修头（TPM + GRR + SCG）",
        config=os.path.join("code", "dmm_dc2f", "configs",
                            "raga_dc2f_1x_dronevehicle.py"),
        work_root=os.path.join("work_dirs", "raga_dc2f_dronevehicle_loop"),
        metrics_xlsx=os.path.join(
            "work_dirs", "raga_dc2f_dronevehicle_metrics.xlsx"),
        default_data_root=os.path.join("data", "dv"),
        env_var="DMM_DV_ROOT",
        weights_dir=os.path.join("weights", "raga_dc2f_dronevehicle"),
        classes="5 类（car / truck / bus / van / freight-car）",
    ),
}


def main(dataset):
    """循环训练入口。"""
    if dataset not in PROFILES:
        raise SystemExit(
            "未知数据集：{0}（可选 {1}）".format(dataset, sorted(PROFILES)))

    import dmm_gaucho.train_loop as engine

    engine.PROFILES.clear()
    engine.PROFILES.update(PROFILES)
    if dataset == "vedai_dc2f":
        # 起点是 weights/dc2f_vedai/ 里那份 DMM + DCFL 权重：DCFL 的 cycle_7
        # 正是从它、用 1.5625e-06 再训 12 轮，所以这里按第 7 轮起步，
        # 起始权重 / 学习率 / epoch 数就和 DCFL 第 7 轮完全一样，只差分配器。
        # raga 那两条 profile 换了精修头，权重不通用，从第 1 轮正常起步。
        engine.BOOTSTRAP_CYCLE = 7
    return engine.main(dataset)
