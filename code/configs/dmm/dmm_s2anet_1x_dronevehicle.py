# DMM on DroneVehicle (visible + infrared), trained from scratch.
# Prepare the data with:
#   python tools/data_process.py --dataset dronevehicle --src data/DroneVehicle --dst data/dv
#   python tools/VOC2DOTA.py    --dataset dronevehicle --src data/DroneVehicle --dst data/dv
_base_ = [
    "./_base_/dmm_s2anet_rgbt.py",
    "./_base_/schedule_dmm_1x.py",
    "../_base_/datasets/DroneVehicle.py",
    "../_base_/default_runtime.py",
]

# 注册自定义的实时进度条 hook
custom_imports = dict(imports=["mmrotate.hooks.progress_bar"], allow_failed_imports=False)

# 终端里单行刷新的进度条，例如：
# Epoch 1/12 [=========>----------] 137/8739 1m16s 556ms/step loss: 1.2841 lr: 5.00e-05 eta: 21:30:12
# 不装这个的话 default_runtime 里的 LoggerHook interval=100000，
# 要跑满 100000 次迭代才会打印第一行（batch=2 时约 11 个 epoch，等于全程看不到输出）
custom_hooks = [dict(type="ProgressBarHook", bar_width=24)]

# classes: car, truck, bus, van, freight-car
num_classes = 5

model = dict(
    bbox_head_init=dict(num_classes=num_classes),
    bbox_head_refine=dict(num_classes=num_classes),
)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2, save_best="auto"),
)
