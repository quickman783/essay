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

# classes: car, truck, bus, van, freight-car
num_classes = 5

model = dict(
    bbox_head_init=dict(num_classes=num_classes),
    bbox_head_refine=dict(num_classes=num_classes),
)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2, save_best="auto"),
)
