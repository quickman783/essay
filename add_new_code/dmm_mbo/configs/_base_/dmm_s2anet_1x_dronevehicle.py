# DMM on DroneVehicle (visible + infrared), trained from scratch.
# Prepare the data with:
#   python tools/data_process.py --dataset dronevehicle --src data/DroneVehicle --dst data/dv
#   python tools/VOC2DOTA.py    --dataset dronevehicle --src data/DroneVehicle --dst data/dv
_base_ = ['dmm_s2anet_rgbt.py', 'schedule_dmm_1x.py', 'datasets/DroneVehicle.py', 'default_runtime.py']

# classes: car, truck, bus, van, freight-car
num_classes = 5

# Both heads already have five classes in the shared model config. A partial
# list override would replace the entire refine-head definition in MMEngine.

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2, save_best="auto"),
)
