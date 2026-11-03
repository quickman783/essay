# DMM 在 VEDAI 上训练（可见光 + 红外双模态，从头训练）
#
# 数据准备命令（在仓库根目录执行）：
#   python code/tools/data_process.py --dataset vedai --src data/VEDAI --dst datasets/vedai
#   python code/tools/VOC2DOTA.py     --dataset vedai --src data/VEDAI --dst datasets/vedai \
#       --raw_annotations data/VEDAI/Annotations512
_base_ = [
    "./_base_/dmm_s2anet_rgbt.py",       # 模型结构
    "./_base_/schedule_dmm_1x.py",       # 训练策略：12 epoch、AdamW 1e-4
    "../_base_/datasets/VEDAI.py",       # 数据集与数据增强管线
    "../_base_/default_runtime.py",      # 日志、可视化等运行时设置
]

# 注册自定义的实时进度条 hook
custom_imports = dict(imports=["mmrotate.hooks.progress_bar"], allow_failed_imports=False)

# 终端里单行刷新的进度条，例如：
# Epoch 1/12 [=========>----------] 137/813 16s 118ms/step loss: 1.2841 lr: 1.00e-04 eta: 1:19:20
custom_hooks = [dict(type="ProgressBarHook", bar_width=24)]

# 类别：car, truck, tractor, camping_car, van, pickup
num_classes = 6

model = dict(
    bbox_head_init=dict(num_classes=num_classes),
    bbox_head_refine=dict(num_classes=num_classes),
)

# 每个 epoch 存一次权重，最多保留最近 2 个，另外单独保存 mAP 最高的那个
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2, save_best="auto"),
)
