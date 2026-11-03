"""DMM 的共用训练策略与优化器，两个数据集都使用这份设置。"""

max_epochs = 12

# 每个 epoch 训练完做一次验证
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# 先线性预热 1000 次迭代，第 8 和第 11 个 epoch 结束时学习率乘 0.1
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[8, 11], gamma=0.1),
]

# AdamW，初始学习率 1e-4，权重衰减 0.05
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
)
