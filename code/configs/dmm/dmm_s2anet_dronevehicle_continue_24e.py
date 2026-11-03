"""Resume the completed 12-epoch run for 12 more epochs at its final LR.

Use --resume and load_from=the full epoch_12.pth checkpoint. The inherited
schedule ends at epoch 12, so its final learning rate remains unchanged.
"""

_base_ = './dmm_s2anet_1x_dronevehicle.py'

max_epochs = 24
train_cfg = dict(max_epochs=max_epochs)
