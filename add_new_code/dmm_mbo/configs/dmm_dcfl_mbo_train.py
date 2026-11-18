_base_ = ['_base_/dmm_dcfl_mbo_fast_loop_dronevehicle.py']
import os
pretrained_vmamba = os.path.join(
    os.environ.get('DMM_BASE_ROOT', '{{fileDirname}}/../../../code'),
    'weights/vssm/vssm_tiny_0230_ckpt_epoch_262.pth')
model = dict(
    backbone_vi=dict(pretrained=pretrained_vmamba),
    backbone_ir=dict(pretrained=pretrained_vmamba))
