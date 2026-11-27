"""DMM on DroneVehicle with CIF at all four backbone scales."""

_base_ = ['../../../code/configs/dmm/dmm_s2anet_1x_dronevehicle.py']

custom_imports = dict(imports=['mmrotate.hooks.progress_bar', 'dmm_cif'], allow_failed_imports=False)
model = dict(fusblock=dict(
    _delete_=True, type='CIFFusion', in_channels=[96, 192, 384, 768],
    reduction=16))
work_dir = './work_dirs/dmm_cif_1x_dronevehicle'
