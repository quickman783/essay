"""Build fusion-only inference model for an exported MBO checkpoint."""
_base_ = ['_base_/dmm_dcfl_mbo_loop_dronevehicle.py']
model = dict(mbo_auxiliary=False)
