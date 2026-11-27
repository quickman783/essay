"""CIF from CMFADet (Yu and Shin, TGRS 2026) adapted to the DMM FPN."""

import torch
from torch import nn

from mmrotate.registry import MODELS


class CIFStage(nn.Module):
    """Exchange channel-gated residuals between registered RGB and IR maps."""

    def __init__(self, channels, reduction):
        super().__init__()
        self.channels = channels
        paired_channels = 2 * channels
        hidden = max(1, paired_channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.Linear(paired_channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, paired_channels, bias=False),
            nn.Sigmoid())

    def forward(self, visible, infrared):
        if (visible.ndim != 4 or visible.shape != infrared.shape
                or visible.shape[1] != self.channels):
            raise ValueError(
                f'CIF expects matching NCHW feature maps with {self.channels} '
                f'channels, received {visible.shape} and {infrared.shape}')
        paired = torch.cat((visible, infrared), dim=1)
        weights = self.channel_gate(paired.mean(dim=(-2, -1)))
        visible_weight, infrared_weight = weights.split(self.channels, dim=1)
        visible_enhanced = visible + infrared * infrared_weight[:, :, None, None]
        infrared_enhanced = infrared + visible * visible_weight[:, :, None, None]
        # DMM's FPN consumes one feature map of the original width per stage.
        return visible_enhanced + infrared_enhanced


@MODELS.register_module()
class CIFFusion(nn.Module):
    """Replace all per-scale DCFM blocks through DMM's fusion interface."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        if not in_channels or any(channels < 1 for channels in in_channels):
            raise ValueError('CIF requires positive channel counts for each stage')
        if reduction < 1:
            raise ValueError('CIF reduction must be positive')
        self.stages = nn.ModuleList(
            CIFStage(channels, reduction) for channels in in_channels)

    def forward(self, x_vi, x_ir, return_aux=False):
        if len(x_vi) != len(self.stages) or len(x_ir) != len(self.stages):
            raise ValueError('RGB and IR pyramids must match the configured CIF stages')
        features = tuple(stage(visible, infrared) for stage, visible, infrared
                         in zip(self.stages, x_vi, x_ir))
        return (features, None) if return_aux else features
