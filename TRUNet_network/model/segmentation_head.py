# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class SegmentationHead3d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear3d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv3d, upsampling)
