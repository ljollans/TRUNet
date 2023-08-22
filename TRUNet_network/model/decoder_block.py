# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from TRUNet_network.model.ConvReLU import Conv3dReLU


class DecoderBlock3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        # convolve 1
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # convolve 2
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # upsample
        self.up = F.interpolate

    def forward(self, x, skip=None):
        x = self.up(input=x, scale_factor=2, mode='trilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x.float())
        x = self.conv2(x)
        return x

