from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
# coding=utf-8

import torch.nn as nn

from TRUNet_network.model.ConvReLU import Conv3dReLU


class EncoderBlock3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        # convolve 1
        self.conv1 = Conv3dReLU(
            in_channels,
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
        # downsample
        self.down = F.interpolate

    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EncoderCup3d(nn.Module):  # cascade downsampler
    def __init__(self, config):
        super().__init__()
        self.config = config

        if len(config.decoder_channels) == 4:
            out_channels = [config.decoder_channels[-2], config.decoder_channels[-3], config.decoder_channels[-4]]
            in_channels = [3, config.decoder_channels[-2], config.decoder_channels[-3]]
        elif len(config.decoder_channels) == 3:
            out_channels = [config.decoder_channels[-2], config.decoder_channels[-3]]
            in_channels = [3, config.decoder_channels[-2]]

        blocks = [
            EncoderBlock3d(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, config.skip_channels)
        ]

        self.blocks = nn.ModuleList(blocks)
        self.down = F.interpolate

    def forward(self, x):
        features_tmp = []
        ctr = 4
        for i, encoder_block in enumerate(self.blocks):
            x = encoder_block(x)
            features_tmp.append(self.down(input=x, scale_factor=1 / ctr, mode='trilinear', align_corners=True))
            ctr *= 2
        features = []
        for i in range(len(features_tmp)):
            features.append(features_tmp[-(i + 1)])

        return x, features
