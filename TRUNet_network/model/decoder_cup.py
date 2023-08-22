# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn

from TRUNet_network.model.ConvReLU import Conv3dReLU
from TRUNet_network.model.decoder_block import DecoderBlock3d


class DecoderCup3d(nn.Module):  # cascade upsampler
    def __init__(self, config):
        super().__init__()
        self.config = config
        if len(config.decoder_channels) == 4:
            head_channels = 512
        elif len(config.decoder_channels) == 3:
            head_channels = 256

        # define a model that takes the hidden input size and applies a convolution with output size = head_channels
        # (D, H/16, W/16)
        self.conv_more = Conv3dReLU(
            config.hidden_size,  # these are the inchannels
            head_channels,  # these are the outchannels
            kernel_size=3,
            padding=1,
            stride=1,
            use_batchnorm=True,
        )

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            if len(config.skip_channels) == 4:
                for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                    skip_channels[3 - i] = 0
            elif len(config.skip_channels) == 3:
                for i in range(3 - self.config.n_skip):  # re-select the skip channels according to n_skip
                    skip_channels[2 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])

        out_channels = decoder_channels

        blocks = [
            DecoderBlock3d(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, skip_channels)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):

        B, n_patch, hidden = hidden_states.size()
        # reshape from (B, n_patch, hidden) to (B, h, w, l, hidden)

        h, w, l = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(np.cbrt(n_patch))
        # h,w,l are the number of rows/columns/... in the grid of patches. 
        # i.e. for a 224 input with 16x16 patches there are 14x14
        x = hidden_states.permute(0, 2, 1)
        # rearranges the original tensor according to the desired ordering and returns a new multidimensional rotated
        # tensor
        x = x.contiguous().view(B, hidden, h, w, l)  # [24, 768, 14, 14, 14]
        # .contiguous Returns a contiguous in memory tensor containing the same data as self tensor
        # reshapes
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None

            x = decoder_block(x, skip=skip)

        return x
