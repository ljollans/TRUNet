# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Dropout, Conv3d
from torch.nn.modules.utils import _triple

from TRUNet_network.model.resnetv2_3d import ResNetV23d
from TRUNet_network.model.encode_3d import EncoderCup3d


class Embeddings3d(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings3d, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _triple(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (
            img_size[0] // config.patches.size // grid_size[0], img_size[1] // config.patches.size // grid_size[1],
            img_size[2] // config.patches.size // grid_size[2])
            patch_size_real = (patch_size[0] * config.patches.size, patch_size[1] * config.patches.size,
                               patch_size[2] * config.patches.size)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) * (
                        img_size[2] // patch_size_real[2])
            self.hybrid = True
        else:
            patch_size = _triple(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV23d(block_units=config.resnet.num_layers,
                                           width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        else:
            self.convolutions = EncoderCup3d(config)
            if len(config.decoder_channels) == 4:
                in_channels = config.decoder_channels[-4]
            elif len(config.decoder_channels) == 3:
                in_channels = config.decoder_channels[-3]

        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # self.position_embeddings = nn.Parameter(torch.reshape(torch.arange(0,n_patches * config.hidden_size), (1, n_patches, config.hidden_size)).float())

        self.dropout = Dropout(config.transformer_dropout_rate)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            x, features = self.convolutions(x)

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # add position to the patches
        z = self.position_embeddings
        embeddings = x + z
        embeddings = self.dropout(embeddings)
        return embeddings, features
