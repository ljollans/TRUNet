# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import torch.nn as nn
from torch.nn import LayerNorm

from TRUNet_network.model.block import Block


# from An image is worth 16x16 words:
# The Transformer encoder (Vaswani et al., 2017) consists of alternating
# layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3).


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        # make the layers within the transformer
        for _ in range(config.transformer_num_layers):
            # block contains multi-head attention and mlp
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
