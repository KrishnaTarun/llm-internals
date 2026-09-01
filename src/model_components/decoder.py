import torch
from torch import nn
import torch.nn.functional as F

from model_components.attention import Attention
from model_components.sub_blocks import FeedForward, ResidualConnection


class DecoderLayer(nn.Module):
    def __init__(self, dmodel:int, dff:int, num_heads:int, dropout:float=0.1) -> None:
        super().__init__()
        # masked-multihead attention
        # initialized similarly but in forward will have mask
        # (lower-triangular matrix)
        self.mmha = Attention(dmodel, num_heads, dropout)
        # And this is cross-attention as (query, key) coming
        # from encoder layer outputs
        self.mha = Attention(dmodel, num_heads, dropout)
        self.ffn = FeedForward(dmodel, dff, dropout)

        self.rc1 = ResidualConnection(dmodel, dropout)
        self.rc2 = ResidualConnection(dmodel, dropout)
        self.rc3 = ResidualConnection(dmodel, dropout)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # Masked multi-head attention (self-attention)
        mha = self.mmha(x, x, x, look_ahead_mask)
        x = self.rc1(x, mha)

        # Multi-head attention (encoder-decoder attention)
        mha = self.mha(x, enc_output, enc_output, padding_mask)
        x = self.rc2(x, mha)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.rc3(x, ffn_output)

        return x