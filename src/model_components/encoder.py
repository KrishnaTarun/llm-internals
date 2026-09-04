

import torch
from torch import nn

from model_components.attention import Attention
from model_components.sub_blocks import FeedForward, ResidualConnection


class EncoderLayer(nn.Module):
    def __init__(self, dmodel:int, dff:int, num_heads:int, dropout:float=0.1) -> None:
        super().__init__()
        # dff: int: The dimensionality of the feed-forward network's hidden layer.
        # This is typically larger than dmodel to allow for more complex transformations.

        self.mha = Attention(dmodel, num_heads, dropout)
        self.ffn = FeedForward(dmodel, dff, dropout)

        # each encoder block consit of 2 residual connectiions
        self.rc1 = ResidualConnection(dmodel, dropout)
        self.rc2 = ResidualConnection(dmodel, dropout) 


    def forward(self, x, mask=None):

        mha = self.mha(x, x, x, mask)
        x = self.rc1(x, mha)

        ffn = self.ffn(x)
        x = self.rc2(x, ffn)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_layers:int,
                       dmodel:int, 
                       dff:int, 
                       num_heads:int, 
                       dropout:float=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dmodel, dff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x

if __name__ == "__main__":

    # Example usage
    batch_size = 2
    seq_length = 5
    dmodel = 8
    dff = 32
    num_heads = 4
    num_layers = 1

    src_pad_idx = 0
        # tgt_pad_ids = 0
    
    src = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]) # (batch, seq_length)
    # tgt = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    
    padding_mask = (src != src_pad_idx).int().unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_length)
    print(padding_mask.shape)
    print(padding_mask)

    x = torch.rand(batch_size, seq_length, dmodel)
    mask = padding_mask

    encoder_block = TransformerEncoderBlock(num_layers, dmodel, dff, num_heads)
    output = encoder_block(x, mask)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)