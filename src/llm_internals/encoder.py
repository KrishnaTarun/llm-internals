

import torch
from torch import nn
import torch.nn.functional as F

from src.llm_internals.attention import Attention



class FeedForward(nn.Module):
    def __init__(self, dmodel:torch.Tensor, dff:int, dropout:float=0.1) -> None:
        super().__init__()
    
        self.W1 = nn.Linear(dmodel, dff, bias=True)
        self.W2 = nn.Linear(dff, dmodel, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.W1(x)
        x = F.relu(x)
        x = self.W2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dmodel:torch.Tensor, dff:int, num_heads:int, dropout:float=0.1) -> None:
        super().__init__()
        # dff: int: The dimensionality of the feed-forward network's hidden layer.
        # This is typically larger than dmodel to allow for more complex transformations.

        self.mha = Attention(dmodel, num_heads, dropout)
        self.ffn = FeedForward(dmodel, dff, dropout)

    def forward(self, x, mask=None):
        # Self-attention
        mha = self.mha(x, x, x, mask)
        

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_layers:int,
                       dmodel:torch.Tensor, 
                       dff:int, 
                       num_heads:int, 
                       dropout:float=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dmodel, dff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

