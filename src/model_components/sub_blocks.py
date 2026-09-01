# this contains sub-modules that are reusbale across different model architectures
import torch
from torch import nn
import torch.nn.functional as F


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
    
class ResidualConnection(nn.Module):
    def __init__(self, dmodel:int, dropout:float=0.1) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dmodel)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))

