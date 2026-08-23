
from torch import nn

class AttentionMechanism(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, query, key, value):
        pass
