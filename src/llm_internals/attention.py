from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


dataclass(frozen=True)
class AttentionConfig:
    model_dim: int
    num_heads: int
    num_kv_heads: int
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1]")

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def _scaled_dot_product(self, query, key, value, mask=None):

        dim_k = query.size()[-1]

        #matmul (also referred to as to as logits or attention scores)
        scores = torch.matmul(query, key.transpose(-2, -1))

        #scale (scaled dot product)
        scores = scores / math.sqrt(dim_k)

        #TODO: excluding mask for time being
        #softmax
        attention_weights = F.softmax(scores, dim=-1)

        #scaled dot product
        return torch.matmul(attention_weights, value)


    def forward(self, query, key, value):
        pass

        

class MultiHeadAttention(Attention):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            AttentionConfig(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                dropout=dropout,
            )
        )


class MultiQueryAttention(Attention):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            AttentionConfig(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=1,
                dropout=dropout,
            )
        )


class GroupedQueryAttention(Attention):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            AttentionConfig(
                model_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
            )
        )
    