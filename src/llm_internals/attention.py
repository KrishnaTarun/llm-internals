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
    def __init__(self, dmodel:torch.Tensor, num_heads:int, dropout:float=0.1) -> None:
        super().__init__()

        #TODO: add dropout and masking functionality

        #this is embedding dimension
        self.emb_dim = dmodel
        self.heads = num_heads
        self.head_dim = self.emb_dim // self.heads

        assert self.emb_dim % self.heads == 0, "Embedding dimension must be divisible by number of heads"

        #projection heads, W(q, i), W(k, i), W(v, i) for each head i.
        """
        To avoid confusion W_q, w_k, W_v could have been also intialized with 
        nn.Linear(emb_dim, emd_dim), emb_dim = head_dim * heads, but we are intializing 
        them with nn.Linear(emb_dim, head_dim * heads).
        """

        self.W_Q = nn.Linear(self.emb_dim, self.head_dim * self.heads, bias=False) 
        self.W_K = nn.Linear(self.emb_dim, self.head_dim * self.heads, bias=False)
        self.W_V = nn.Linear(self.emb_dim, self.head_dim * self.heads, bias=False)
        
        self.W_O = nn.Linear(self.head_dim * self.heads, self.emb_dim, bias=False)

        # Ignore for the moment
        self.dropout = nn.Dropout(dropout)


    def _scaled_dot_product(self, query, key, value, mask=None):

        #dimesion of projection head
        dim_k = query.size()[-1]

        #matmul (also referred to as to as logits or attention scores)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # normalization
        scores = scores / math.sqrt(dim_k)


        # Causal masking: mask out future tokens in the sequence.
        # This is done by setting the scores of the future tokens to -inf,
        # so that after applying softmax, their probabilities become 0.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        
        #scaled dot product
        return torch.matmul(attention_weights, value)


    def forward(self, query, key, value):


        q = self.W_Q(query) # (b, seq_len, emb_dim)
        k = self.W_K(key) # (b, seq_len, emb_dim)
        v = self.W_V(value) # (b, seq_len, emb_dim)

        # prepare q, k, v for multi-head attention
        q = q.view(q.size(0), q.size(1), self.heads, self.head_dim).transpose(1, 2)  # (b, heads, seq_len, head_dim)
        k = k.view(k.size(0), k.size(1), self.heads, self.head_dim).transpose(1, 2)  # (b, heads, seq_len, head_dim)
        v = v.view(v.size(0), v.size(1), self.heads, self.head_dim).transpose(1, 2)  # (b, heads, seq_len, head_dim)

        # reshape it to (b, seq_len, heads * head_dim)
        at_heads = self._scaled_dot_product(q, k, v)  # (b, heads, seq_len, head_dim)
        at_heads = at_heads.transpose(1, 2).contiguous() # (b, seq_len, heads, head_dim)
        at_heads = at_heads.view(at_heads.size(0), -1, self.emb_dim)  # (b, seq_len, heads * head_dim)

        return self.W_O(at_heads)  # (b, seq_len, emb_dim)



mod = Attention(dmodel=8, num_heads=2, dropout=0.1)
out = mod(torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8))  
print(out.shape)  

# class MultiHeadAttention(Attention):
#     def __init__(
#         self,
#         model_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#     ) -> None:
#         super().__init__(
#             AttentionConfig(
#                 model_dim=model_dim,
#                 num_heads=num_heads,
#                 num_kv_heads=num_heads,
#                 dropout=dropout,
#             )
#         )


# class MultiQueryAttention(Attention):
#     def __init__(
#         self,
#         model_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#     ) -> None:
#         super().__init__(
#             AttentionConfig(
#                 model_dim=model_dim,
#                 num_heads=num_heads,
#                 num_kv_heads=1,
#                 dropout=dropout,
#             )
#         )


# class GroupedQueryAttention(Attention):
#     def __init__(
#         self,
#         model_dim: int,
#         num_heads: int,
#         num_kv_heads: int,
#         dropout: float = 0.0,
#     ) -> None:
#         super().__init__(
#             AttentionConfig(
#                 model_dim=model_dim,
#                 num_heads=num_heads,
#                 num_kv_heads=num_kv_heads,
#                 dropout=dropout,
#             )
#         )
    