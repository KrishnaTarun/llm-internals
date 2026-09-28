# this contains sub-modules that are reusbale across different model architectures
import math

import torch
import torch.nn.functional as F
from torch import nn


class Embedding(nn.Module):
    """Embed token IDs and scale them by the model dimension."""

    def __init__(self, vocab_size: int, dmodel: int) -> None:
        """Initialize an embedding table."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dmodel)
        self.dmodel = dmodel

    def forward(self, x):
        """Return scaled embeddings for token IDs."""
        return self.embedding(x) * math.sqrt(self.dmodel)


class FeedForward(nn.Module):
    """Apply the position-wise feed-forward transformation."""

    def __init__(self, dmodel: torch.Tensor, dff: int, dropout: float = 0.1) -> None:
        """Initialize the two-layer feed-forward network."""
        super().__init__()

        self.W1 = nn.Linear(dmodel, dff, bias=False)
        self.W2 = nn.Linear(dff, dmodel, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply linear, ReLU, and linear transformations."""
        x = self.W1(x)
        x = F.relu(x)
        x = self.W2(x)
        return x


class ResidualConnection(nn.Module):
    """Apply dropout, residual addition, and layer normalization."""

    def __init__(self, dmodel: int, dropout: float = 0.1, model_type: str = "Seq2Seq") -> None:
        """Initialize dropout and layer normalization."""
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.Identity()
        
        if model_type=="Seq2Seq":
            self.layer_norm = nn.LayerNorm(dmodel)

    def forward(self, x, sublayer_output):
        """Combine an input with a sublayer output."""
        return self.layer_norm(x + self.dropout(sublayer_output))


class ProjectionLayer(nn.Module):
    """Project decoder outputs into target vocabulary logits."""

    def __init__(self, d_model, vocab_size) -> None:
        """Initialize the vocabulary projection layer."""
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x) -> torch.Tensor:
        """Return vocabulary logits for decoder representations."""
        return self.proj(x)
