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
        
        #Causal Attention
        self.mmha = Attention(dmodel, num_heads, dropout)
        # And this is cross-attention as (query, key) coming
        # from encoder layer outputs
        self.mha = Attention(dmodel, num_heads, dropout)
        self.ffn = FeedForward(dmodel, dff, dropout)

        self.rc1 = ResidualConnection(dmodel, dropout)
        self.rc2 = ResidualConnection(dmodel, dropout)
        self.rc3 = ResidualConnection(dmodel, dropout)

    def forward(self, x, enc_output, causal_mask=None, padding_mask=None):
        # Masked multi-head attention (self-attention)
        mha = self.mmha(x, x, x, causal_mask)
        x = self.rc1(x, mha)

        # Multi-head attention (encoder-decoder attention)
        mha = self.mha(x, enc_output, enc_output, padding_mask)
        x = self.rc2(x, mha)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.rc3(x, ffn_output)

        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_layers:int,
                       dmodel:int, 
                       dff:int, 
                       num_heads:int, 
                       dropout:float=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dmodel, dff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_output, causal_mask=None, padding_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, causal_mask, padding_mask)
        return x

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_length = 5
    dmodel = 16
    dff = 64
    num_heads = 4
    src_pad_idx = 0
    tgt_pad_ids = 0

    src = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    tgt = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])

    padding_mask = (src != src_pad_idx).int().unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_length)
    look_ahead_mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)
    padding_mask_tgt = (tgt != tgt_pad_ids).int().unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_length)

    print("Padding mask shape:", padding_mask)
    print("Look-ahead mask shape:", look_ahead_mask)
    # print("Target padding mask shape:", padding_mask_tgt.shape)
    causal_mask = look_ahead_mask * padding_mask_tgt  # Combine look-ahead and padding masks
    print("Causal mask shape:", causal_mask.shape)
    print("Causal mask:", causal_mask)

    x = torch.rand(batch_size, seq_length, dmodel)
    enc_output = torch.rand(batch_size, seq_length, dmodel)
    

    decoder_layer = DecoderLayer(dmodel, dff, num_heads)
    output = decoder_layer(x, enc_output, causal_mask, padding_mask)
