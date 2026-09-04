
import torch
from torch import nn

class SinCosinePositionalEncoding(nn.Module):
    """
    The one introduced in Attention is All You Need (Vaswani et al., 2017) paper.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(SinCosinePositionalEncoding, self).__init__()

        
        self.dropout = nn.Dropout(p=dropout)
        pos_en = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float)


        #this is used for computation efficiency
        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        #even
        pos_en[:, 0::2] = torch.sin(pos.unsqueeze(1) * denom)
        #odd
        pos_en[:, 1::2] = torch.cos(pos.unsqueeze(1) * denom)

        # to add batch  dimension in-order to amke it broadcastable with the input embedding 
        # tensor of shape (batch_size, seq_len, d_model) 
        pos_en = pos_en.unsqueeze(0)  # shape (1, max_len, d_model)
        
    
        self.register_buffer('pos_en', pos_en)

    def forward(self, x):

        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encodings added
        """
        seq_len = x.size(1)
        x = x + (self.pos_en[:, :seq_len :]).require_grad_(False)
        return self.dropout(x)


if __name__ == "__main__":
    # Example usage
    d_model = 16
    dropout = 0.1
    max_len = 100

    pos_encoding = SinCosinePositionalEncoding(d_model, dropout, max_len)

    # # Create a dummy input tensor of shape (batch_size, seq_len, d_model)
    # batch_size = 2
    # seq_len = 10
    # x = torch.zeros(batch_size, seq_len, d_model)

    # # Get the positional encodings for the input tensor
    # output = pos_encoding(x)

    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)