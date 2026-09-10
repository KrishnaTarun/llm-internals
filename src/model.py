import torch
from torch import nn

from model_components import *
from config import Config, load_config



class Seq2SeqModel(nn.Module):
    """
    A sequence-to-sequence model that combines an encoder and a decoder.
    """

    def __init__(self, 
                 n_layers_enc:int,
                 n_layers_dec:int,
                 d_model:int,
                 dff:int,
                 num_heads:int,
                 dropout:float,
                 src_vocab_size:int,
                 tgt_vocab_size:int,
                 seq_len:int
    ):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = TransformerEncoderBlock(num_layers=n_layers_enc,
                                            dmodel=d_model,
                                            dff=dff,
                                            num_heads=num_heads,
                                            dropout=dropout)
        
        self.decoder = TransformerDecoderBlock(num_layers=n_layers_dec,
                                            dmodel=d_model,
                                            dff=dff,
                                            num_heads=num_heads,
                                            dropout=dropout)

        self.src_emb = Embedding(vocab_size=src_vocab_size, dmodel=d_model)
        self.tgt_emb = Embedding(vocab_size=tgt_vocab_size, dmodel=d_model)

        self.enc_pos_encoding = SinCosinePositionalEncoding(
                                                                d_model=d_model,
                                                                dropout=dropout,
                                                                max_len=seq_len
                                                            )
        self.dec_pos_encoding = SinCosinePositionalEncoding(    d_model=d_model,
                                                                dropout=dropout,
                                                                max_len=seq_len
                                                            )
        self.projection_layer = ProjectionLayer(d_model=d_model,
                                                vocab_size=tgt_vocab_size)

        self.init_parameters()

    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    #     self.src_pad_idx = config.src_pad_idx
    #     self.tgt_pad_idx = config.tgt_pad_idx

    # def create_padding_mask(self, seq, pad_idx):
    #     # Create a mask for padding tokens
    #     return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

    def forward(self, src, tgt, causal_mask=None, padding_mask=None):

        enc_in = self.enc_pos_encoding(self.src_emb(src))
        enc_out = self.encoder(enc_in, padding_mask)

        dec_in = self.dec_pos_encoding(self.tgt_emb(tgt))
        dec_out = self.decoder(dec_in, enc_out, causal_mask, padding_mask)

        return self.projection_layer(dec_out)
