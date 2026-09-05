import torch
from torch import nn

from model_components import *
from config import Config, load_config



class Seq2SeqModel(nn.Module):
    """
    A sequence-to-sequence model that combines an encoder and a decoder.
    """

    def __init__(self, config: Config):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = TransformerEncoderBlock(num_layers=config.model.enc_num_layers,
                                            dmodel=config.model.d_model,
                                            dff=config.model.dff,
                                            num_heads=config.model.num_heads,
                                            dropout=config.model.dropout)
        
        self.decoder = TransformerDecoderBlock(num_layers=config.model.dec_num_layers,
                                            dmodel=config.model.d_model,
                                            dff=config.model.dff,
                                            num_heads=config.model.num_heads,
                                            dropout=config.model.dropout)

        self.src_emb = Embedding(vocab_size=config.data.src_vocab_size, dmodel=config.model.d_model)
        self.tgt_emb = Embedding(vocab_size=config.data.tgt_vocab_size, dmodel=config.model.d_model)

        self.enc_pos_encoding = SinCosinePositionalEncoding(
                                                                d_model=config.model.d_model,
                                                                dropout=config.model.dropout,
                                                                max_len=config.data.seq_len
                                                            )
        self.dec_pos_encoding = SinCosinePositionalEncoding(    d_model=config.model.d_model,
                                                                dropout=config.model.dropout,
                                                                max_len=config.data.seq_len
                                                            )
        self.projection_layer = ProjectionLayer(d_model=config.model.d_model,
                                                vocab_size=config.data.tgt_vocab_size)

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


if __name__ == "__main__":
    # Example usage
    config = load_config("configs/seq2seq_training.yaml")
    print(config)
    
    model = Seq2SeqModel(config)

    # # Example input sequences (batch_size, seq_len)
    src = torch.randint(0, config.data.src_vocab_size, (32, 100))
    tgt = torch.randint(0, config.data.tgt_vocab_size, (32, 100))

    output = model(src, tgt)
    print(output.shape)  # Expected output shape: (batch_size, seq_len, tgt_vocab_size)