from torch import nn

from model_components import (
    Embedding,
    ProjectionLayer,
    SinCosinePositionalEncoding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class Seq2SeqModel(nn.Module):
    """A sequence-to-sequence model that combines an encoder and a decoder."""

    def __init__(
        self,
        n_layers_enc: int,
        n_layers_dec: int,
        d_model: int,
        dff: int,
        num_heads: int,
        dropout: float,
        src_vocab_size: int,
        tgt_vocab_size: int,
        seq_len: int,
    ):
        """Initialize encoder, decoder, embeddings, and output projection."""
        super(Seq2SeqModel, self).__init__()

        self.encoder = TransformerEncoderBlock(
            num_layers=n_layers_enc,
            dmodel=d_model,
            dff=dff,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.decoder = TransformerDecoderBlock(
            num_layers=n_layers_dec,
            dmodel=d_model,
            dff=dff,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.src_emb = Embedding(vocab_size=src_vocab_size, dmodel=d_model)
        self.tgt_emb = Embedding(vocab_size=tgt_vocab_size, dmodel=d_model)

        self.enc_pos_encoding = SinCosinePositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len)
        self.dec_pos_encoding = SinCosinePositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len)
        self.projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

        self.init_parameters()

    def init_parameters(self):
        """Initialize trainable weights for the model's supported layer types."""
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

    def encoderblock(self, src, padding_mask):
        """Encode source token IDs with positional embeddings and attention."""
        enc_in = self.enc_pos_encoding(self.src_emb(src))
        enc_out = self.encoder(enc_in, padding_mask)

        return enc_out

    def decoderblock(self, enc_out, tgt, causal_mask, padding_mask):
        """Decode target token IDs using encoder output and attention masks."""
        dec_in = self.dec_pos_encoding(self.tgt_emb(tgt))
        dec_out = self.decoder(dec_in, enc_out, causal_mask, padding_mask)

        return dec_out

    def forward(self, src, tgt, causal_mask=None, padding_mask=None):
        """Run the source and target tensors through the sequence-to-sequence model."""
        enc_out = self.encoderblock(src, padding_mask)
        dec_out = self.decoderblock(enc_out, tgt, causal_mask, padding_mask)

        return self.projection_layer(dec_out)
