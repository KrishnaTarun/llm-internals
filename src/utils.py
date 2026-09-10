from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import TranslationDataset
from config import Config
from model import *


def get_dataset(config: Config):
    """Build the configured translation dataset."""

    dataset = TranslationDataset(dataset=load_dataset("Helsinki-NLP/opus_books", "en-nl", split="train"),
                                 seq_len=config.data.seq_len)

    train, val = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train, shuffle=True, batch_size =config.training.batch_size)
    val_loader = DataLoader(train, shuffle=True, batch_size =config.training.batch_size)
    return dataset, train_loader, val_loader





def get_model(config: Config, src_vocab: int, tgt_vocab: int):
    if config.model.model_type == "Seq2Seq":
        return Seq2SeqModel(
            n_layers_enc=config.model.enc_num_layers,
            n_layers_dec=config.model.dec_num_layers,
            d_model=config.model.d_model,
            dff=config.model.dff,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            src_vocab_size=src_vocab,
            tgt_vocab_size=tgt_vocab,
            seq_len=config.data.seq_len,
        )

    raise ValueError(f"Unsupported model type: {config.model.model_type}")


