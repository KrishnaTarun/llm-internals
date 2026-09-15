from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, random_split

from config import Config
from dataset import TranslationDataset
from model import Seq2SeqModel


def get_dataset(config: Config):
    """Build the configured translation dataset."""
    dataset = TranslationDataset(
        dataset=load_dataset("Helsinki-NLP/opus_books", "en-nl", split="train"),
        seq_len=config.data.seq_len,
    )

    train, val = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train, shuffle=True, batch_size=config.training.batch_size)
    val_loader = DataLoader(val, shuffle=False, batch_size=config.training.batch_size)
    return dataset, train_loader, val_loader


def get_model(config: Config, src_vocab: int, tgt_vocab: int):
    """Build the configured model using the supplied vocabulary sizes."""
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


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, checkpoint_dir: str | Path):
    """Persist model and optimizer state to a single latest checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str | Path, device: str):
    """Restore model and optimizer state from a saved checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(checkpoint.get("epoch", 0))
    print(f"Loaded checkpoint from: {checkpoint_path} (epoch {epoch})")
    return epoch
