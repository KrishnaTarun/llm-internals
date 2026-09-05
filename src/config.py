from dataclasses import dataclass, field

import torch
import dacite
import yaml

@dataclass
class ModelConfig:

    d_model: int = 128
    dff: int = 512
    num_heads: int = 8
    enc_num_layers: int = 2
    dec_num_layers: int = 1
    dropout: float = 0.1

@dataclass
class DatasetConfig:

    seq_len: int = 100
    src_vocab_size: int = 100
    tgt_vocab_size: int = 100
    src_pad_idx: int = 0
    tgt_pad_idx: int = 0

@dataclass
class TrainingConfig:

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    warmup_steps: int = 4000
    seed: int = 42

@dataclass
class Config:
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DatasetConfig = field(default_factory=DatasetConfig)

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return dacite.from_dict(Config, config_data)
