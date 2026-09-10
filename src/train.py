import torch
from torch import nn

from config import Config, load_config
from model import Seq2SeqModel
from utils import get_model, get_dataset


#TODO: GPU check and device assignment
def train(config:Config):
    """
    Train the model based on the provided configuration.

    Args:
        config (Config): Configuration object containing model, training, and dataset parameters.
    """
    # Set random seed for reproducibility
    # FIXME this properly
    torch.manual_seed(config.training.seed)

    #get data
    #FIXME
    dataset, train_loader, val_loader = get_dataset(config)
    src_vocab_size = dataset.src_vocab_size
    tgt_vocab_size = dataset.tgt_vocab_size

    # Initialize the model
    #FIXME: Add support for other model types in the future
    model = get_model(config, src_vocab=src_vocab_size, tgt_vocab= tgt_vocab_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=config.data.tgt_pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    # Placeholder for data loading (to be implemented)
    # train_loader = ...

    

if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config("configs/seq2seq_training.yaml")
    print(config)

    # Start training
    train(config)