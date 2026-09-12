import torch
from torch import nn

from config import Config, load_config
from model import Seq2SeqModel


# TODO: GPU check and device assignment
def train(config: Config):
    """
    Train the model based on the provided configuration.

    Args:
        config (Config): Configuration object containing model, training, and dataset parameters.
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.training.seed)

    # Initialize the model
    # FIXME: Add support for other model types in the future
    if config.model.model_type == "Seq2Seq":
        model = Seq2SeqModel(config).to(config.training.device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=config.data.tgt_pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Placeholder for data loading (to be implemented)
    # train_loader = ...


if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config("configs/seq2seq_training.yaml")
    print(config)

    # Start training
    train(config)
