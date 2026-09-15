import torch

from config import Config, load_config
from trainer import LLMTrainer
from utils import get_dataset, get_model


# TODO: GPU check and device assignment
def train(config: Config):
    """Train the model based on the provided configuration.

    Args:
        config (Config): Configuration containing model, training, and data parameters.
    """
    # Set random seed for reproducibility
    # FIXME this properly
    torch.manual_seed(config.training.seed)

    # get data
    # FIXME
    dataset, train_loader, val_loader = get_dataset(config)
    src_vocab_size = dataset.src_vocab_size
    tgt_vocab_size = dataset.tgt_vocab_size

    # Initialize the model
    # FIXME: Add support for other model types in the future
    model = get_model(config, src_vocab=src_vocab_size, tgt_vocab=tgt_vocab_size)

    trainer = LLMTrainer(model, config, dataset)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config("configs/seq2seq_training.yaml")
    print(config)

    # Start training
    train(config)
