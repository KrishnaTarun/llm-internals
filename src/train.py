import argparse

import torch

from config import Config, load_config
from trainer import LLMTrainer
from utils import get_dataset, get_model


# TODO: GPU check and device assignment
def train(config: Config):
    """Build and train the configured model architecture."""
    # Set random seed for reproducibility
    # FIXME this properly
    torch.manual_seed(config.training.seed)

    # get data
    # FIXME
    # no need of test loader that should be isolated
    # in standalone script like infernece.py
    train_loader, val_loader = get_dataset(config)

    # Initialize the model
    # FIXME: Add support for other model types in the future
    model = get_model(config)

    trainer = LLMTrainer(model, config)

    if config.training.resume_from is not None:
        trainer.load_checkpoint(config.training.resume_from)

    trainer.fit(train_loader, val_loader, start_epoch=trainer.current_epoch)


def main() -> None:
    """Parse CLI arguments and launch training from the selected config."""
    parser = argparse.ArgumentParser(description="Train a configured language model.")
    parser.add_argument(
        "-c",
        "--config",
        default="configs/gptstyle_training.yaml",
        help="Path to the YAML configuration file (default: %(default)s).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)

    train(config)


if __name__ == "__main__":
    main()
