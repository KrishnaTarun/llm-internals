"""Run text generation with a trained model and tokenizer."""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import Config, load_config
from sampling import Sampling
from utils import get_model


class InferenceRunner:
    """Thin inference wrapper mirroring the project's trainer structure."""

    def __init__(self, config: Config, tokenizer: Tokenizer, model: torch.nn.Module, device: torch.device):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device

    def generate_sample(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int | None = None,
    ):
        """Sample tokens from ``self.model`` while preserving the prompt in the returned sequence."""
        sampling_method = getattr(Sampling, self.config.inference.sampling_strategy, None)
        if not callable(sampling_method):
            raise ValueError(
                f"Unknown sampling type: {self.config.inference.sampling_strategy}. "
                "Must be one of: greedy, probabilistic, top_k, temperature."
            )

        for _ in range(self.config.inference.max_new_tokens):
            model_input = input_ids[:, -self.config.data.seq_len :]
            logits = self.model(model_input)

            next_logit = logits[:, -1, :]
            if self.config.inference.sampling_strategy == "top_k":
                next_id = sampling_method(next_logit, k=self.config.inference.top_k)
            elif self.config.inference.sampling_strategy == "temperature":
                next_id = sampling_method(next_logit, temperature=self.config.inference.temperature)
            else:
                next_id = sampling_method(next_logit)

            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos_token_id is not None and torch.all(next_id == eos_token_id):
                break

        return input_ids

    def generate_text(self, prompt: str) -> str:
        """Generate and decode a continuation of ``prompt`` using the existing logic."""
        input_ids = torch.tensor(self.tokenizer.encode(prompt).ids, device=self.device).unsqueeze(0)
        self.model.eval()

        with torch.inference_mode():
            output_ids = self.generate_sample(
                input_ids,
                eos_token_id=None,
            ).squeeze(0).cpu().tolist()

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load a checkpoint into the underlying model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


def infer(config: Config):
    """Build the model architecture and generate text from a checkpoint."""
    torch.manual_seed(config.training.seed)

    project_root = Path(__file__).resolve().parent.parent
    artifact_dir = project_root / "dataset_artifacts" / "gpt"
    tokenizer_path = artifact_dir / "gpt.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    config.data.vocab_size = tokenizer.get_vocab_size()
    model = get_model(config)

    if config.model.resume_from is None:
        raise ValueError("resume_from must be specified in the config for inference.")

    resume_path = Path(config.model.resume_from)
    if not resume_path.is_absolute():
        resume_path = project_root / resume_path

    device = torch.device(config.training.device)
    runner = InferenceRunner(config, tokenizer, model, device)
    runner.load_checkpoint(resume_path)

    output = runner.generate_text("what is 1+1?")
    print(output)


def main() -> None:
    """Parse CLI arguments and launch inference from the selected config."""
    parser = argparse.ArgumentParser(description="Generate text from a configured model.")
    parser.add_argument(
        "-c",
        "--config",
        default="configs/gptstyle_eval.yaml",
        help="Path to the YAML configuration file (default: %(default)s).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)
    infer(config)


if __name__ == "__main__":
    main()


