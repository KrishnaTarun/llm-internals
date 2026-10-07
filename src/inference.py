"""Run text generation with a trained model and tokenizer."""
import argparse
from pathlib import Path

import config 
import torch
from tokenizers import Tokenizer

from config import Config, load_config
from trainer import LLMTrainer
from transformers.trainer_jit_checkpoint import CheckpointManager
from utils import  get_model

from sampling import Sampling

#FIXME: incorporate tempeature and top_k

def generate_sample(
	model,
	input_ids,
	max_new_tokens,
	context_size,
	sampling_strategy="probabilistic",
	temperature=0.0,
	top_k=None,
	eos_token_id=None,
):

	#
	sampling_method = getattr(Sampling, sampling_strategy, None)
	if not callable(sampling_method):
		raise ValueError(f"Unknown sampling type: {sampling_strategy}. Must be one of: greedy, probabilistic, top_k, temperature.")
	
	"""Sample tokens from ``model``, preserving the prompt in the output."""

	for _ in range(max_new_tokens):
		model_input = input_ids[:, -context_size:]
		logits = model(model_input)
		#need last prediction
		nxt_logit = logits[:, -1, :]

		nxt_id = sampling_method(nxt_logit)
		input_ids = torch.cat((input_ids, nxt_id), dim=1)
		if eos_token_id is not None and torch.all(nxt_id == eos_token_id):
			break

	return input_ids


def generate_text(device, model, tokenizer, prompt, max_new_tokens=50, context_size=None, **kwargs):
	"""Generate and decode a continuation of ``prompt``.

	``generate_sample`` is expected to return token IDs including the prompt.
	Sampling options supported by that function can be passed through ``kwargs``.
	"""
	
	input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)
	model.eval()
	# with torch.inference_mode(): #read abouth this before using it
	output_ids = generate_sample(
			model,
			input_ids,
			max_new_tokens=max_new_tokens,
			context_size=context_size,
			**kwargs,
	).squeeze(0).cpu().numpy().tolist()

	return tokenizer.decode(output_ids, skip_special_tokens=True)


def infer(config: Config):
    """Build the model architecture."""
    # Set random seed for reproducibility
    torch.manual_seed(config.training.seed)

    
	#======get tokenizer first==============
    project_root = Path(__file__).resolve().parent.parent
    artifact_dir = project_root / "dataset_artifacts" / "gpt"
    tokenizer_path = artifact_dir / "gpt.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
	#=========================================

	# Initialize the model
    config.data.vocab_size = tokenizer.get_vocab_size() # this is important to set the vocab size in config before model initialization
    model = get_model(config)

    if config.model.resume_from is None:
    	raise ValueError("resume_from must be specified in the config for inference.")
    device = config.training.device
    ckpoint = torch.load(config.model.resume_from, map_location=device	)

    model.load_state_dict(ckpoint["model_state_dict"])
    model.to(device)

    out = generate_text(device, model, tokenizer, prompt="Once upon a time", max_new_tokens=50, context_size=config.data.seq_len)
    print(out)

def main() -> None:
    """Parse CLI arguments and launch training from the selected config."""
    parser = argparse.ArgumentParser(description="Train a configured language model.")
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


