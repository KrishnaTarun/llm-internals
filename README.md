# LLM Internals

A from-scratch journey into understanding how modern Large Language Models work.

The goal of this repository is to **learn by implementing**. I start with the original **Transformer architecture from _Attention Is All You Need_**, building a minimal sequence-to-sequence model from first principles. From there, the project will progressively move toward decoder-only architectures and explore the key ideas behind modern LLMs.

The focus is not on building the largest model, but on understanding the **mechanisms, design choices, and trade-offs** that make these models work.

## Working Environment

This project uses Python 3.11 and [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.

1. Install Python 3.11 and uv if they are not already installed depending on OS. On macOS, uv can be installed with `brew install uv`.
2. From the repository root, create the virtual environment with Python 3.11:

	```sh
	uv venv --python 3.11 .venv
	```

3. Install the locked project dependencies, including the development tools:

	```sh
	uv sync
	```

4. Activate the environment in macOS or Linux:

	```sh
	source .venv/bin/activate
	```

## Directory Structure

```text
.
├── configs/                 # Training configurations
├── dataset_artifacts/       # Tokenizers and prepared dataset artifacts
├── src/
│   ├── model_components/    # Encoder, decoder, attention, and sub-blocks
│   ├── config.py            # Configuration loading
│   ├── dataset.py           # Dataset preparation
│   ├── model.py             # Model definitions
│   ├── tokenization.py      # Tokenizer utilities
│   ├── train.py             # Training entry point
│   ├── trainer.py           # Training loop
│   └── utils.py             # Shared utilities
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Locked dependency versions
└── .python-version          # Project Python version
```
