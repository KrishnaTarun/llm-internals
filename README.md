# LLM Internals

A from-scratch journey into understanding how modern Large Language Models work.

The goal of this repository is to **learn by implementing**. I start with the original **Transformer architecture from _Attention Is All You Need_**, building a minimal sequence-to-sequence model from first principles. From there, the project will progressively move toward decoder-only architectures and explore the key ideas behind modern LLMs.

The focus is not on building the largest model, but on understanding the **mechanisms, design choices, and trade-offs** that make these models work.

## Learning Path

You can read and follow the implementation without installing the environment or downloading the datasets. For a guided tour, read the files in this order:

1. **Compare the configurations:** [`seq2seq_training.yaml`](configs/seq2seq_training.yaml) and [`gptstyle_training.yaml`](configs/gptstyle_training.yaml) show the model sizes, layer counts, and sequence lengths used in the examples.
2. **Learn the building blocks:** [`attention.py`](src/model_components/attention.py), [`pos_encoding.py`](src/model_components/pos_encoding.py), and [`sub_blocks.py`](src/model_components/sub_blocks.py) implement multi-head attention, positional information, embeddings, feed-forward layers, and residual connections.
3. **Follow the Transformer layers:** [`encoder.py`](src/model_components/encoder.py) builds the encoder stack; [`decoder.py`](src/model_components/decoder.py) contains both the encoder-connected Seq2Seq decoder and the causal GPT-style decoder.
4. **Compare complete models:** [`model.py`](src/model.py) assembles these components into the Seq2Seq and GPT-style models.
5. **Trace a training run:** [`tokenization.py`](src/tokenization.py) and [`dataset.py`](src/dataset.py) prepare tokens and batches; [`train.py`](src/train.py), [`trainer.py`](src/trainer.py), and [`utils.py`](src/utils.py) connect configuration, model creation, optimization, and checkpointing.

At a high level, Seq2Seq encodes an input sequence and uses its decoder to produce a target sequence. GPT-style is decoder-only: causal attention lets each position use earlier tokens to predict the next one. Both models are implemented from scratch here; they are not pretrained models.

## Why Seq2Seq and GPT-Style?

Seq2Seq is a good place to begin because it shows both main parts of the original Transformer: an **encoder** reads the input, and a **decoder** creates the output using what the encoder learned. In translation, for example, the encoder reads an English sentence and the decoder generates its Dutch translation. This makes it easier to study attention, masking, and how the encoder and decoder work together.

GPT-style focuses on the **decoder**. It does not have a separate encoder; instead, causal attention lets it use the text so far to predict the next token. This is the basic pattern behind many decoder-only language models.

Studying Seq2Seq first gives you a foundation for understanding the Transformer parts. Comparing it with GPT-style then shows how a model can use the decoder on its own for language generation. These two architectures are useful starting points, though they do not cover every type of language model.

## Setting up the Environment

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

## System Information

The project was run on the following system:

- **Device:** MacBook Air (MacBookAir8,2)
- **Processor:** 1.6 GHz dual-core Intel Core i5
- **Memory:** 8 GB
- **Operating system:** macOS 14.8.7
- **Architecture:** x86_64
- **Python:** 3.11

Training time and resource usage may vary on other hardware and operating systems.

## Running the Models

### Datasets

Training downloads the source datasets from Hugging Face Datasets on the first run, so an internet connection is required. Downloaded datasets are managed by the Hugging Face cache.

- **Seq2Seq translation:** Uses the English-to-Dutch (`en-nl`) configuration of [OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books), taking its `train` split and dividing it into 80% training and 20% validation examples. The English and Dutch WordLevel tokenizers are stored in `dataset_artifacts/translation/`. Inputs are truncated or padded to 100 tokens.
- **GPT-style language model:** Uses the raw-text `wikitext-2-raw-v1` configuration of [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), with its `train` and `validation` splits. Text is tokenized with a WordLevel tokenizer stored in `dataset_artifacts/gpt/gpt.json`, then divided into overlapping 100-token sequences with a stride of 50.

The tokenizer JSON files are generated locally under the corresponding `dataset_artifacts/` (generated locally as well) subdirectory.

### Training

Run commands from the repository root. The training script accepts a YAML config; if `--config` is omitted, it defaults to `configs/gptstyle_training.yaml`.

To train the GPT-style model:

```sh
uv run python src/train.py --config configs/gptstyle_training.yaml
```

The Seq2Seq configuration is `configs/seq2seq_training.yaml`, and its intended command is:

```sh
uv run python src/train.py --config configs/seq2seq_training.yaml
```

<!-- However, Seq2Seq training currently fails before training starts: the dataset loader returns three values while the training entry point expects two, and model construction refers to an undefined dataset variable. The Seq2Seq command will work after that integration issue is fixed.

Both configs currently specify 10 epochs and a batch size of 64. Training and validation losses are printed in the terminal. The default device is CUDA when available and CPU otherwise. -->

**GPU note:** Training has been tested on CPU. The current [`pyproject.toml`](pyproject.toml) configures `torch` and `torchvision` to install from PyTorch's CPU-only package index, so using a GPU requires installing PyTorch builds compatible with your operating system and GPU, and adjusting the package source configuration. Follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for the right build. The current device selection checks for CUDA; it does not select Apple's MPS backend automatically. GPU setup may therefore need some platform-specific adjustments and verification.

**The number of encoder and decoder layers can be configured as per requirement. The supplied configs use relatively few layers to keep training compute and memory requirements manageable: Seq2Seq uses 2 encoder and 1 decoder layer, while GPT-style uses 3 decoder layers.**

### Training Outputs

Checkpoints are saved under `output/checkpoints/<model_type>/model.pt`, for example `output/checkpoints/GPTstyle/model.pt` or `output/checkpoints/Seq2Seq/model.pt`. The same checkpoint file is overwritten after each epoch; it stores the model state, optimizer state, and completed epoch. To resume, set `training.resume_from` in the selected YAML config to the checkpoint path.

The current training loop does not save generated text or a separate predictions file. It saves checkpoints and prints loss metrics to the terminal.

## Directory Structure

```text
.
├── configs/                 # Training configurations
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
