# LLM Internals — Roadmap

Build an LLM from first principles, understand each component, and progressively explore modern LLM techniques.

## 1. Transformer Foundations

* [ ] Tokenization
* [ ] Token embeddings
* [ ] Positional embeddings
* [x] Scaled dot-product attention
* [ ] Causal masking
* [x] Single-head attention
* [x] Multi-head attention
* [x] Feed-forward network
* [x] Residual connections
* [x] LayerNorm
* [x] Single-layer Transformer
* [ ] Language-model head
* [ ] Next-token prediction
* [ ] Text generation

## 2. Training

* [ ] Cross-entropy loss
* [ ] Backpropagation
* [ ] Adam / AdamW
* [ ] Learning-rate scheduling
* [ ] Warmup
* [ ] Gradient clipping
* [ ] Gradient accumulation
* [ ] Checkpointing
* [ ] Validation & perplexity

## 3. Modern LLM Components

* [ ] Pre-LN vs Post-LN
* [ ] RMSNorm
* [ ] RoPE
* [ ] SwiGLU
* [ ] Weight tying
* [ ] MQA
* [ ] GQA
* [ ] KV cache
* [ ] Sampling: temperature / top-k / top-p
* [ ] Flash Attention

## 4. Efficiency

* [ ] Parameter counting
* [ ] FLOPs estimation
* [ ] Memory analysis
* [ ] Mixed precision
* [ ] Quantization
* [ ] Pruning
* [ ] Knowledge distillation
* [ ] Speculative decoding

## 5. Scaling

* [ ] Scale model depth
* [ ] Scale model width
* [ ] Scale context length
* [ ] Training-data experiments
* [ ] Scaling-law experiments
* [ ] Benchmark training speed
* [ ] Benchmark inference speed

## 6. Advanced LLMs

* [ ] Mixture-of-Experts
* [ ] Sparse attention
* [ ] LoRA
* [ ] QLoRA
* [ ] DPO
* [ ] RLHF concepts
* [ ] RAG

## 7. Research Experiments

* [ ] Early-exit Transformers
* [ ] Layer skipping
* [ ] Token-level routing
* [ ] Dynamic attention heads
* [ ] Dynamic MLP computation
* [ ] Compute-budgeted inference
* [ ] Learned routing
* [ ] Compare dynamic vs dense models

## 8. Documentation

For every major component:

* [ ] Mathematical explanation
* [ ] From-scratch implementation
* [ ] Unit tests
* [ ] Small experiment
* [ ] Results / visualization
* [ ] Key observations
* [ ] References

## Final Goal

> **From a single-layer Transformer → understand modern LLMs → investigate efficient and dynamic LLM architectures.**
