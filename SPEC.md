# TinyGPT Technical Specification

## 1. Objective

Train a decoder-only language model from random initialization on a single personal computer while implementing and understanding the complete pipeline.

## 2. Model stages

### Debug model
- 2–5M parameters
- context length: 128–256
- purpose: correctness, shape checks, and overfit-one-batch tests

### Research model
- 10–20M parameters
- context length: 256–512
- purpose: hyperparameter and ablation experiments

### Final model
- approximately 30–50M parameters
- target: ~40M
- context length: 512

## 3. Reference final architecture

- decoder-only Transformer
- vocab size: 16,384
- layers: 10
- model width: 512
- attention heads: 8
- FFN hidden size: approximately 1,360
- normalization: RMSNorm
- positional encoding: RoPE
- activation: SwiGLU
- tied input/output embeddings
- no linear biases unless an experiment requires them

## 4. Core components to implement

- byte-level BPE tokenizer
- token embeddings
- RMSNorm
- RoPE
- causal multi-head self-attention
- SwiGLU MLP
- Transformer block and language-model head
- cross-entropy objective
- AdamW
- warmup + cosine learning-rate schedule
- gradient clipping and accumulation
- checkpoint save/resume
- generation with temperature / top-k / top-p
- KV cache

## 5. Data pipeline

`raw → normalize → filter → deduplicate → split → tokenize → pack`

Start with a TinyStories-scale open dataset for correctness. The final corpus must use clearly licensed or otherwise appropriate data sources.

## 6. Evaluation

Track at minimum:
- train loss
- validation loss
- perplexity
- learning rate
- gradient norm
- tokens/sec
- checkpoint size
- peak memory where available
- generation samples from fixed prompts

## 7. Required experiments

1. vocab size: 8K vs 16K
2. model-size scaling: 4M / 8M / 16M / 32M
3. GELU vs SwiGLU
4. raw vs cleaned corpus
5. base model vs SFT model

## 8. Completion criteria

The project is complete when one reproducible command path covers:

`data preparation → tokenizer training → pretraining → evaluation → SFT → local inference`

and the repository contains tests, experiment records, plots, configs, and a technical report.
