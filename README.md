# TinyGPT From Scratch

A single-machine, from-scratch small language model project inspired by Stanford CS336 and semester-long “train a GPT” projects.

## Goal

Build the complete language-model lifecycle locally:

```text
raw text
  ↓
data cleaning
  ↓
byte-level BPE tokenizer
  ↓
decoder-only Transformer
  ↓
pretraining
  ↓
evaluation
  ↓
supervised fine-tuning
  ↓
local inference
```

The final target is a ~40M parameter model, while development starts with a tiny debug model that can run on a personal computer.

## Principles

- Implement the core model ourselves.
- Do not use `GPT2LMHeadModel`, `LlamaForCausalLM`, or Hugging Face `Trainer`.
- Every core module gets unit tests.
- Tensor-heavy modules document shapes.
- First make it correct, then make it fast.
- Keep CPU, Apple MPS, and CUDA compatibility when practical.
- Treat training data, experiments, evaluation, and reproducibility as first-class parts of the project.

## Milestones

1. Engineering foundation
2. Byte-level BPE tokenizer
3. RMSNorm, RoPE, and SwiGLU
4. Causal multi-head self-attention
5. Decoder-only Transformer
6. Optimizer and training loop
7. TinyStories-scale first training run
8. Data cleaning and deduplication
9. Scaling-law experiments
10. Systems benchmarks
11. Final ~40M pretraining run
12. Evaluation and ablations
13. Supervised fine-tuning
14. KV-cache inference and CLI chat

See [SPEC.md](SPEC.md), [ROADMAP.md](ROADMAP.md), and [AGENTS.md](AGENTS.md).

## Status

Project initialization in progress.
