# 14-Week Roadmap

## Week 1 — Engineering foundation
- package layout
- `uv`/Python project config
- pytest
- device selection
- deterministic seeds
- coding conventions

**Gate:** `pytest` runs successfully on CPU and device detection is tested.

## Week 2 — Byte-level BPE
- pre-tokenization
- byte vocabulary
- pair statistics
- merge loop
- encode/decode
- save/load
- round-trip tests

**Gate:** Unicode, Chinese, emoji, punctuation, newline, and empty-input tests pass.

## Week 3 — Transformer primitives
- RMSNorm
- RoPE
- SwiGLU
- numerical/shape tests

## Week 4 — Attention and decoder
- causal self-attention
- mask correctness
- multi-head reshape logic
- Transformer block
- full decoder LM

**Gate:** forward and backward tests pass.

## Week 5 — Optimization and training
- AdamW
- warmup + cosine scheduler
- gradient clipping
- gradient accumulation
- checkpointing
- structured logging

**Critical gate:** overfit one batch until loss becomes very small.

## Week 6 — First real training run
- TinyStories-scale corpus
- 5–10M model
- fixed prompt sampling
- train/validation curves

## Week 7 — Data engineering
- normalization
- document filtering
- length/language checks
- exact and approximate deduplication
- split strategy
- token packing

## Week 8 — Scaling study
Train 4M / 8M / 16M / 32M models under controlled token budgets and record validation loss, wall time, throughput, and a compute proxy.

## Week 9 — Systems study
Compare:
- naive attention
- PyTorch SDPA
- fp32 vs supported mixed precision
- batch size
- gradient accumulation
- optional `torch.compile`

CUDA stretch goals: Triton and FlashAttention.

## Week 10 — Final corpus
Build a 100M–200M token corpus appropriate for available compute and storage.

## Week 11 — Final pretraining
Train the ~40M model and save reproducible checkpoints.

## Week 12 — Evaluation and ablations
Run required experiments and write results.

## Week 13 — SFT
- conversation template
- instruction dataset
- assistant-only loss masking
- base vs SFT comparison

## Week 14 — Inference and polish
- KV cache
- temperature/top-k/top-p
- CLI chat
- final README
- technical report
- reproducibility guide
