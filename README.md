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

## Local Setup

Recommended environment manager: [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Liinn0529/tinygpt-from-scratch.git
cd tinygpt-from-scratch

uv sync --extra dev
uv run pytest
uv run python -m tinygpt --config configs/debug.yaml
```

Expected smoke-check output includes the TinyGPT version, selected device, config path, and loaded config sections.

You can also run:

```bash
uv run tinygpt-smoke --config configs/debug.yaml
```

Device selection prefers CUDA, then Apple MPS, then CPU.

## Principles

- Implement the core model ourselves.
- Do not use `GPT2LMHeadModel`, `LlamaForCausalLM`, or Hugging Face `Trainer`.
- Every core module gets unit tests.
- Tensor-heavy modules document shapes.
- First make it correct, then make it fast.
- Keep CPU, Apple MPS, and CUDA compatibility when practical.
- Treat training data, experiments, evaluation, and reproducibility as first-class parts of the project.

## Project Documents

- [SPEC.md](SPEC.md) — architecture, scope, completion criteria
- [ROADMAP.md](ROADMAP.md) — 14-week implementation plan
- [LEARNING_GUIDE.md](LEARNING_GUIDE.md) — theory, derivations, interview questions, weekly learning gates
- [AGENTS.md](AGENTS.md) — rules Codex/agents must follow
- [CODEX_WORKFLOW.md](CODEX_WORKFLOW.md) — how to assign, review, test, and merge Codex work

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
11. Final corpus preparation
12. Final ~40M pretraining run
13. Evaluation and ablations
14. Supervised fine-tuning
15. KV-cache inference, CLI chat, and final report

## Recommended Workflow

Work through the GitHub Issues in numerical order.

For every issue:

```text
read issue
  ↓
ask Codex for algorithm + interface + test plan
  ↓
implement smallest correct version
  ↓
run tests
  ↓
human explanation/review
  ↓
merge
  ↓
move to next issue
```

Do not skip the overfit-one-batch gate before real pretraining.

## Status

Project scaffold and semester roadmap are initialized.

**Start with GitHub Issue #1: Engineering foundation and reproducibility.**
