# TinyGPT Learning Guide

This guide turns the repository roadmap into a learning curriculum. The goal is not only to finish the code, but to understand why each part works well enough to explain it in an interview or technical discussion.

---

## Week 1 — Engineering Foundation

### You should understand

- Python package layout
- reproducible experiments
- random seeds
- CPU / Apple MPS / CUDA device selection
- configuration-driven experiments
- why tests matter before training

### You should be able to explain

1. Why is reproducibility difficult in deep learning?
2. What does a random seed control?
3. Why can the same seed still produce slightly different GPU results?
4. Why should model hyperparameters live in config files rather than source code?
5. Why is an overfit-one-batch test useful?

### Coding objective

Build the smallest reliable development environment before any model logic.

### Acceptance

- package installs
- `pytest` passes
- config loading works
- deterministic seed behavior is tested
- device detection works

---

## Week 2 — Byte-Level BPE Tokenizer

### Core theory

Language models operate on token IDs rather than raw text.

A byte-level BPE tokenizer starts from byte symbols and repeatedly merges the most frequent adjacent pair.

At each iteration:

```text
count adjacent pairs
        ↓
choose most frequent pair
        ↓
merge that pair
        ↓
add merged symbol to vocabulary
        ↓
repeat
```

### You should understand

- tokenization vs vocabulary
- Unicode vs UTF-8 bytes
- byte-level tokenization
- BPE merge rules
- vocabulary size trade-offs
- sequence length vs vocabulary size
- special tokens

### You should be able to answer

1. Why not use one token per character?
2. Why not use one token per word?
3. Why is byte-level tokenization robust to arbitrary Unicode?
4. What happens if vocabulary size becomes too large?
5. Why does tokenizer choice affect training FLOPs?
6. Why must the tokenizer be frozen before pretraining?

### Required experiment

Compare an 8K and 16K vocabulary on:
- mean tokens per document
- tokenized corpus size
- training throughput
- validation loss under controlled settings

---

## Week 3 — RMSNorm, RoPE, SwiGLU

# RMSNorm

For hidden state `x ∈ R^d`:

```text
RMS(x) = sqrt(mean(x^2) + eps)
RMSNorm(x) = g * x / RMS(x)
```

Understand the difference between RMSNorm and LayerNorm.

### Interview questions

- Why normalize hidden states?
- What does RMSNorm remove compared with LayerNorm?
- What is learned in RMSNorm?
- Does RMSNorm change vector direction?

# RoPE

For each pair of hidden dimensions, rotate by a position-dependent angle.

For a 2D pair:

```text
[x_1']
[x_2'] =
[ cos θ  -sin θ ] [x_1]
[ sin θ   cos θ ] [x_2]
```

RoPE is applied to Q and K, not normally V.

### Key property

The dot product between rotated Q and K depends on relative position.

You should understand why:

```text
<R(pos_i) q, R(pos_j) k>
```

depends on `pos_j - pos_i`.

### Interview questions

- Why apply RoPE to Q and K?
- Why not V?
- How does RoPE encode relative position?
- Why can long-context extrapolation fail?
- What is the connection between RoPE frequency and context length?

# SwiGLU

Typical form:

```text
SwiGLU(x) = SiLU(x W_gate) ⊙ (x W_up)
output = SwiGLU(x) W_down
```

### Interview questions

- Why use gated MLPs?
- GELU vs SwiGLU?
- Why is the FFN hidden dimension often not exactly 4 × d_model in modern LLMs?

---

## Week 4 — Causal Self-Attention and Decoder-Only Transformer

### Core equation

```text
Attention(Q,K,V)
= softmax(QK^T / sqrt(d_k) + mask) V
```

### Shapes

For:
- batch size `B`
- sequence length `T`
- model width `D`
- number of heads `H`
- head dimension `d_h = D/H`

Typical shapes:

```text
X:       [B, T, D]
Q,K,V:   [B, T, D]
heads:   [B, H, T, d_h]
scores:  [B, H, T, T]
output:  [B, T, D]
```

### You must understand

- Q / K / V meaning
- causal masking
- why divide by `sqrt(d_k)`
- attention complexity
- residual connections
- pre-norm Transformer
- weight tying
- next-token prediction

### Derivation target

Be able to explain why self-attention has approximately:

```text
O(T^2 D)
```

attention complexity.

### Interview questions

1. Why does attention scale quadratically with sequence length?
2. Why is causal masking required in autoregressive pretraining?
3. MHA vs MQA vs GQA?
4. Why use residual connections?
5. Pre-norm vs post-norm?
6. Why tie embedding and output weights?

---

## Week 5 — Optimization and Training Loop

### Cross entropy

Given logits `z` and correct token `y`:

```text
p_i = exp(z_i) / Σ_j exp(z_j)
L = -log p_y
```

Language-model loss averages this over predicted next tokens.

### AdamW

Understand:
- first moment estimate
- second moment estimate
- bias correction
- decoupled weight decay

### Learning-rate schedule

Typical plan:

```text
warmup
  ↓
peak learning rate
  ↓
cosine decay
```

### You should be able to answer

- Adam vs AdamW?
- Why warm up the learning rate?
- Why clip gradients?
- Gradient accumulation vs larger batch?
- What does batch size mean in tokens?
- Why might training diverge?
- How do you diagnose NaNs?

### Critical experiment

Overfit one batch.

If the model cannot memorize a tiny batch, do not start real pretraining.

---

## Week 6 — First Pretraining Run

### Objective

Train a 5–10M model on a small story-style dataset.

### Learn to interpret

- train loss
- validation loss
- overfitting
- underfitting
- learning-rate curve
- gradient norm
- tokens/sec
- sample quality

### Questions to answer

1. When does generation change from noise into grammatical text?
2. Does validation loss track perceived generation quality?
3. What happens when learning rate is too high?
4. What happens when context length is too short?

### Required artifacts

- training curve
- validation curve
- fixed-prompt samples at multiple checkpoints
- experiment note with config and commit hash

---

## Week 7 — Data Engineering

### Core pipeline

```text
raw documents
    ↓
normalization
    ↓
format cleanup
    ↓
quality filtering
    ↓
language/length filtering
    ↓
deduplication
    ↓
train/validation split
    ↓
tokenization
    ↓
sequence packing
```

### You should understand

- exact deduplication
- approximate deduplication
- data leakage
- train/validation contamination
- document boundaries
- packing efficiency
- data quality vs data quantity

### Interview questions

- Why deduplicate pretraining data?
- Why is benchmark contamination dangerous?
- How can near-duplicate detection work?
- Why can low-quality data hurt even when there is more of it?
- What is token packing and why does it improve utilization?

---

## Week 8 — Scaling Laws

### Goal

Train multiple model sizes and token budgets.

Recommended grid:

```text
Models:
4M
8M
16M
32M

Token budgets:
10M
20M
40M
80M
```

Not every combination must be trained if local compute is limited.

### Core ideas

Study empirical relationships among:

- model parameters
- data tokens
- compute
- validation loss

A simple fit may use:

```text
L(C) = A C^{-alpha} + B
```

### You should be able to discuss

- model scaling
- data scaling
- compute-optimal training
- why small-scale experiments may predict larger runs
- limitations of extrapolation

### Interview questions

- What is a scaling law?
- What does compute-optimal mean?
- Why can a model be undertrained?
- Why might more parameters be worse under fixed compute?

---

## Week 9 — Systems and Performance

### Benchmark

Compare:

- naive attention
- PyTorch SDPA
- fp32
- bf16/fp16 where supported
- different batch sizes
- gradient accumulation
- optional `torch.compile`

### Measure

- tokens/sec
- latency
- peak memory
- training step time

### Core theory

Understand that attention can be limited by:

- arithmetic throughput
- memory bandwidth
- activation memory
- kernel launch overhead

### Interview questions

- What is a CUDA kernel?
- Why is FlashAttention faster even though asymptotic complexity remains O(T^2)?
- What is IO-aware computation?
- Why does mixed precision reduce memory?
- bf16 vs fp16?
- Why can a larger batch improve GPU utilization?

---

## Week 10 — Final Corpus

### Objective

Prepare a 100M–200M token corpus that your machine can realistically train on.

### Required decisions

Document:

- source
- license / usage suitability
- language distribution
- filtering rules
- dedup strategy
- token count
- validation split policy

### Acceptance

Produce a dataset card in the repository describing exactly what was used and how it was prepared.

---

## Week 11 — Final Pretraining

### Target

Train the ~40M model.

### You should monitor

- loss
- validation loss
- learning rate
- grad norm
- throughput
- memory
- checkpoint health
- fixed-prompt generations

### Failure modes to recognize

- divergence
- NaNs
- dead learning
- data loader bottleneck
- pathological repetition
- checkpoint corruption
- validation leakage

### Required result

A base model checkpoint and reproducible training record.

---

## Week 12 — Evaluation and Ablation

### Required comparisons

1. 8K vs 16K BPE
2. model-size scaling
3. GELU vs SwiGLU
4. raw vs cleaned data
5. base vs SFT

### Evaluation dimensions

- language-model loss
- perplexity
- generation quality
- speed
- memory
- parameter count

### Important mindset

Do not report only a single score.

For every experiment state:

- hypothesis
- controlled variables
- changed variable
- measurement
- conclusion
- limitation

---

## Week 13 — Supervised Fine-Tuning

### Data format

Example:

```text
<|user|>
Explain Newton's second law.
<|assistant|>
Newton's second law states ...
```

### Key idea

During SFT, usually compute loss only on assistant response tokens.

### You should understand

- base model vs instruction model
- SFT loss masking
- chat templates
- catastrophic forgetting
- data quality in instruction tuning

### Interview questions

- Why does SFT change behavior without adding new architecture?
- Why mask user tokens?
- SFT vs continued pretraining?
- Full fine-tuning vs LoRA?

---

## Week 14 — Inference and KV Cache

### Generation

Implement:

- greedy decoding
- temperature
- top-k
- top-p

### KV cache

Without KV cache, each generated token recomputes keys and values for the entire prefix.

With KV cache:

```text
past K,V
   +
new token K,V
   ↓
reuse previous states
```

### You should be able to answer

- Why does KV cache speed up autoregressive decoding?
- What memory does KV cache consume?
- Why does inference still process one token at a time?
- Prefill vs decode?
- Throughput vs latency?

---

# Final Interview Checklist

By the end of the project, you should be able to explain the complete path:

```text
raw UTF-8 text
→ byte-level BPE
→ token IDs
→ embeddings
→ Q/K/V
→ RoPE
→ causal attention
→ residual stream
→ SwiGLU
→ logits
→ cross entropy
→ backward pass
→ AdamW
→ checkpoint
→ sampling
→ KV cache
```

You should also be able to answer:

1. Why does a Transformer learn next-token prediction?
2. Why does next-token prediction produce general language ability?
3. Where does O(n²) arise?
4. What limits context length?
5. Why does data quality matter?
6. How do model size, data size, and compute interact?
7. Why is FlashAttention faster?
8. How does SFT differ from pretraining?
9. Why does KV cache matter?
10. What would you change if scaling from 40M to 7B parameters?

If you can explain these from your own implementation, the project has achieved its learning goal.
