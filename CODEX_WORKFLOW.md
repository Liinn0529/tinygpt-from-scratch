# Codex Workflow

This project is deliberately organized so Codex can help without turning the repository into a black box.

## 1. One Issue = One Unit of Work

Do not ask Codex to "build the whole LLM".

Instead, work sequentially through GitHub Issues.

For each issue:

1. read `SPEC.md`
2. read `ROADMAP.md`
3. read `AGENTS.md`
4. read the target GitHub Issue
5. inspect only the relevant files
6. propose the smallest implementation plan
7. implement
8. run tests
9. summarize changes
10. open a PR or prepare a clean commit

## 2. Standard Prompt Template

Use this as the default Codex instruction:

```text
Read SPEC.md, ROADMAP.md, LEARNING_GUIDE.md, and AGENTS.md.

Work only on GitHub Issue #<N>.

Before coding:
1. summarize the algorithm
2. list the files you expect to change
3. define public interfaces
4. document important tensor/data shapes
5. list edge cases
6. describe the tests you will write

Then implement the smallest correct version.

Constraints:
- do not modify unrelated modules
- do not use Hugging Face model classes or Trainer
- do not hide core logic behind high-level libraries
- preserve CPU compatibility
- preserve MPS/CUDA compatibility when feasible
- add tests for every new core component

After coding:
1. run the relevant tests
2. report files changed
3. report test results
4. report known limitations
5. explain the algorithm in plain language
6. stop; do not start the next issue
```

## 3. Branch Naming

Recommended branch names:

```text
issue-01-foundation
issue-02-bpe
issue-03-transformer-primitives
issue-04-attention
issue-05-decoder-lm
issue-06-training
...
```

## 4. Human Review Gate

Before merging a Codex change, you should be able to answer:

- What algorithm was implemented?
- What are the input/output shapes?
- Where is the core mathematical operation?
- What is the computational complexity?
- Which tests demonstrate correctness?
- What would fail if one key line were removed?

If you cannot answer those, do not merge yet.

## 5. Neural-Network Module Review Checklist

For RMSNorm, RoPE, attention, SwiGLU, Transformer blocks, and similar modules:

- shape test
- dtype/device behavior
- forward numerical sanity
- backward/gradient test
- comparison with a tiny reference where possible
- edge-case sequence lengths
- no accidental future-token access for causal attention

## 6. Training-Code Review Checklist

Before a real training run:

- tiny forward/backward passes
- loss is finite
- gradient norm is finite
- overfit-one-batch passes
- checkpoint save/load passes
- checkpoint resume matches uninterrupted training within tolerance
- random seed is recorded
- config is saved
- git commit is recorded

## 7. Experiment Workflow

Every meaningful experiment should record:

```text
experiment name
git commit
config file
device
dataset/version
tokenizer/version
seed
token budget
parameter count
wall time
metrics
conclusion
limitations
```

Store notes under `experiments/`.

## 8. Debugging Prompt Template

When something fails, do not ask Codex to rewrite everything.

Use:

```text
Investigate this failure without changing unrelated code.

Observed behavior:
<paste error/log>

Expected behavior:
<describe expected result>

First:
1. identify the most likely failure points
2. propose the smallest diagnostic tests
3. inspect relevant code paths

Only after reproducing the issue, make the smallest fix.

Add a regression test that fails before the fix and passes after it.
```

## 9. Performance Optimization Rule

Correctness comes first.

The order is:

```text
readable reference implementation
        ↓
correctness tests
        ↓
benchmark
        ↓
optimized implementation
        ↓
numerical equivalence check
        ↓
benchmark again
```

Do not replace a readable implementation with an optimized one unless both remain testable.

## 10. When to Ask Codex for Explanations

After each issue, ask Codex:

```text
Teach me the implementation you just made.

For every important function:
- explain its purpose
- explain the math
- explain tensor shapes
- explain complexity
- explain common bugs
- give me 3 interview questions
- ask me 3 comprehension questions
```

This turns Codex from a code generator into a study partner.

## 11. Pull Request Template

A good PR should answer:

- What changed?
- Why?
- Which issue does it close?
- What tests were run?
- Are there numerical/reference comparisons?
- What remains intentionally unimplemented?
- Are there performance implications?

## 12. Merge Policy

Merge only when:

- acceptance criteria in the issue are met
- tests pass
- you understand the implementation
- no unrelated changes are mixed in
- the next milestone does not depend on hidden assumptions

The project should remain readable enough that, at the end, you can trace a token from raw text all the way through training and generation.
