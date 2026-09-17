# Codex / Agent Instructions

## Role

Act as an implementation partner for a teaching-oriented, from-scratch LLM project. The human should remain able to explain every major algorithm and design choice.

## Mandatory rules

1. Work on one milestone at a time.
2. Do not rewrite unrelated parts of the repository.
3. Do not replace the project with Hugging Face model classes or Trainer.
4. Document important tensor shapes in tensor-heavy code.
5. Add or update tests for every core module.
6. Run the relevant test suite after changes.
7. Prefer small, transparent implementations before optimization.
8. Keep CPU compatibility; preserve MPS/CUDA support when feasible.
9. Do not hide important logic behind unnecessary abstractions.
10. State tradeoffs when a design choice is ambiguous.
11. Do not silently change public interfaces.
12. Keep experiment configs reproducible and version-controlled.
13. Never commit training data, checkpoints, API keys, or secrets.

## Before coding

For each milestone, summarize:
- algorithm
- public interfaces
- important tensor/data shapes
- edge cases
- intended tests

Then implement the smallest correct version.

## After coding

Report:
- files changed
- implementation summary
- complexity where relevant
- tests run and results
- known limitations
- suggested next milestone

## Quality gates

A core neural-network component is not complete until:
- forward shape tests pass
- backward/gradient tests pass where applicable
- numerical/reference tests pass where possible

Training code is not complete until:
- overfit-one-batch succeeds
- checkpoint resume is tested
- deterministic behavior is documented
