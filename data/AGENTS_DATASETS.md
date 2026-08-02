# Codex instructions: integrate PRA datasets

Prioritize standalone PyTorch. Do not modify Hugging Face wrappers until the standalone loaders and evaluations work.

Tasks:

1. Add a dataset module, e.g. `pra_core/datasets.py`.
2. Support JSONL schemas in this `data/` directory.
3. Add train/eval CLI arguments:
   - `--dataset-stage stage0_synthetic_memory`
   - `--data-dir data`
   - `--max-examples`
4. Implement loaders for:
   - documents
   - references
   - questions
5. Register references into `ReferenceTable` before tokenization.
6. Ensure `<REF_1>` etc. are atomic tokens.
7. Add metrics:
   - `answer_exact_match`
   - `expected_ref_hit`
   - `expected_anchor_hit`
   - `num_expansions`
   - `retrieved_tokens`
   - `full_context_tokens`
8. Add tests for all stages using the tiny sample files.
9. Add docs explaining how each stage increases realism.

Design principle:
Start with synthetic data where the retrieval target is known, then progress toward code repos, docs, Wikipedia, and GitHub-like structures.
