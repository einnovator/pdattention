# Prompt for Codex: integrate PRA dataset pack

Integrate the `data/` directory into the existing PRA repo.

Focus only on the standalone PyTorch path first.

Implement:

1. Dataset loader module:
   - `pra_core/datasets.py`
   - `PRAExample`, `PRADocument`, `PRAReference` dataclasses or Pydantic models.
   - JSONL loading for `documents.jsonl`, `references.jsonl`, `questions.jsonl`.

2. Reference integration:
   - Register every row from `references.jsonl` into `ReferenceTable`.
   - Ensure question prompts use `<REF_N>` tokens.
   - Resolver should resolve documents by `uri` and anchors.

3. Training integration:
   - Add CLI flags:
     - `--data-dir data`
     - `--dataset-stage stage0_synthetic_memory`
     - `--max-examples`
   - Use `questions.jsonl` as supervised examples.

4. Evaluation baselines:
   - no retrieval
   - full context oracle
   - simple RAG oracle by expected ref
   - PRA reference retrieval

5. Metrics:
   - exact match answer
   - expected reference selected
   - expected anchor selected
   - expansion count
   - retrieved token count
   - full-context token count

6. Tests:
   - one test per stage loads sample data
   - tokenizer treats `<REF_1>` as one token
   - resolver can resolve base URI and anchored URI
   - PRA forward works with at least one active reference

Do not work on Hugging Face/Llama/Mistral/Qwen wrappers until the standalone path passes tests.
