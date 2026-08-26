# Paper 3.1 experiments

Paper 3.1 tests generated natural-language summaries as persistent routing
addresses. The selected address always resolves to the original source chunk;
the summary is never supplied as answer evidence and never replaces native K/V.

## Gate order

Run a small validation frontier first. Every generation is cached by model,
prompt, geometry, and exact source hash, so interrupted runs resume safely.

```powershell
python experiments/paper3_1_summary_index/run_study.py `
  --split validation --max-per-dataset 1 `
  --condition tiny_135m:generic:1:32 `
  --condition tiny_135m:retrieval:1:32 `
  --condition subb_600m:retrieval:1:32 `
  --condition candidate_1b:retrieval:1:32 `
  --condition mid_4b:retrieval:1:32 `
  --condition teacher_8b:generic:1:32 `
  --condition teacher_8b:retrieval:1:32
```

The runner reconstructs the inherited HotpotQA, QASPER, 2WikiMultiHopQA, and
MuSiQue identities, asserts tokenizer/chunk-boundary parity, and replays native
mean, source BM25/exact, Paper-2.8 rank-16, compact rank-8/eight-centroid, and
oracle-identity controls. Summary text is scored independently by exact-token,
BM25, frozen MiniLM embeddings, and fixed-weight hybrids.

After inspecting `expanded_validation/validation_policy.json`, rerun the selected frozen
conditions on the untouched test identities. Do not select a scorer, prompt, or
model from test results.

```powershell
python experiments/paper3_1_summary_index/run_study.py `
  --split test --run-name test_subb `
  --datasets qasper musique --max-per-dataset 8 `
  --condition subb_600m:retrieval:1:32

python experiments/paper3_1_summary_index/run_study.py `
  --split test --run-name test_teacher_hotpot `
  --datasets hotpotqa --max-per-dataset 4 `
  --condition teacher_8b:generic:1:32

python experiments/paper3_1_summary_index/run_study.py `
  --split test --run-name test_teacher_2wiki `
  --datasets 2wikimultihopqa --max-per-dataset 4 `
  --condition teacher_8b:retrieval:1:32
```

Equal-token geometry is encoded as `PROFILE:PROMPT:FACETS:TOKENS`. For example,
the following conditions all receive 32 generated routing tokens per chunk:

```powershell
--condition teacher_8b:faceted:1:32
--condition teacher_8b:faceted:2:32
--condition teacher_8b:faceted:4:32
--condition teacher_8b:faceted:8:32
```

The tracked geometry diagnostic uses the 1.2B model on two HotpotQA validation
identities. Run the controlled omission audit with:

```powershell
python experiments/paper3_1_summary_index/run_omission_benchmark.py `
  --profiles subb_600m --prompt retrieval --count-per-type 4 `
  --selection-budget 1 --token-budget 32

python experiments/paper3_1_summary_index/build_publication_artifacts.py
```

The general concept and teacher-headroom gates are mixed/closed. No LoRA or
downstream native-K/V answer-generation result belongs to this experiment.

## Artifact contract

- `summary_cache/*.jsonl`: append-only generated addresses and ingestion timing.
- `<split>/parity_rows.csv`: source identity and chunk-boundary checks.
- `<split>/summary_addresses.jsonl`: clipped addresses with model/prompt provenance.
- `<split>/per_example.csv`: row-level retrieval, bytes, latency, and materialization counts.
- `<split>/summary.csv`: condition means.
- `<split>/paired_effects.csv`: paired bootstrap effects versus native mean.
- `<split>/channel_overlap.csv`: overlap and unique evidence versus source BM25.
- `<split>/manifest.json`: source hashes, model metadata, seeds, and frozen policy.
- `publication/`: frozen headline, paired-effect, cost, omission, and geometry artifacts.

Large inherited Q/K tensors remain untracked in the Paper 2.8 worktree and are
referenced by SHA-256. Summary caches contain no source text, only generated
addresses and exact source hashes.
