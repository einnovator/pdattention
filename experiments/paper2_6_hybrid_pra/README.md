# Paper 2.6 experiments

The original runners in this directory isolate discovery. The confirmation
runners below additionally test the indexed product path, native-K/V
materialization, and answer generation without rewriting the frozen discovery
cohorts.

## Indexed token-native confirmation

Benchmark exact/token-ngram posting lists, indexed BM25, fuzzy candidate
narrowing, deduplicated channel unions, index memory/build/update cost, and
warm query latency under both Qwen3 and SmolLM2 tokenizers:

```powershell
$env:PYTHONPATH = 'src;.'
python -m experiments.paper2_6_hybrid_pra.run_indexed_token_native --local-files-only
```

The tokenizer invariance table includes case, leading whitespace, punctuation,
hyphenation, concatenation/splitting, typo, abbreviation, identifier alias,
stop-token insertion, and two NFKC-equivalent Unicode forms. Fixed stop-token
suppression and corpus-IDF weighting are separate rows. The measured update is
an immutable full-statistics rebuild after adding 64 chunks; it does not claim
an incremental-IDF implementation.

## End-to-end Gate B

Run 20 validation and 50 fresh test identities per dataset for Qwen3-0.6B and
SmolLM2-135M. Semantic, exact, BM25, approximate, hybrid, adaptive, and oracle
conditions share a four-chunk native-K/V budget. Disabled, shuffled,
irrelevant, empty, and bounded direct-context controls use the same examples:

```powershell
python -m experiments.paper2_6_hybrid_pra.run_end_to_end_confirmation --local-files-only
```

The JSONL checkpoint is append-only during execution. Final CSV generation
deduplicates stable `(model, dataset, example, condition)` identities, so an
interrupted run can be resumed with the same command.

## Channel interactions

Cross all discovery channels with 16/32/64-token chunks, late versus earlier
routing layers, and Top-2/4/8 chunk budgets on a separate held-out slice:

```powershell
python -m experiments.paper2_6_hybrid_pra.run_hybrid_interactions --local-files-only
```

## Final iteration

Replay the frozen 132-identity cohort, export detailed confidence provenance,
run controlled ambiguity fixtures, estimate bootstrap stability, and create the
paper artifacts:

```powershell
$env:PYTHONPATH = 'src;.'
python experiments/paper2_6_hybrid_pra/run_final_iteration.py --local-files-only
```

The runner uses the original cohort seed (`20260811`) only to reconstruct the
cached QASPER/Hotpot identities. Bootstrap and calibration diagnostics use the
independent deterministic seed `20260822`.

After changing only aggregation or plots, reuse the detailed candidate rows:

```powershell
python experiments/paper2_6_hybrid_pra/run_final_iteration.py `
  --local-files-only --reuse-detailed
```

Use `--postprocess-only` only when the frozen tensor bundles are unavailable;
that fallback is intentionally limited to confidence fields already serialized
by the preceding channel-selection run.

Outputs are written to
`docs/papers/shared/results/paper2_6_hybrid_pra/final_iteration/`. The bootstrap
rows are uncertainty estimates over the frozen identities, not cohort expansion.
## Normalized PRA efficiency

After rerunning the frozen channel-selection replay, build the normalized
root/successor analysis with:

```powershell
python -m experiments.paper2_6_hybrid_pra.normalized_efficiency
```

The analysis deduplicates selected identities across stages and keeps search
comparisons separate from the conceptual working-set fraction.
