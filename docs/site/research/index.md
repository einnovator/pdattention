# Research / Evidence

This appendix connects the product documentation to paper terminology and
checked-in artifacts. The labels below are useful for reproducing experiments;
they are intentionally absent from the ordinary deployment guide.

## Paper-to-product terminology

| Research label | Product-facing concept |
| --- | --- |
| E0 | Selected Context through an ordinary engine path |
| E1 | Typed PRA Transport and logical resource semantics |
| E2 | Native Memory consumed by model attention |
| E3 | Native Serving under scheduler ownership |

Gateway research modes map as follows:

| Research mode | Public CLI alias | Behavior |
| --- | --- | --- |
| G00 | `passthrough` | Ordinary compatible request |
| G10 | `selected-context` | Typed input rendered for an ordinary engine |
| G01 | `upgrade` | Recognize supported typed ordinary traffic |
| G11 | `typed-transport` | Preserve typed resources end to end |

Old CLI spellings remain accepted for reproducibility. New application and
deployment documentation should use the public aliases.

## Evidence sources

The public engine pages are generated from:

- `src/pra_hf/model_profiles/engine_documentation_registry.json` for product
  capability, recommendation, provenance, and missing-value status;
- `docs/papers/shared/results/pra_product_matrix_v2.json` for measured
  model/engine/profile/workload rows;
- engine-specific JSON artifacts cited on each generated page.

Regenerate and verify the pages with:

```bash
python -m experiments.paper4_5_runtime.build_technical_site
python -m experiments.paper4_5_runtime.build_technical_site --check
```

Generated claims include an evidence date and source path. A source path is a
provenance receipt, not an invitation to compare unmatched workloads.

## Experimental training

The standalone trainer remains useful for architecture and optimization
research:

```bash
python scripts/train_standalone.py --model standalone_tiny
python scripts/eval_standalone.py \
  --model standalone_tiny \
  --checkpoint out/standalone_tiny/checkpoints/best.pt
```

Pretrained-model adapter and profile commands are documented in the [CLI
reference](../cli.md). Learned routing, consumer-layer, and memory-use adapters
must retain disjoint calibration and held-out cohorts.

## Reproduction rules

1. Pin repository, model, tokenizer, engine, and hardware revisions.
2. Preserve raw per-example rows and random seeds.
3. Freeze selector output before comparing text and native representations.
4. Keep ingestion, request, and reuse costs separate.
5. Run disabled, empty, irrelevant, shuffled, and full-context controls where
   the mechanism claim requires them.
6. Report `NOT_MEASURED` rather than filling missing metrics with zero.
7. Keep smoke evidence, natural workloads, and serving evidence visibly distinct.

See [Metrics & Qualification](../metrics.md) for the public comparison contract.
