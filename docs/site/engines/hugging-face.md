# Hugging Face

_Evidence current through 2026-09-01; generated from checked-in registries._

## What this engine is for

Reference integration for model development, correctness checks, and portable Python experiments.

## Best PRA deployment today

Use Selected Context for ordinary pipelines. Use Native Memory when validating model-level PRA behavior or a measured model profile.

## Supported PRA capabilities

| Capability | Status |
| --- | --- |
| Selected Context | Validated |
| Typed PRA Transport | Validated |
| Native Memory | Validated |
| Native Serving | Not measured |

## Architecture

The reference runtime supports typed resources and layer-specific native memory, but it is not a scheduler-managed serving system.

```text
application -> typed context -> PRA route/select/materialize
            -> Hugging Face -> generated response
```

## Requirements and tested boundary

- Python 3.10 or newer
- PyTorch and Transformers
- A model adapter supported by the runtime

## Quickstart

```bash
pra runtime doctor -e hf
pra runtime inspect Qwen/Qwen3-1.7B -e hf
pra runtime serve Qwen/Qwen3-1.7B -e hf --storage balanced
```

Inspect the capability report before relying on anything beyond Selected
Context. An unavailable capability must fail explicitly or fall back only
when the request permits that fallback.

## Measured results

| Metric | Value | Evidence |
| --- | --- | --- |
| Natural serving tails | NOT_MEASURED | Not measured |
| Reference mechanism | Validated across Qwen, Llama, and Gemma adapters | Model-backed |

## Metrics and explicit gaps

- **Natural serving tails:** NOT_MEASURED  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Not measured.
- **Reference mechanism:** Validated across Qwen, Llama, and Gemma adapters  Provenance: `docs/papers/shared/results/pra_product_matrix_v2.json`; evidence: Model-backed.

Unknown metrics remain `NOT_MEASURED`; this page does not convert them to
zero or infer economic benefit from token reduction alone.

## When to choose Selected Context

Choose it for maximum engine portability, new models, and workloads without repeated immutable evidence.

## When Native Memory may help

Consider it for repeated selected resources after model-specific parity and quality gates pass.

## Limitations

- No production scheduler ownership
- Profile qualification remains model and workload specific

## Research evidence

Current public evidence label: **Model-backed**. See the [research appendix](../research/index.md) for paper-level names and the [qualification contract](../metrics.md) before comparing engines.

## Troubleshooting

- Run pra doctor in the same Python environment as the CLI
- Use pra model validate before enabling a native profile

## Production recommendation

Start with Selected Context and promote Native Memory only from a measured bundle.
