---
library_name: pra
base_model: mlx-community/Qwen3-4B-8bit
tags:
- pra
- progressive-retrieval-attention
- adapter
- long-context
datasets:
- 2wikimultihopqa
- combined
- hotpotqa
- qasper
license: apache-2.0
---

# PRA Runtime Bundle for mlx-community/Qwen3-4B-8bit · MLX / 8bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-4B-8bit`
- Immutable revision: `0348ad770d2ae658ca47b0579b2d2c37b20bbcac`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `4B`
- Tokenizer revision: `0348ad770d2ae658ca47b0579b2d2c37b20bbcac`
- Serving precision: `INT8` / `MLX-8bit`
- Post-training: `pretrained and post-trained`

## Recommended configuration

- Engine: **mlx**
- Recommended PRA mode: **Native Memory**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **ENGINE_QUALIFIED**
- Native Memory status: **QUALIFIED**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Precision qualification

Precision evidence is scoped to the exact model conversion, engine, mode, and profile. Qualification does not transfer automatically between BF16, INT8, INT4, or encoding-specific formats.

| Family | Encoding | Serving | Feature extraction | Adaptor parameters | Engine | Mode | Profile | Evidence | Datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 | MLX-8bit | INT8 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | NOT_MEASURED |

## Headline results

| Workload | Selected Context quality | Native Memory quality | Delta NM vs SC | Visible-context delta NM vs SC | TTFT delta NM vs SC | Completion delta NM vs SC | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=60) | token_f1=0.1315 | token_f1=0.1315 | +0.0000 | -91.5% | -9.2% | +1.9% | 60/60 | ENGINE_QUALIFIED |

All headline rows freeze the PRA-selected evidence. Deltas are Native Memory minus Selected Context; negative latency and visible-context deltas are reductions. These rows contain no ordinary No-PRA arm.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA; cold direct query (n=60); 2026-09-03; PRA commit `None`; artifact `qualification/matched_e0_e2_qasper.json, qualification/matched_e0_e2_hotpotqa.json, qualification/matched_e0_e2_2wikimultihopqa.json`; SHA-256 `a88cec4ee3398234ab9700d4ee607a3eef7e78a0d19ce14a21d646333af23ea1,023d6d8f306b98e5f3e913f934ea46e00c1ac2492ed15d95aecb9d8463876905,2a6611c03f0f9872a84b9baffeb712e7104cc1f5e02cec96078d991c030f5d13`.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | Model Name: MacBook Pro; Chip: Apple M4 Pro; Memory: 48 GB | 151.1 s | 1.014 s | 4.03 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

Runtime smoke does not establish end-task quality, Native Memory parity, routing quality, or serving economics. The coverage table below identifies the exact follow-up state.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | Mode / no adaptor | Same mode / bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Native Memory | QUALITY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Native Memory | BALANCED | NEEDS_RUN | Native Memory: MEASURED (16) | Native Memory + Bundle: NEEDS_RUN | context, quality, resources, routing, serving |
| mlx | Native Memory | ECONOMY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Native Memory | QASPER-LEARNED | NEEDS_RUN | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | NEEDS_RUN |

## Canonical staged evidence

Each table holds task, hardware, engine, model, precision, and profile fixed. Every delta names its source and target; bundle use is held orthogonal to execution depth.

### combined / mlx-lm / balanced

Exact identity: `mlx-community/Qwen3-4B-8bit` at `0348ad770d2ae658ca47b0579b2d2c37b20bbcac` on `Apple M4 Pro (Mac16,7), 48 GB`; precision `INT8` / `MLX-8bit`.

#### Quality

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.131517 | 0.131517 | NEEDS_RUN | +0 (+0.00%) | NEEDS_RUN |
| Exact Match | fraction | higher_is_better | 0 | 0 | NEEDS_RUN | +0 | NEEDS_RUN |
| Gold Answer Log Probability | log_probability | higher_is_better | -20.1859 | -20.1859 | NEEDS_RUN | +0 (-0.00%) | NEEDS_RUN |

#### Context

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 396.317 | 33.5167 | NEEDS_RUN | -362.8 (-91.54%) | NEEDS_RUN |
| Selected Native K/V Tokens | token | neutral | 0 | 362.8 | NEEDS_RUN | +362.8 | NEEDS_RUN |

#### Serving

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 112.525 | 102.135 | NEEDS_RUN | -10.3901 (-9.23%) | NEEDS_RUN |
| TTFT p95 (ms) | ms | lower_is_better | 130.114 | 115.365 | NEEDS_RUN | -14.749 (-11.34%) | NEEDS_RUN |
| TTFT p99 (ms) | ms | lower_is_better | 136.158 | 127.489 | NEEDS_RUN | -8.66854 (-6.37%) | NEEDS_RUN |
| ITL p50 (ms) | ms | lower_is_better | 19.1811 | 20.0532 | NEEDS_RUN | +0.872125 (+4.55%) | NEEDS_RUN |
| ITL p95 (ms) | ms | lower_is_better | 20.4041 | 20.9611 | NEEDS_RUN | +0.556967 (+2.73%) | NEEDS_RUN |
| ITL p99 (ms) | ms | lower_is_better | 20.6432 | 22.1909 | NEEDS_RUN | +1.5477 (+7.50%) | NEEDS_RUN |
| Output Tokens Per Second | output_token/s | higher_is_better | 43.666 | 42.8374 | NEEDS_RUN | -0.8286 (-1.90%) | NEEDS_RUN |
| Completion Latency Mean (ms) | ms | lower_is_better | 550.365 | 561.034 | NEEDS_RUN | +10.6692 (+1.94%) | NEEDS_RUN |

#### Resources

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NEEDS_RUN | +5.3497e+07 | NEEDS_RUN |
| Retained Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NEEDS_RUN | +5.3497e+07 | NEEDS_RUN |

#### Routing

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | NEEDS_RUN | +0 (+0.00%) | NEEDS_RUN |

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect mlx-community/Qwen3-4B-8bit -e mlx -a EInnovator/pra-qwen3-4b-mlx-8bit
pra evaluate mlx-community/Qwen3-4B-8bit -e mlx -D qasper -a EInnovator/pra-qwen3-4b-mlx-8bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3-4B-8bit -e mlx -a EInnovator/pra-qwen3-4b-mlx-8bit -p balanced
```

## Profiles

| Profile | Purpose | Routing | Consumer layers | Status | Recommendation |
| --- | --- | --- | --- | --- | --- |
| QUALITY | Candidate maximum-quality profile; held-out calibration is incomplete | generic cosine | all eligible | CALIBRATION_PENDING | Not promoted |
| BALANCED | Qualified default preserving the all-eligible consumer geometry | generic cosine | all eligible | QUALIFIED | Default |
| ECONOMY | Reduced-consumer candidate; the held-out quality gate has not passed | generic cosine | CALIBRATION_PENDING | CALIBRATION_PENDING | Not promoted |
| QASPER-LEARNED | Research-only learned routing profile qualified only on matched QASPER routing diagnostics | combined-router-d128 | all eligible | RESEARCH | Not promoted |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| mlx | validated | QUALIFIED | NOT_APPLICABLE | Native Memory with BALANCED |

## End-to-end qualification

| Workload | Mode | Quality | Visible tokens | TTFT p50 | Completion mean | Hardware | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qasper (n=20) | Selected Context | token_f1=0.1597 | 398.6 | 89.77 ms | 540.1 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=20) | Native Memory | token_f1=0.1597 | 28.05 | 80.41 ms | 547.9 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Selected Context | token_f1=0.1399 | 415.4 | 120.9 ms | 561.5 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Native Memory | token_f1=0.1399 | 39.05 | 114 ms | 572.9 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Selected Context | token_f1=0.09496 | 374.9 | 113.3 ms | 549.5 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Native Memory | token_f1=0.09496 | 33.45 | 109.2 ms | 562.3 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Selected Context | token_f1=0.1315 | 396.3 | 112.5 ms | 550.4 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Native Memory | token_f1=0.1315 | 33.52 | 102.1 ms | 561 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| qasper | 370.6 | 52.12 MiB | NEEDS_RUN | 1.015x |
| hotpotqa | 376.3 | 52.92 MiB | NEEDS_RUN | 1.02x |
| 2wikimultihopqa | 341.5 | 48.02 MiB | NEEDS_RUN | 1.023x |
| combined | 362.8 | 51.02 MiB | NEEDS_RUN | 1.019x |

## Research diagnostics

| Dataset | Router/profile | Metric | Value | Cohort | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| qasper | balanced | R@20% | 0.4464 | 16 | CONTROLLED |
| qasper | qasper-learned | R@20% | 0.5663 | 16 | CONTROLLED |
| hotpotqa | balanced | R@20% | 0.4071 | 16 | CONTROLLED |
| hotpotqa | qasper-learned | R@20% | 0.1804 | 16 | CONTROLLED |
| combined | balanced | R@20% | 0.4268 | 32 | CONTROLLED |
| combined | qasper-learned | R@20% | 0.3733 | 32 | CONTROLLED |

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Qwen3-4B-8bit -e mlx -a EInnovator/pra-qwen3-4b-mlx-8bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- The learned router improves QASPER but is not uniformly positive on HotpotQA; it is opt-in rather than the bundle default.
- Paired natural-QA evidence contains 20 examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 8bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- Routing evidence compares a frozen generic router with a small learned router; it does not establish end-task generation quality.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

- Datasets: `QASPER and HotpotQA`
- Train Examples: `48`
- Validation Examples: `16`
- Held Out Test Examples: `32`
- Seeds: `[11, 23, 37, 53, 71]`
- Selection: `maximum combined validation AUC0-30`
- Method: `multi-positive softmax`
- Parameter Count: `655360`
- Base Revision: `0348ad770d2ae658ca47b0579b2d2c37b20bbcac`

## Reproducibility

- PRA commit: `81f42d69936bf50eb6fe11a0f7477b415bbf250d`
- Bundle build commit: `81f42d69936bf50eb6fe11a0f7477b415bbf250d`
- Bundle schema: `2`
- PRA package: `0.2.0rc1`
- Component fingerprints and file checksums are recorded in `bundle.yaml`.

## Community/support

- [PRA documentation](https://einnovator.github.io/pdattention/)
- [Source repository](https://github.com/einnovator/pdattention)
- [Issues](https://github.com/einnovator/pdattention/issues)
- [Contribution guide](https://github.com/einnovator/pdattention/blob/main/CONTRIBUTING.md)
- [Canonical PRA Bundles Collection](https://huggingface.co/collections/EInnovator/pra-bundles-6a971e52093232f858e660f6)
- [EInnovator on Hugging Face](https://huggingface.co/EInnovator)
