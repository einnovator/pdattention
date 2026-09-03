---
library_name: pra
base_model: mlx-community/Qwen3-8B-6bit
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

# PRA Runtime Bundle for mlx-community/Qwen3-8B-6bit · MLX / 6bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-8B-6bit`
- Immutable revision: `35a99712f90d6c2c9a2407a3857e104a46edd9e6`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `8B`
- Tokenizer revision: `35a99712f90d6c2c9a2407a3857e104a46edd9e6`
- Serving precision: `INT6` / `MLX-6bit`
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
| INT6 | MLX-6bit | INT6 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | qasper, hotpotqa, 2wikimultihopqa |

## Headline results

| Workload | Baseline quality | PRA quality | Quality Δ | Input/context Δ | TTFT Δ | Completion Δ | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=60) | token_f1=0.1582 | token_f1=0.1582 | +0.0000 | -91.5% | +0.7% | +1.4% | 60/60 | ENGINE_QUALIFIED |

All headline rows use the same frozen selected evidence in the baseline and PRA paths. Deltas are PRA minus baseline; negative latency and context deltas are reductions.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA; cold direct query (n=60); 2026-09-03; PRA commit `None`; artifact `qualification/matched_e0_e2_qasper.json, qualification/matched_e0_e2_hotpotqa.json, qualification/matched_e0_e2_2wikimultihopqa.json`; SHA-256 `301034d1049a1bf5817dffd0e8f61dc17e2b134e6cf1fa94eb420649b1a12d6d,fba567547ee2a4bdd99a9c38d92cd080f59de01a5a89dc45a0e15c52949ec715,ef1be51ef852b8c93f6751fe0e0be236d29bae8643651cb8b78f3e7c62a2c22f`.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | Model Name: MacBook Pro; Chip: Apple M4 Pro; Memory: 48 GB | 202.8 s | 1.004 s | 6.24 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

Runtime smoke does not establish end-task quality, Native Memory parity, routing quality, or serving economics. The coverage table below identifies the exact follow-up state.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Native Memory | QUALITY | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NO_QUALIFIED_ADAPTER | context, quality, resources, routing, serving |
| mlx | Native Memory | ECONOMY | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING |

## Canonical three-condition evidence

Each table holds task, hardware, engine, model, mode, and profile fixed. Deltas are candidate minus No PRA and retain their mathematical sign.

### combined / mlx-lm / balanced

Exact identity: `mlx-community/Qwen3-8B-6bit` at `35a99712f90d6c2c9a2407a3857e104a46edd9e6` on `Apple M4 Pro (Mac16,7), 48 GB`; precision `UNSPECIFIED` / `UNSPECIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.158232 | 0.158232 | +0 (+0.00%) |
| Exact Match | fraction | higher_is_better | 0 | 0 | +0 |
| Gold Answer Log Probability | log_probability | higher_is_better | -18.8568 | -18.8568 | +0 (-0.00%) |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 396.317 | 33.5167 | -362.8 (-91.54%) |
| Selected Native K/V Tokens | token | neutral | 0 | 362.8 | +362.8 |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 174.612 | 175.757 | +1.14525 (+0.66%) |
| TTFT p95 (ms) | ms | lower_is_better | 224.646 | 203.595 | -21.0517 (-9.37%) |
| TTFT p99 (ms) | ms | lower_is_better | 227.007 | 219.186 | -7.82108 (-3.45%) |
| ITL p50 (ms) | ms | lower_is_better | 27.1323 | 28.1637 | +1.03139 (+3.80%) |
| ITL p95 (ms) | ms | lower_is_better | 28.9412 | 29.379 | +0.437819 (+1.51%) |
| ITL p99 (ms) | ms | lower_is_better | 29.4701 | 31.0573 | +1.58718 (+5.39%) |
| Output Tokens Per Second | output_token/s | higher_is_better | 29.8239 | 29.3964 | -0.427475 (-1.43%) |
| Completion Latency Mean (ms) | ms | lower_is_better | 806.747 | 818.315 | +11.5686 (+1.43%) |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | +5.3497e+07 |
| Retained Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | +5.3497e+07 |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | +0 (+0.00%) |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect mlx-community/Qwen3-8B-6bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-6bit
pra evaluate mlx-community/Qwen3-8B-6bit -e mlx -D qasper -a EInnovator/pra-qwen3-8b-mlx-6bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3-8B-6bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-6bit -p balanced
```

## Profiles

| Profile | Purpose | Routing | Consumer layers | Status | Recommendation |
| --- | --- | --- | --- | --- | --- |
| QUALITY | Candidate maximum-quality profile; held-out calibration is incomplete | generic cosine | all eligible | CALIBRATION_PENDING | Not promoted |
| BALANCED | Qualified default preserving the all-eligible consumer geometry | generic cosine | all eligible | QUALIFIED | Default |
| ECONOMY | Reduced-consumer candidate; the held-out quality gate has not passed | generic cosine | CALIBRATION_PENDING | CALIBRATION_PENDING | Not promoted |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| mlx | validated | QUALIFIED | NOT_APPLICABLE | Native Memory with BALANCED |

## End-to-end qualification

| Workload | Mode | Quality | Visible tokens | TTFT p50 | Completion mean | Hardware | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qasper (n=20) | Selected Context | token_f1=0.1836 | 398.6 | 133.6 ms | 771.8 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=20) | Native Memory | token_f1=0.1836 | 28.05 | 128.4 ms | 788.3 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Selected Context | token_f1=0.1791 | 415.4 | 206.8 ms | 823.8 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Native Memory | token_f1=0.1791 | 39.05 | 194.7 ms | 837.5 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Selected Context | token_f1=0.112 | 374.9 | 195.1 ms | 824.7 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Native Memory | token_f1=0.112 | 33.45 | 193 ms | 829.1 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Selected Context | token_f1=0.1582 | 396.3 | 174.6 ms | 806.7 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Native Memory | token_f1=0.1582 | 33.52 | 175.8 ms | 818.3 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| qasper | 370.6 | 52.12 MiB | NEEDS_RUN | 1.021x |
| hotpotqa | 376.3 | 52.92 MiB | NEEDS_RUN | 1.017x |
| 2wikimultihopqa | 341.5 | 48.02 MiB | NEEDS_RUN | 1.005x |
| combined | 362.8 | 51.02 MiB | NEEDS_RUN | 1.014x |

## Research diagnostics

No separate routing diagnostic is packaged for this bundle.

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Qwen3-8B-6bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-6bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed.
- Paired natural-QA evidence contains 20 examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 6bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- The selector-frozen natural-QA run qualifies the generic Native Memory path; an exact learned-adaptor arm still requires a separate run.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

## Reproducibility

- PRA commit: `904e04070ce0309f45a161650e9907c66be5822e`
- Bundle build commit: `904e04070ce0309f45a161650e9907c66be5822e`
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
