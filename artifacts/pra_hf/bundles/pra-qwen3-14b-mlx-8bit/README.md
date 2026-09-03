---
library_name: pra
base_model: mlx-community/Qwen3-14B-8bit
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

# PRA Runtime Bundle for mlx-community/Qwen3-14B-8bit · MLX / 8bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-14B-8bit`
- Immutable revision: `da33cf28f06636847fd9e93e0a03d819b84cb55e`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `14B`
- Tokenizer revision: `da33cf28f06636847fd9e93e0a03d819b84cb55e`
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
| INT8 | MLX-8bit | INT8 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | qasper, hotpotqa, 2wikimultihopqa |

## Headline results

| Workload | Baseline quality | PRA quality | Quality Δ | Input/context Δ | TTFT Δ | Completion Δ | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=60) | token_f1=0.1494 | token_f1=0.1494 | +0.0000 | -91.5% | -2.4% | +0.3% | 60/60 | ENGINE_QUALIFIED |

All headline rows use the same frozen selected evidence in the baseline and PRA paths. Deltas are PRA minus baseline; negative latency and context deltas are reductions.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA; cold direct query (n=60); 2026-09-03; PRA commit `None`; artifact `qualification/matched_e0_e2_qasper.json, qualification/matched_e0_e2_hotpotqa.json, qualification/matched_e0_e2_2wikimultihopqa.json`; SHA-256 `bbb070206b08d1ce2c9c4fe4e807c9f02204e81d99d0ef4a4487f2915b62b4cd,2fad18cd234a531a40b30f5e69e6646c7020bc0f7a5a111f8bc2764f0a7fb16e,0e535af1ef7a189bf85d223f17317dfcacd9c74cf3d851b82e835c3003c8d5dd`.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | Model Name: MacBook Pro; Chip: Apple M4 Pro; Memory: 48 GB | 569.2 s | 2.828 s | 14.66 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

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

Exact identity: `mlx-community/Qwen3-14B-8bit` at `da33cf28f06636847fd9e93e0a03d819b84cb55e` on `Apple M4 Pro (Mac16,7), 48 GB`; precision `UNSPECIFIED` / `UNSPECIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.149402 | 0.149402 | +0 (+0.00%) |
| Exact Match | fraction | higher_is_better | 0 | 0 | +0 |
| Gold Answer Log Probability | log_probability | higher_is_better | -17.4831 | -17.4831 | +0 (-0.00%) |

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
| TTFT p50 (ms) | ms | lower_is_better | 335.664 | 327.575 | -8.08944 (-2.41%) |
| TTFT p95 (ms) | ms | lower_is_better | 414.035 | 393.91 | -20.1256 (-4.86%) |
| TTFT p99 (ms) | ms | lower_is_better | 430.039 | 406.098 | -23.9403 (-5.57%) |
| ITL p50 (ms) | ms | lower_is_better | 63.2739 | 63.7036 | +0.429721 (+0.68%) |
| ITL p95 (ms) | ms | lower_is_better | 64.9334 | 65.6904 | +0.75698 (+1.17%) |
| ITL p99 (ms) | ms | lower_is_better | 65.2931 | 66.3538 | +1.06065 (+1.62%) |
| Output Tokens Per Second | output_token/s | higher_is_better | 13.3873 | 13.3466 | -0.0407081 (-0.30%) |
| Completion Latency Mean (ms) | ms | lower_is_better | 1795.02 | 1800.26 | +5.23693 (+0.29%) |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 5.94412e+07 | +5.94412e+07 |
| Retained Detail Bytes | byte | lower_is_better | 0 | 5.94412e+07 | +5.94412e+07 |

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
pra inspect mlx-community/Qwen3-14B-8bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-8bit
pra evaluate mlx-community/Qwen3-14B-8bit -e mlx -D qasper -a EInnovator/pra-qwen3-14b-mlx-8bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3-14B-8bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-8bit -p balanced
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
| qasper (n=20) | Selected Context | token_f1=0.1947 | 398.6 | 271.6 ms | 1752 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=20) | Native Memory | token_f1=0.1947 | 28.05 | 264 ms | 1756 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Selected Context | token_f1=0.1608 | 415.4 | 402 ms | 1839 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Native Memory | token_f1=0.1608 | 39.05 | 386.6 ms | 1842 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Selected Context | token_f1=0.09272 | 374.9 | 387.3 ms | 1793 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Native Memory | token_f1=0.09272 | 33.45 | 382 ms | 1803 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Selected Context | token_f1=0.1494 | 396.3 | 335.7 ms | 1795 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Native Memory | token_f1=0.1494 | 33.52 | 327.6 ms | 1800 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| qasper | 370.6 | 57.91 MiB | NEEDS_RUN | 1.002x |
| hotpotqa | 376.3 | 58.80 MiB | NEEDS_RUN | 1.002x |
| 2wikimultihopqa | 341.5 | 53.36 MiB | NEEDS_RUN | 1.005x |
| combined | 362.8 | 56.69 MiB | NEEDS_RUN | 1.003x |

## Research diagnostics

No separate routing diagnostic is packaged for this bundle.

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Qwen3-14B-8bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-8bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed.
- Paired natural-QA evidence contains 20 examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 8bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- The selector-frozen natural-QA run qualifies the generic Native Memory path; an exact learned-adaptor arm still requires a separate run.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

## Reproducibility

- PRA commit: `99b47d069581ace5e7fcd6ea1bdce2d9c2b2f012`
- Bundle build commit: `99b47d069581ace5e7fcd6ea1bdce2d9c2b2f012`
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
