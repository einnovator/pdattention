---
library_name: pra
base_model: mlx-community/Qwen3-32B-4bit
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

# PRA Runtime Bundle for mlx-community/Qwen3-32B-4bit · MLX / 4bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-32B-4bit`
- Immutable revision: `bcaaf7f538adf166c1080a2befdb4f6019f66639`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `32B`
- Tokenizer revision: `bcaaf7f538adf166c1080a2befdb4f6019f66639`
- Serving precision: `INT4` / `MLX-4bit`
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
| INT4 | MLX-4bit | INT4 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | qasper, hotpotqa, 2wikimultihopqa |

## Headline results

| Workload | Baseline quality | PRA quality | Quality Δ | Input/context Δ | TTFT Δ | Completion Δ | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=15) | token_f1=0.2312 | token_f1=0.2312 | +0.0000 | -89.1% | -6.6% | +0.5% | 15/15 | ENGINE_QUALIFIED |

All headline rows use the same frozen selected evidence in the baseline and PRA paths. Deltas are PRA minus baseline; negative latency and context deltas are reductions.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA (n=15); 2026-09-01; PRA commit `4b4486a66c80d09aa7982be29812d4027c57a4e3`; artifact `qualification/qwen3_32b_mlx_profiles.json`; SHA-256 `79bccb629dd3805a7fb39c0eb109f6ac2dc53ea5dbf5c2a8aed7f9224093dd04`.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Native Memory | QUALITY | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Native Memory | BALANCED | MEASURED (16) | MEASURED (16) | NEEDS_RUN | context, quality, resources, serving |
| mlx | Native Memory | ECONOMY | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING |

## Canonical three-condition evidence

Each table holds task, hardware, engine, model, mode, and profile fixed. Deltas are candidate minus No PRA and retain their mathematical sign.

### combined / mlx-lm / balanced

Exact identity: `mlx-community/Qwen3-32B-4bit` at `bcaaf7f538adf166c1080a2befdb4f6019f66639` on `Apple M4 Pro (Mac16,7), 48 GB`; precision `UNSPECIFIED` / `UNSPECIFIED`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.231164 | 0.231164 | +0 (+0.00%) |
| Exact Match | fraction | higher_is_better | 0 | 0 | +0 |
| Gold Answer Log Probability | log_probability | higher_is_better | -9.57946 | -9.57946 | +0 (-0.00%) |

PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 315.533 | 34.2667 | -281.267 (-89.14%) |
| Selected Native K/V Tokens | token | neutral | 0 | 18001.1 | +18001.1 |

PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 524.92 | 490.204 | -34.7164 (-6.61%) |
| TTFT p95 (ms) | ms | lower_is_better | 799.757 | 797.258 | -2.49929 (-0.31%) |
| TTFT p99 (ms) | ms | lower_is_better | 799.757 | 797.258 | -2.49929 (-0.31%) |
| ITL p50 (ms) | ms | lower_is_better | 77.7091 | 79.0556 | +1.34651 (+1.73%) |
| ITL p95 (ms) | ms | lower_is_better | 78.2616 | 80.3438 | +2.08218 (+2.66%) |
| ITL p99 (ms) | ms | lower_is_better | 78.2616 | 80.3438 | +2.08218 (+2.66%) |
| Output Tokens Per Second | output_token/s | higher_is_better | 12.8734 | 12.6266 | -0.246821 (-1.92%) |
| Completion Latency Mean (ms) | ms | lower_is_better | 1177.04 | 1183.23 | +6.18768 (+0.53%) |

PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 7.37324e+07 | +7.37324e+07 |
| Retained Detail Bytes | byte | lower_is_better | 0 | 7.37324e+07 | +7.37324e+07 |
| Peak Memory Bytes | byte | lower_is_better | 1.91537e+10 | 1.9058e+10 | -9.56826e+07 (-0.50%) |

PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect mlx-community/Qwen3-32B-4bit -e mlx -a EInnovator/pra-qwen3-32b-mlx-4bit
pra evaluate mlx-community/Qwen3-32B-4bit -e mlx -D qasper -a EInnovator/pra-qwen3-32b-mlx-4bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3-32B-4bit -e mlx -a EInnovator/pra-qwen3-32b-mlx-4bit -p balanced
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
| 2wikimultihopqa (n=5) | Selected Context | token_f1=0.1778 | 351.2 | 494 ms | 1155 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=5) | Native Memory | token_f1=0.1778 | 31.6 | 490.2 ms | 1164 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=5) | Selected Context | token_f1=0.3657 | 262 | 791.6 ms | 1334 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=5) | Native Memory | token_f1=0.3657 | 43.4 | 791.7 ms | 1343 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=5) | Selected Context | token_f1=0.15 | 333.4 | 489.1 ms | 1042 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=5) | Native Memory | token_f1=0.15 | 27.8 | 487.7 ms | 1043 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=15) | Selected Context | token_f1=0.2312 | 315.5 | 524.9 ms | 1177 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=15) | Native Memory | token_f1=0.2312 | 34.27 | 490.2 ms | 1183 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 2.045e+04 | 79.90 MiB | 17.74 GiB | 1.007x |
| hotpotqa | 1.399e+04 | 54.65 MiB | 17.71 GiB | 1.007x |
| qasper | 1.956e+04 | 76.40 MiB | 17.75 GiB | 1x |
| combined | 1.8e+04 | 70.32 MiB | 17.75 GiB | 1.005x |

## Research diagnostics

No separate routing diagnostic is packaged for this bundle.

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Qwen3-32B-4bit -e mlx -a EInnovator/pra-qwen3-32b-mlx-4bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed.
- Paired natural-QA evidence contains five examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 4bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- The selector-frozen natural-QA run qualifies the generic Native Memory path; an exact learned-adaptor arm still requires a separate run.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

## Reproducibility

- PRA commit: `430292dc5b8b57a9d99158bf945a0a118b2c50c1`
- Bundle build commit: `430292dc5b8b57a9d99158bf945a0a118b2c50c1`
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
