---
library_name: pra
base_model: mlx-community/Qwen3-8B-4bit
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

# PRA Runtime Bundle for mlx-community/Qwen3-8B-4bit · MLX / 4bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-8B-4bit`
- Immutable revision: `545dc4251c05440727734bcd94334791f6ab0192`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `8B`
- Tokenizer revision: `545dc4251c05440727734bcd94334791f6ab0192`
- Post-training: `pretrained and post-trained`

## Recommended configuration

- Engine: **mlx**
- Recommended PRA mode: **Native Memory**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **ENGINE_QUALIFIED**
- Native Memory status: **QUALIFIED**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Headline results

| Workload | Baseline quality | PRA quality | Quality Δ | Input/context Δ | TTFT Δ | Completion Δ | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=15) | token_f1=0.2365 | token_f1=0.2365 | +0.0000 | -89.1% | -0.6% | +3.0% | 15/15 | ENGINE_QUALIFIED |

All headline rows use the same frozen selected evidence in the baseline and PRA paths. Deltas are PRA minus baseline; negative latency and context deltas are reductions.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA (n=15); 2026-09-01; PRA commit `4b4486a66c80d09aa7982be29812d4027c57a4e3`; artifact `qualification/qwen3_8b_mlx_profiles.json`; SHA-256 `43a316c4ecca24420d5d5bc839d2b7a94d9a1074cbe5449092f23a6ba0acf47f`.

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

Exact identity: `mlx-community/Qwen3-8B-4bit` at `545dc4251c05440727734bcd94334791f6ab0192` on `Apple M4 Pro (Mac16,7), 48 GB`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.236455 | 0.236455 | +0 (+0.00%) |
| Exact Match | fraction | higher_is_better | 0 | 0 | +0 |
| Gold Answer Log Probability | log_probability | higher_is_better | -12.2676 | -12.2676 | +0 (-0.00%) |

PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 315.533 | 34.2667 | -281.267 (-89.14%) |
| Selected Native K/V Tokens | token | neutral | 0 | 10125.6 | +10125.6 |

PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 115.775 | 115.061 | -0.713958 (-0.62%) |
| TTFT p95 (ms) | ms | lower_is_better | 183.02 | 231.031 | +48.0105 (+26.23%) |
| TTFT p99 (ms) | ms | lower_is_better | 183.02 | 231.031 | +48.0105 (+26.23%) |
| ITL p50 (ms) | ms | lower_is_better | 18.787 | 19.4877 | +0.700714 (+3.73%) |
| ITL p95 (ms) | ms | lower_is_better | 18.9825 | 20.5185 | +1.53595 (+8.09%) |
| ITL p99 (ms) | ms | lower_is_better | 18.9825 | 20.5185 | +1.53595 (+8.09%) |
| Output Tokens Per Second | output_token/s | higher_is_better | 53.3045 | 51.1062 | -2.19834 (-4.12%) |
| Completion Latency Mean (ms) | ms | lower_is_better | 275.946 | 284.088 | +8.14217 (+2.95%) |

PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 4.14745e+07 | +4.14745e+07 |
| Retained Detail Bytes | byte | lower_is_better | 0 | 4.14745e+07 | +4.14745e+07 |
| Peak Memory Bytes | byte | lower_is_better | 5.18009e+09 | 5.13407e+09 | -4.60226e+07 (-0.89%) |

PRA - Adaptor Bundle: `NEEDS_RUN` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect mlx-community/Qwen3-8B-4bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-4bit
pra evaluate mlx-community/Qwen3-8B-4bit -e mlx -D qasper -a EInnovator/pra-qwen3-8b-mlx-4bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3-8B-4bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-4bit -p balanced
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
| 2wikimultihopqa (n=5) | Selected Context | token_f1=0.3578 | 351.2 | 114.2 ms | 271.9 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=5) | Native Memory | token_f1=0.3578 | 31.6 | 113.8 ms | 278.3 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=5) | Selected Context | token_f1=0.3016 | 262 | 180.3 ms | 311.6 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=5) | Native Memory | token_f1=0.3016 | 43.4 | 179 ms | 325 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=5) | Selected Context | token_f1=0.05 | 333.4 | 114.1 ms | 244.4 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=5) | Native Memory | token_f1=0.05 | 27.8 | 112.6 ms | 249 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=15) | Selected Context | token_f1=0.2365 | 315.5 | 115.8 ms | 275.9 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=15) | Native Memory | token_f1=0.2365 | 34.27 | 115.1 ms | 284.1 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 1.151e+04 | 44.94 MiB | 4.77 GiB | 1.023x |
| hotpotqa | 7870 | 30.74 MiB | 4.74 GiB | 1.043x |
| qasper | 1.1e+04 | 42.98 MiB | 4.78 GiB | 1.019x |
| combined | 1.013e+04 | 39.55 MiB | 4.78 GiB | 1.03x |

## Research diagnostics

No separate routing diagnostic is packaged for this bundle.

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Qwen3-8B-4bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-4bit -D qasper -o .pra/runs/qasper
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
