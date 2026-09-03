---
library_name: pra
base_model: mlx-community/Qwen3-14B-4bit
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

# PRA Runtime Bundle for mlx-community/Qwen3-14B-4bit · MLX / 4bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-14B-4bit`
- Immutable revision: `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `14B`
- Tokenizer revision: `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`
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
| INT4 | MLX-4bit | INT4 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | NOT_MEASURED |

## Headline results

| Workload | Selected Context quality | Native Memory quality | Delta NM vs SC | Visible-context delta NM vs SC | TTFT delta NM vs SC | Completion delta NM vs SC | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=15) | token_f1=0.2775 | token_f1=0.2775 | +0.0000 | -89.1% | -0.2% | +1.1% | 15/15 | ENGINE_QUALIFIED |

All headline rows freeze the PRA-selected evidence. Deltas are Native Memory minus Selected Context; negative latency and visible-context deltas are reductions. These rows contain no ordinary No-PRA arm.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA (n=15); 2026-09-01; PRA commit `4b4486a66c80d09aa7982be29812d4027c57a4e3`; artifact `qualification/qwen3_14b_mlx_profiles.json`; SHA-256 `4e6d844bd34a22fe3f15ce205e4928070638f39ae73a144f8320b373b83ec7fd`.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | Mode / no adaptor | Same mode / bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Native Memory | QUALITY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Native Memory | BALANCED | NEEDS_RUN | Native Memory: MEASURED (16) | Native Memory + Bundle: NEEDS_RUN | context, quality, resources, serving |
| mlx | Native Memory | ECONOMY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Native Memory | QASPER-LEARNED | NEEDS_RUN | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | NEEDS_RUN |

## Canonical staged evidence

Each table holds task, hardware, engine, model, precision, and profile fixed. Every delta names its source and target; bundle use is held orthogonal to execution depth.

### combined / mlx-lm / balanced

Exact identity: `mlx-community/Qwen3-14B-4bit` at `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` on `Apple M4 Pro (Mac16,7), 48 GB`; precision `INT4` / `MLX-4bit`.

#### Quality

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.27746 | 0.27746 | NEEDS_RUN | +0 (+0.00%) | NEEDS_RUN |
| Exact Match | fraction | higher_is_better | 0 | 0 | NEEDS_RUN | +0 | NEEDS_RUN |
| Gold Answer Log Probability | log_probability | higher_is_better | -9.97839 | -9.97839 | NEEDS_RUN | +0 (-0.00%) | NEEDS_RUN |

#### Context

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 315.533 | 34.2667 | NEEDS_RUN | -281.267 (-89.14%) | NEEDS_RUN |
| Selected Native K/V Tokens | token | neutral | 0 | 11250.7 | NEEDS_RUN | +11250.7 | NEEDS_RUN |

#### Serving

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 210.724 | 210.373 | NEEDS_RUN | -0.351416 (-0.17%) | NEEDS_RUN |
| TTFT p95 (ms) | ms | lower_is_better | 339.229 | 338.285 | NEEDS_RUN | -0.94425 (-0.28%) | NEEDS_RUN |
| TTFT p99 (ms) | ms | lower_is_better | 339.229 | 338.285 | NEEDS_RUN | -0.94425 (-0.28%) | NEEDS_RUN |
| ITL p50 (ms) | ms | lower_is_better | 34.3822 | 35.2247 | NEEDS_RUN | +0.842512 (+2.45%) | NEEDS_RUN |
| ITL p95 (ms) | ms | lower_is_better | 34.6274 | 35.4551 | NEEDS_RUN | +0.827768 (+2.39%) | NEEDS_RUN |
| ITL p99 (ms) | ms | lower_is_better | 34.6274 | 35.4551 | NEEDS_RUN | +0.827768 (+2.39%) | NEEDS_RUN |
| Output Tokens Per Second | output_token/s | higher_is_better | 29.0911 | 28.4113 | NEEDS_RUN | -0.679798 (-2.34%) | NEEDS_RUN |
| Completion Latency Mean (ms) | ms | lower_is_better | 507.829 | 513.248 | NEEDS_RUN | +5.41944 (+1.07%) | NEEDS_RUN |

#### Resources

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 4.60827e+07 | NEEDS_RUN | +4.60827e+07 | NEEDS_RUN |
| Retained Detail Bytes | byte | lower_is_better | 0 | 4.60827e+07 | NEEDS_RUN | +4.60827e+07 | NEEDS_RUN |
| Peak Memory Bytes | byte | lower_is_better | 8.85034e+09 | 8.79468e+09 | NEEDS_RUN | -5.56564e+07 (-0.63%) | NEEDS_RUN |

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect mlx-community/Qwen3-14B-4bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-4bit
pra evaluate mlx-community/Qwen3-14B-4bit -e mlx -D qasper -a EInnovator/pra-qwen3-14b-mlx-4bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3-14B-4bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-4bit -p balanced
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
| 2wikimultihopqa (n=5) | Selected Context | token_f1=0.2833 | 351.2 | 209.9 ms | 499.3 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=5) | Native Memory | token_f1=0.2833 | 31.6 | 210.4 ms | 505.3 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=5) | Selected Context | token_f1=0.4419 | 262 | 333.7 ms | 574.8 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=5) | Native Memory | token_f1=0.4419 | 43.4 | 333.9 ms | 580 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=5) | Selected Context | token_f1=0.1071 | 333.4 | 209.1 ms | 449.4 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=5) | Native Memory | token_f1=0.1071 | 27.8 | 208.3 ms | 454.5 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=15) | Selected Context | token_f1=0.2775 | 315.5 | 210.7 ms | 507.8 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=15) | Native Memory | token_f1=0.2775 | 34.27 | 210.4 ms | 513.2 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 1.278e+04 | 49.94 MiB | 8.18 GiB | 1.012x |
| hotpotqa | 8744 | 34.16 MiB | 8.15 GiB | 1.009x |
| qasper | 1.222e+04 | 47.75 MiB | 8.19 GiB | 1.011x |
| combined | 1.125e+04 | 43.95 MiB | 8.19 GiB | 1.011x |

## Research diagnostics

| Dataset | Router/profile | Metric | Value | Cohort | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| qasper | balanced | R@20% | 0.3182 | 16 | CONTROLLED |
| qasper | qasper-learned | R@20% | 0.6787 | 16 | CONTROLLED |
| hotpotqa | balanced | R@20% | 0.4942 | 16 | CONTROLLED |
| hotpotqa | qasper-learned | R@20% | 0.3144 | 16 | CONTROLLED |
| combined | balanced | R@20% | 0.4062 | 32 | CONTROLLED |
| combined | qasper-learned | R@20% | 0.4966 | 32 | CONTROLLED |

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Qwen3-14B-4bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-4bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- The learned router improves QASPER but is not uniformly positive on HotpotQA; it is opt-in rather than the bundle default.
- Paired natural-QA evidence contains 5 examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 4bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
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
- Parameter Count: `1310720`
- Base Revision: `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`

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
