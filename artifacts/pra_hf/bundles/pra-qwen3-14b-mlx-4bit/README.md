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
| combined (n=15) | token_f1=0.2775 | token_f1=0.2775 | +0.0000 | -89.1% | -0.2% | +1.1% | 15/15 | ENGINE_QUALIFIED |

All headline rows use the same frozen selected evidence in the baseline and PRA paths. Deltas are PRA minus baseline; negative latency and context deltas are reductions.

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
| mlx | validated | QUALIFIED | NOT_MEASURED | Native Memory with BALANCED |
| hf | portable | NOT_MEASURED for the full-precision HF counterpart | NOT_MEASURED | Selected Context; exact MLX artifact only |

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

Native Memory uses the same selector output as Selected Context. Exact paired parity, visible-input reduction, active detail bytes, and latency deltas are reported above; it is recommended only where the profile and engine tables say so.

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
- Paired natural-QA evidence contains five examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 4-bit MLX model and revision; it does not transfer automatically to full-precision Hugging Face weights or another quantization.
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

- PRA commit: `69b03f8e01e9a330b2c5aeb03d4fb2373d98a146`
- Bundle build commit: `69b03f8e01e9a330b2c5aeb03d4fb2373d98a146`
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
