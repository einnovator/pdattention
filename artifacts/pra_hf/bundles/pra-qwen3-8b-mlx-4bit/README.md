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
| mlx | validated | QUALIFIED | NOT_MEASURED | Native Memory with BALANCED |
| hf | portable | NOT_MEASURED for the full-precision HF counterpart | NOT_MEASURED | Selected Context; exact MLX artifact only |

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

Native Memory uses the same selector output as Selected Context. Exact paired parity, visible-input reduction, active detail bytes, and latency deltas are reported above; it is recommended only where the profile and engine tables say so.

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

- The learned router improves QASPER but is not uniformly positive on HotpotQA; it is opt-in rather than the bundle default.
- Paired natural-QA evidence contains five examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 4-bit MLX model and revision; it does not transfer automatically to full-precision Hugging Face weights or another quantization.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

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
