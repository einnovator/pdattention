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
| combined (n=15) | token_f1=0.2312 | token_f1=0.2312 | +0.0000 | -89.1% | -6.6% | +0.5% | 15/15 | ENGINE_QUALIFIED |

All headline rows use the same frozen selected evidence in the baseline and PRA paths. Deltas are PRA minus baseline; negative latency and context deltas are reductions.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA (n=15); 2026-09-01; PRA commit `4b4486a66c80d09aa7982be29812d4027c57a4e3`; artifact `qualification/qwen3_32b_mlx_profiles.json`; SHA-256 `79bccb629dd3805a7fb39c0eb109f6ac2dc53ea5dbf5c2a8aed7f9224093dd04`.

## Canonical three-condition evidence

Each table holds task, hardware, engine, model, mode, and profile fixed. Deltas are candidate minus No PRA and retain their mathematical sign.

### combined / mlx-lm / balanced

Exact identity: `mlx-community/Qwen3-32B-4bit` at `bcaaf7f538adf166c1080a2befdb4f6019f66639` on `Apple M4 Pro (Mac16,7), 48 GB`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| token_f1 | fraction | higher_is_better | 0.231164 | 0.231164 | NOT_MEASURED | +0 (+0.00%) | NOT_MEASURED |
| exact_match | fraction | higher_is_better | 0 | 0 | NOT_MEASURED | +0 | NOT_MEASURED |
| gold_answer_log_probability | log_probability | higher_is_better | -9.57946 | -9.57946 | NOT_MEASURED | +0 (-0.00%) | NOT_MEASURED |

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| visible_tokens | token | lower_is_better | 315.533 | 34.2667 | NOT_MEASURED | -281.267 (-89.14%) | NOT_MEASURED |
| selected_native_kv_tokens | token | neutral | 0 | 18001.1 | NOT_MEASURED | +18001.1 | NOT_MEASURED |

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ttft_p50_ms | ms | lower_is_better | 524.92 | 490.204 | NOT_MEASURED | -34.7164 (-6.61%) | NOT_MEASURED |
| ttft_p95_ms | ms | lower_is_better | 799.757 | 797.258 | NOT_MEASURED | -2.49929 (-0.31%) | NOT_MEASURED |
| ttft_p99_ms | ms | lower_is_better | 799.757 | 797.258 | NOT_MEASURED | -2.49929 (-0.31%) | NOT_MEASURED |
| completion_latency_mean_ms | ms | lower_is_better | 1177.04 | 1183.23 | NOT_MEASURED | +6.18768 (+0.53%) | NOT_MEASURED |

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Delta No Adaptor | Delta Bundle |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| active_detail_bytes | byte | lower_is_better | 0 | 7.37324e+07 | NOT_MEASURED | +7.37324e+07 | NOT_MEASURED |
| retained_detail_bytes | byte | lower_is_better | 0 | 7.37324e+07 | NOT_MEASURED | +7.37324e+07 | NOT_MEASURED |
| peak_memory_bytes | byte | lower_is_better | 1.91537e+10 | 1.9058e+10 | NOT_MEASURED | -9.56826e+07 (-0.50%) | NOT_MEASURED |

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
| mlx | validated | QUALIFIED | NOT_MEASURED | Native Memory with BALANCED |
| hf | portable | NOT_MEASURED for the full-precision HF counterpart | NOT_MEASURED | Selected Context; exact MLX artifact only |

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

- The learned router improves QASPER but is not uniformly positive on HotpotQA; it is opt-in rather than the bundle default.
- Paired natural-QA evidence contains five examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 4bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- Routing evidence compares a frozen generic router with a small learned router; it does not establish end-task generation quality.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

## Reproducibility

- PRA commit: `678a463f68a5ef7ae32fe700d99e23274ac61854`
- Bundle build commit: `678a463f68a5ef7ae32fe700d99e23274ac61854`
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
