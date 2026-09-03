---
library_name: pra
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
tags:
- pra
- progressive-retrieval-attention
- adapter
- long-context
datasets:
- combined
- hotpotqa
- qasper
license: apache-2.0
---

# PRA Runtime Bundle for Qwen/Qwen2.5-Coder-1.5B-Instruct · HF / bfloat16

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Immutable revision: `2e1fd397ee46e1388853d2af2c993145b0f1098a`
- Architecture: `Qwen2ForCausalLM`
- Parameters: `1.5B`
- Tokenizer revision: `2e1fd397ee46e1388853d2af2c993145b0f1098a`
- Post-training: `code and instruction tuning`

## Recommended configuration

- Engine: **hf**
- Recommended PRA mode: **Selected Context**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **CONTROLLED**
- Native Memory status: **AVAILABLE**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Headline results

No paired end-task headline is available for this exact model, revision, quantization, engine, profile, and execution mode. Routing diagnostics below must not be interpreted as application quality.

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect Qwen/Qwen2.5-Coder-1.5B-Instruct -e hf -a EInnovator/pra-qwen2-5-coder-1-5b-instruct
pra evaluate Qwen/Qwen2.5-Coder-1.5B-Instruct -e hf -D qasper -a EInnovator/pra-qwen2-5-coder-1-5b-instruct
pra recommend .pra/runs/latest
pra serve Qwen/Qwen2.5-Coder-1.5B-Instruct -e hf -a EInnovator/pra-qwen2-5-coder-1-5b-instruct -p balanced
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
| hf | validated | AVAILABLE | NOT_MEASURED | Selected Context with BALANCED |
| mlx | portable | NOT_MEASURED for a quantized MLX derivative | NOT_MEASURED | Selected Context; exact HF artifact only |

## End-to-end qualification

What remains to be measured: paired end-task quality for this exact bundle identity.

## Native Memory qualification

What remains to be measured: paired Selected Context versus Native Memory quality and serving economics.

## Research diagnostics

| Dataset | Router/profile | Metric | Value | Cohort | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| qasper | balanced | R@20% | 0.173 | 16 | CONTROLLED |
| qasper | qasper-learned | R@20% | 0.3938 | 16 | CONTROLLED |
| hotpotqa | balanced | R@20% | 0.3982 | 16 | CONTROLLED |
| hotpotqa | qasper-learned | R@20% | 0.2153 | 16 | CONTROLLED |
| combined | balanced | R@20% | 0.2856 | 32 | CONTROLLED |
| combined | qasper-learned | R@20% | 0.3045 | 32 | CONTROLLED |

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate Qwen/Qwen2.5-Coder-1.5B-Instruct -e hf -a EInnovator/pra-qwen2-5-coder-1-5b-instruct -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- The learned router improves QASPER but is not uniformly positive on HotpotQA; it is opt-in rather than the bundle default.
- The held-out routing diagnostic contains 16 examples per dataset and supports controlled routing claims only.
- Native consumer-layer profiles and end-task generation remain uncalibrated for this exact identity.
- The qualification identity is the exact bfloat16 HF model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- Routing evidence compares a frozen generic router with a small learned router; it does not establish end-task generation quality.
- This qualification uses natural-document routing; code retrieval and coding-agent task success remain separately unmeasured.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

- Datasets: `QASPER and HotpotQA`
- Train Examples: `48`
- Validation Examples: `16`
- Held Out Test Examples: `32`
- Seeds: `[11, 23, 37, 53, 71]`
- Selection: `maximum combined validation AUC0-30`
- Method: `multi-positive softmax`
- Parameter Count: `393216`
- Base Revision: `2e1fd397ee46e1388853d2af2c993145b0f1098a`

## Reproducibility

- PRA commit: `de12658ff994e6215c07eb34cda32c566ee344d8`
- Bundle build commit: `de12658ff994e6215c07eb34cda32c566ee344d8`
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
