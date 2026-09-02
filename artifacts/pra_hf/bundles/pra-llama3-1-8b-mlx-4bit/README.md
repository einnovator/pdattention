---
library_name: pra
base_model: mlx-community/Llama-3.1-8B-Instruct-4bit
tags:
- pra
- progressive-retrieval-attention
- adapter
- long-context
datasets:
- combined
- hotpotqa
- qasper
license: llama3.1
---

# PRA Runtime Bundle for mlx-community/Llama-3.1-8B-Instruct-4bit · MLX / 4bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Llama-3.1-8B-Instruct-4bit`
- Immutable revision: `90215b22ec18e72f623dde2ea7af4097025160e2`
- Architecture: `LlamaForCausalLM`
- Parameters: `8B`
- Tokenizer revision: `90215b22ec18e72f623dde2ea7af4097025160e2`

## Recommended configuration

- Engine: **mlx**
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
pra inspect mlx-community/Llama-3.1-8B-Instruct-4bit -e mlx -a EInnovator/pra-llama3-1-8b-mlx-4bit
pra evaluate mlx-community/Llama-3.1-8B-Instruct-4bit -e mlx -D qasper -a EInnovator/pra-llama3-1-8b-mlx-4bit
pra recommend .pra/runs/latest
pra serve mlx-community/Llama-3.1-8B-Instruct-4bit -e mlx -a EInnovator/pra-llama3-1-8b-mlx-4bit -p balanced
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
| mlx | validated | AVAILABLE | NOT_MEASURED | Selected Context with BALANCED |
| hf | portable | NOT_MEASURED for the full-precision HF counterpart | NOT_MEASURED | Selected Context; exact MLX artifact only |

## End-to-end qualification

What remains to be measured: paired end-task quality for this exact bundle identity.

## Native Memory qualification

What remains to be measured: paired Selected Context versus Native Memory quality and serving economics.

## Research diagnostics

| Dataset | Router/profile | Metric | Value | Cohort | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| qasper | balanced | R@20% | 0.3182 | 16 | CONTROLLED |
| qasper | qasper-learned | R@20% | 0.4683 | 16 | CONTROLLED |
| hotpotqa | balanced | R@20% | 0.6158 | 16 | CONTROLLED |
| hotpotqa | qasper-learned | R@20% | 0.4205 | 16 | CONTROLLED |
| combined | balanced | R@20% | 0.467 | 32 | CONTROLLED |
| combined | qasper-learned | R@20% | 0.4444 | 32 | CONTROLLED |

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Llama-3.1-8B-Instruct-4bit -e mlx -a EInnovator/pra-llama3-1-8b-mlx-4bit -D qasper -o .pra/runs/qasper
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
- Parameter Count: `1048576`
- Base Revision: `90215b22ec18e72f623dde2ea7af4097025160e2`

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
