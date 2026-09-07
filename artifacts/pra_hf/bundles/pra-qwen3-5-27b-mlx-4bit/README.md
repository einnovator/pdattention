---
library_name: pra
base_model: mlx-community/Qwen3.5-27B-4bit
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

# PRA Runtime Bundle for mlx-community/Qwen3.5-27B-4bit · MLX / 4bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3.5-27B-4bit`
- Immutable revision: `45797d2985a12c55e6473686e9ea91b95e959553`
- Architecture: `Qwen3_5ForConditionalGeneration`
- Parameters: `27B`
- Tokenizer revision: `45797d2985a12c55e6473686e9ea91b95e959553`
- Serving precision: `INT4` / `MLX-4bit`
- Post-training: `pretrained and post-trained`

## Recommended configuration

- Engine: **mlx**
- Recommended PRA mode: **Selected Context**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **CONTROLLED**
- Native Memory status: **UNAVAILABLE_HYBRID_STATE**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Precision qualification

Precision evidence is scoped to the exact model conversion, engine, mode, and profile. Qualification does not transfer automatically between BF16, INT8, INT4, or encoding-specific formats.

| Family | Encoding | Serving | Feature extraction | Adaptor parameters | Engine | Mode | Profile | Evidence | Datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT4 | MLX-4bit | INT4 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Selected Context | BALANCED | CONTROLLED | NOT_MEASURED |

## Headline results

No paired end-task headline is available for this exact model, revision, quantization, engine, profile, and execution mode. Routing diagnostics below must not be interpreted as application quality.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | Mode / no adaptor | Same mode / bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Selected Context | QUALITY | CALIBRATION_PENDING | Selected Context: CALIBRATION_PENDING | Selected Context + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Selected Context | BALANCED | NEEDS_RUN | Selected Context: NEEDS_RUN | Selected Context + Bundle: NOT_APPLICABLE | NEEDS_RUN |
| mlx | Selected Context | ECONOMY | CALIBRATION_PENDING | Selected Context: CALIBRATION_PENDING | Selected Context + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Selected Context | QASPER-LEARNED | NEEDS_RUN | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | NEEDS_RUN |

## Canonical staged evidence

A complete staged cohort is not packaged for this exact identity.

| Condition | Evidence status |
| --- | --- |
| No PRA | `NEEDS_RUN` |
| Selected Context | `NEEDS_RUN` |
| Selected Context + Bundle | `NEEDS_RUN` |

Existing selector-frozen Selected Context versus Native Memory measurements remain reported below as transport evidence; they are not silently relabeled as adaptor evidence.

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect mlx-community/Qwen3.5-27B-4bit -e mlx -a EInnovator/pra-qwen3-5-27b-mlx-4bit
pra evaluate mlx-community/Qwen3.5-27B-4bit -e mlx -D qasper -a EInnovator/pra-qwen3-5-27b-mlx-4bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3.5-27B-4bit -e mlx -a EInnovator/pra-qwen3-5-27b-mlx-4bit -p balanced
```

## Profiles

| Profile | Purpose | Routing | Consumer layers | Status | Recommendation |
| --- | --- | --- | --- | --- | --- |
| QUALITY | Candidate maximum-quality profile; held-out calibration is incomplete | generic cosine | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 | CALIBRATION_PENDING | Not promoted |
| BALANCED | Qualified default preserving the all-eligible consumer geometry | generic cosine | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 | QUALIFIED | Default |
| ECONOMY | Reduced-consumer candidate; the held-out quality gate has not passed | generic cosine | CALIBRATION_PENDING | CALIBRATION_PENDING | Not promoted |
| QASPER-LEARNED | Research-only learned routing profile qualified only on matched QASPER routing diagnostics | combined-router-d128 | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 | RESEARCH | Not promoted |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| mlx | validated | UNAVAILABLE_HYBRID_STATE | NOT_APPLICABLE | Selected Context with BALANCED |

## End-to-end qualification

What remains to be measured: paired end-task quality for this exact bundle identity.

## Native Memory qualification

What remains to be measured: paired Selected Context versus Native Memory quality and serving economics.

## Research diagnostics

| Dataset | Router/profile | Metric | Value | Cohort | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| qasper | balanced | R@20% | 0.3752 | 16 | CONTROLLED |
| qasper | qasper-learned | R@20% | 0.6099 | 16 | CONTROLLED |
| hotpotqa | balanced | R@20% | 0.5183 | 16 | CONTROLLED |
| hotpotqa | qasper-learned | R@20% | 0.3079 | 16 | CONTROLLED |
| combined | balanced | R@20% | 0.4467 | 32 | CONTROLLED |
| combined | qasper-learned | R@20% | 0.4589 | 32 | CONTROLLED |

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Qwen3.5-27B-4bit -e mlx -a EInnovator/pra-qwen3-5-27b-mlx-4bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- The learned router improves QASPER but is not uniformly positive on HotpotQA; it is opt-in rather than the bundle default.
- The held-out routing diagnostic contains 16 examples per dataset and supports controlled routing claims only.
- Detached Native Memory is unavailable for this hybrid recurrent/attention topology; the bundle fails closed to Selected Context.
- The qualification identity is the exact 4bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- The controlled end-task arm contains 16 examples per dataset; routing and generation claims remain dataset-specific.
- Qwen3.5 interleaves 48 Gated DeltaNet layers with 16 full-attention layers. Selected Context and routing are supported, but detached Native Memory is unavailable until recurrent DeltaNet state has an explicit lifecycle and composition contract.
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
- Base Revision: `45797d2985a12c55e6473686e9ea91b95e959553`

## Reproducibility

- PRA commit: `4b9371d2c4d567c804e7c78e3ee908cbb1440924`
- Bundle build commit: `4b9371d2c4d567c804e7c78e3ee908cbb1440924`
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
