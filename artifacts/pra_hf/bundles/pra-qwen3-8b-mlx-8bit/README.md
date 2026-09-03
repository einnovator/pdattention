---
library_name: pra
base_model: mlx-community/Qwen3-8B-8bit
tags:
- pra
- progressive-retrieval-attention
- adapter
- long-context
license: apache-2.0
---

# PRA Runtime Bundle for mlx-community/Qwen3-8B-8bit · MLX / 8bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-8B-8bit`
- Immutable revision: `48a0b75b1ae72503e21e1558d040bc227510ff06`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `8B`
- Tokenizer revision: `48a0b75b1ae72503e21e1558d040bc227510ff06`
- Post-training: `pretrained and post-trained`

## Recommended configuration

- Engine: **mlx**
- Recommended PRA mode: **Selected Context**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **NOT_MEASURED**
- Native Memory status: **AVAILABLE**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Headline results

No paired end-task headline is available for this exact model, revision, quantization, engine, profile, and execution mode. Routing diagnostics below must not be interpreted as application quality.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Native Memory | QUALITY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| mlx | Selected Context | BALANCED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| mlx | Native Memory | ECONOMY | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

## Canonical three-condition evidence

A complete matched No PRA / PRA - No Adaptor / PRA - Adaptor Bundle cohort is not packaged for this exact identity.

| Condition | Evidence status |
| --- | --- |
| No PRA | `NEEDS_RUN` |
| PRA - No Adaptor | `NEEDS_RUN` |
| PRA - Adaptor Bundle | `NEEDS_RUN` |

Existing selector-frozen Selected Context versus Native Memory measurements remain reported below as transport evidence; they are not silently relabeled as adaptor evidence.

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect mlx-community/Qwen3-8B-8bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-8bit
pra evaluate mlx-community/Qwen3-8B-8bit -e mlx -D qasper -a EInnovator/pra-qwen3-8b-mlx-8bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3-8B-8bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-8bit -p balanced
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
| mlx | validated | AVAILABLE | NOT_MEASURED | Selected Context with BALANCED |
| hf | portable | NOT_MEASURED for the full-precision HF counterpart | NOT_MEASURED | Selected Context; exact MLX artifact only |

## End-to-end qualification

What remains to be measured: paired end-task quality for this exact bundle identity.

## Native Memory qualification

What remains to be measured: paired Selected Context versus Native Memory quality and serving economics.

## Research diagnostics

No separate routing diagnostic is packaged for this bundle.

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Qwen3-8B-8bit -e mlx -a EInnovator/pra-qwen3-8b-mlx-8bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed.
- Only immutable-config structural validation is available for this exact quantized identity.
- Native consumer-layer profiles and end-task generation remain uncalibrated for this exact identity.
- The qualification identity is the exact 8bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- End-task quality, serving latency, and native-memory parity remain NOT_MEASURED for this exact identity.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

## Reproducibility

- PRA commit: `33850e8e7d53694f771c6f388d3b8e73d21ee482`
- Bundle build commit: `33850e8e7d53694f771c6f388d3b8e73d21ee482`
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
