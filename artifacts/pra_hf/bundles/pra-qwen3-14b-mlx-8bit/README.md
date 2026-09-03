---
library_name: pra
base_model: mlx-community/Qwen3-14B-8bit
tags:
- pra
- progressive-retrieval-attention
- adapter
- long-context
license: apache-2.0
---

# PRA Runtime Bundle for mlx-community/Qwen3-14B-8bit · MLX / 8bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-14B-8bit`
- Immutable revision: `da33cf28f06636847fd9e93e0a03d819b84cb55e`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `14B`
- Tokenizer revision: `da33cf28f06636847fd9e93e0a03d819b84cb55e`
- Post-training: `pretrained and post-trained`

## Recommended configuration

- Engine: **mlx**
- Recommended PRA mode: **Selected Context**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **SMOKE**
- Native Memory status: **AVAILABLE**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Headline results

No paired end-task headline is available for this exact model, revision, quantization, engine, profile, and execution mode. Routing diagnostics below must not be interpreted as application quality.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | Model Name: MacBook Pro; Chip: Apple M4 Pro; Memory: 48 GB | 569.2 s | 2.828 s | 14.66 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

End-task quality, Native Memory parity, learned routing, TTFT, ITL, and sustained throughput remain `NOT_MEASURED` for this exact identity.

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
pra inspect mlx-community/Qwen3-14B-8bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-8bit
pra evaluate mlx-community/Qwen3-14B-8bit -e mlx -D qasper -a EInnovator/pra-qwen3-14b-mlx-8bit
pra recommend .pra/runs/latest
pra serve mlx-community/Qwen3-14B-8bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-8bit -p balanced
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
| mlx | SMOKE | AVAILABLE | NOT_MEASURED | Selected Context with BALANCED |
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
pra evaluate mlx-community/Qwen3-14B-8bit -e mlx -a EInnovator/pra-qwen3-14b-mlx-8bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed.
- Only immutable-config structural validation is available for this exact quantized identity.
- Native consumer-layer profiles and end-task generation remain uncalibrated for this exact identity.
- The qualification identity is the exact 8bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- The runtime smoke loads the exact quantized checkpoint and generates a fixed prompt; it is not an end-task quality or serving benchmark.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

## Reproducibility

- PRA commit: `5c1be10d5aca50c7ae93194b68e30c0d64fefd0c`
- Bundle build commit: `5c1be10d5aca50c7ae93194b68e30c0d64fefd0c`
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
