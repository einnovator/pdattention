---
library_name: pra
base_model: mlx-community/Llama-3.2-1B-Instruct-8bit
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
license: llama3.2
---

# PRA Runtime Bundle for mlx-community/Llama-3.2-1B-Instruct-8bit · MLX / 8bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Llama-3.2-1B-Instruct-8bit`
- Immutable revision: `d48cdf0a4ea22d893b7c63a99d6a693e24822795`
- Architecture: `LlamaForCausalLM`
- Parameters: `1B`
- Tokenizer revision: `d48cdf0a4ea22d893b7c63a99d6a693e24822795`
- Serving precision: `INT8` / `MLX-8bit`
- Post-training: `instruction tuning`

## Recommended configuration

- Engine: **mlx**
- Recommended PRA mode: **Native Memory**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **NEEDS_RERUN**
- Native Memory status: **NEEDS_RERUN**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Precision qualification

Precision evidence is scoped to the exact model conversion, engine, mode, and profile. Qualification does not transfer automatically between BF16, INT8, INT4, or encoding-specific formats.

| Family | Encoding | Serving | Feature extraction | Adaptor parameters | Engine | Mode | Profile | Evidence | Datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 | MLX-8bit | INT8 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Native Memory | BALANCED | NEEDS_RERUN | NOT_MEASURED |

## Headline results

**Evidence correction:** Earlier native-runtime headline measurements are quarantined. They predate the corrected live-K/V, original-position, copy-accounting, and lifecycle contract. The bundle's structural mapping and routing-only diagnostics remain available, but native exactness, latency, throughput, memory, cache-hit, and copy-savings claims require rerun.

No paired end-task headline is available for this exact model, revision, quantization, engine, profile, and execution mode. Routing diagnostics below must not be interpreted as application quality.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | Model Name: MacBook Pro; Chip: Apple M4 Pro; Memory: 48 GB | 46.73 s | 0.4636 s | 1.24 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

Runtime smoke does not establish end-task quality, Native Memory parity, routing quality, or serving economics. The coverage table below identifies the exact follow-up state.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

**Evidence correction:** 1 pre-fix native-runtime record(s) are quarantined from this card. They predate the corrected live-K/V, original-position, copy-accounting, and lifecycle contract and require rerun. Structural compatibility and routing-only evidence are unaffected.

| Engine | Mode | Profile | No PRA | Mode / no adaptor | Same mode / bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Native Memory | QUALITY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Native Memory | BALANCED | NEEDS_RUN | Native Memory: NEEDS_RUN | Native Memory + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RUN |
| mlx | Native Memory | ECONOMY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |

## Canonical staged evidence

A complete staged cohort is not packaged for this exact identity.

| Condition | Evidence status |
| --- | --- |
| No PRA | `NEEDS_RUN` |
| Native Memory | `NEEDS_RUN` |
| Native Memory + Bundle | `NO_QUALIFIED_ADAPTER` |

Earlier selector-frozen Selected Context versus Native Memory measurements are quarantined pending corrected-engine rerun.

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect mlx-community/Llama-3.2-1B-Instruct-8bit -e mlx -a EInnovator/pra-llama3-2-1b-mlx-8bit
pra evaluate mlx-community/Llama-3.2-1B-Instruct-8bit -e mlx -D qasper -a EInnovator/pra-llama3-2-1b-mlx-8bit
pra recommend .pra/runs/latest
pra serve mlx-community/Llama-3.2-1B-Instruct-8bit -e mlx -a EInnovator/pra-llama3-2-1b-mlx-8bit -p balanced
```

## Profiles

| Profile | Purpose | Routing | Consumer layers | Status | Recommendation |
| --- | --- | --- | --- | --- | --- |
| QUALITY | Candidate maximum-quality profile; held-out calibration is incomplete | generic cosine | all eligible | NEEDS_RERUN | Not promoted |
| BALANCED | Qualified default preserving the all-eligible consumer geometry | generic cosine | all eligible | NEEDS_RERUN | Pending rerun |
| ECONOMY | Reduced-consumer candidate; the held-out quality gate has not passed | generic cosine | CALIBRATION_PENDING | NEEDS_RERUN | Not promoted |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| mlx | validated | NEEDS_RERUN | NOT_APPLICABLE | Selected Context pending rerun |

## End-to-end qualification

What remains to be measured: paired end-task quality for this exact bundle identity.

## Native Memory qualification

What remains to be measured: paired Selected Context versus Native Memory quality and serving economics.

## Research diagnostics

No separate routing diagnostic is packaged for this bundle.

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/Llama-3.2-1B-Instruct-8bit -e mlx -a EInnovator/pra-llama3-2-1b-mlx-8bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed.
- Paired natural-QA evidence contains 20 examples per dataset and supports engine qualification, not production qualification.
- Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers.
- The qualification identity is the exact 8bit MLX model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- The selector-frozen natural-QA run qualifies the generic Native Memory path; an exact learned-adaptor arm still requires a separate run.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

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
