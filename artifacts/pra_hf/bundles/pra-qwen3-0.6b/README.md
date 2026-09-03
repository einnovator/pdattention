---
library_name: pra
base_model: Qwen/Qwen3-0.6B
tags:
- pra
- progressive-retrieval-attention
- adapter
- long-context
datasets:
- allenai/qasper
- paper4_5_cross_model_diagnostic
license: apache-2.0
---

# PRA Runtime Bundle for Qwen/Qwen3-0.6B · HF

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `Qwen/Qwen3-0.6B`
- Immutable revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `596049920`
- Tokenizer revision: `c1899de289a04d12100db370d81485cdf75e47ca`

## Recommended configuration

- Engine: **hf**
- Recommended PRA mode: **Selected Context**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **RESEARCH**
- Native Memory status: **controlled validation**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Headline results

No paired end-task headline is available for this exact model, revision, quantization, engine, profile, and execution mode. Routing diagnostics below must not be interpreted as application quality.

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
pra inspect Qwen/Qwen3-0.6B -e hf -a EInnovator/pra-qwen3-0.6b
pra evaluate Qwen/Qwen3-0.6B -e hf -D qasper -a EInnovator/pra-qwen3-0.6b
pra recommend .pra/runs/latest
pra serve Qwen/Qwen3-0.6B -e hf -a EInnovator/pra-qwen3-0.6b -p balanced
```

## Profiles

| Profile | Purpose | Routing | Consumer layers | Status | Recommendation |
| --- | --- | --- | --- | --- | --- |
| QUALITY | Candidate maximum-quality profile; calibration is incomplete | generic cosine | all eligible | CALIBRATION_PENDING | Not promoted |
| BALANCED | Conservative training-free reference configuration | generic cosine | all eligible | QUALIFIED | Default |
| ECONOMY | Reduced-consumer candidate; calibration is incomplete | generic cosine | CALIBRATION_PENDING | CALIBRATION_PENDING | Not promoted |
| QASPER-LEARNED | Research-only QASPER routing profile | qasper-router-d128 | all eligible | RESEARCH | Not promoted |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| hf | validated | controlled validation | NOT_MEASURED | Selected Context; qualify Native Memory locally |
| mlx | validated | NOT_MEASURED for this exact base revision | NOT_MEASURED | Selected Context |
| vllm | validated | NOT_MEASURED for this bundle | NOT_MEASURED for this bundle | Selected Context |

## End-to-end qualification

What remains to be measured: paired end-task quality for this exact bundle identity.

## Native Memory qualification

What remains to be measured: paired Selected Context versus Native Memory quality and serving economics.

## Research diagnostics

| Dataset | Router/profile | Metric | Value | Cohort | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| allenai/qasper | qasper-learned | None | 0.4054 | 16 | CONTROLLED |

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate Qwen/Qwen3-0.6B -e hf -a EInnovator/pra-qwen3-0.6b -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- This is a research/reference bundle for mechanism reproduction, not the flagship production-economics demonstration.
- The routing cohort is small and QASPER-specific; transfer routing is not a production claim.
- Native Memory has controlled HF evidence only for this exact model revision and must be requalified per engine, quantization, and hardware.
- Native Serving is not qualified by this bundle.

## Training/creation

- Datasets: `QASPER`
- Seed: `11`
- Method: `multi-positive softmax`
- Parameter Count: `262144`
- Base Revision: `c1899de289a04d12100db370d81485cdf75e47ca`

## Reproducibility

- PRA commit: `52ea386f56fc8d81346325619f3523e585abf78d`
- Bundle build commit: `52ea386f56fc8d81346325619f3523e585abf78d`
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
