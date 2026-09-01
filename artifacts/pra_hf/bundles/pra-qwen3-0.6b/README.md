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

# PRA bundle for Qwen/Qwen3-0.6B

## What this is

This repository contains a Progressive Retrieval Attention (PRA) structural adapter, learned adapters, runtime profiles, compatibility metadata, and qualification evidence. It does not contain or duplicate the base-model weights.

## Base model

- ID: `Qwen/Qwen3-0.6B`
- Immutable revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `596049920`
- Tokenizer revision: `c1899de289a04d12100db370d81485cdf75e47ca`

## What PRA provides

PRA provides portable Selected Context, model-specific structural mapping, optional learned routing, and measured profiles. Native Memory and Native Serving are enabled only on engine/model combinations marked as qualified below.

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

## Bundle contents

| Component | Type | Status | Path |
| --- | --- | --- | --- |
| structural | structural | validated | `structural_adapter` |
| qasper-router-d128 | routing | validated-artifact | `learned_adapters/qasper-router-d128` |

## Profiles

| Profile | Purpose | Routing | Consumer layers | Status |
| --- | --- | --- | --- | --- |
| reference | Training-free structural reference and regression checks | None | all eligible | measured-smoke |
| balanced | QASPER-oriented learned routing with conservative consumers | qasper-router-d128 | all eligible | controlled-research |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| hf | validated | controlled validation | NOT_MEASURED | Selected Context; qualify Native Memory locally |
| mlx | validated | NOT_MEASURED for this exact base revision | NOT_MEASURED | Selected Context |
| vllm | validated | NOT_MEASURED for this bundle | NOT_MEASURED for this bundle | Selected Context |

## Expected metrics

| Engine | Hardware | Workload | Mode | Quality | Visible tokens | TTFT | Throughput | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| huggingface_eager 4.55.4 | NVIDIA GeForce GTX 950M | allenai/qasper (n=16) | Learned routing | 0.4054 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| huggingface_eager | NVIDIA GeForce GTX 950M | paper4_5_cross_model_diagnostic (n=3) | Native Memory (hot) | 0.4364 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | SMOKE |

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate on your system

```bash
pra evaluate Qwen/Qwen3-0.6B -a EInnovator/pra-qwen3-0.6b -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## How to choose Selected Context vs Native Memory

Selected Context is the portable baseline and should be the first deployment. Native Memory is incremental, model-specific, and engine/workload dependent; include it in local qualification before promotion.

## Known limitations

- The routing cohort is small and QASPER-specific; transfer routing is not a production claim.
- Native Memory has controlled HF evidence only for this exact model revision and must be requalified per engine, quantization, and hardware.
- Native Serving is not qualified by this bundle.

## Training / creation

- Datasets: `QASPER`
- Seed: `11`
- Method: `multi-positive softmax`
- Parameter Count: `262144`
- Base Revision: `c1899de289a04d12100db370d81485cdf75e47ca`

## Reproducibility

- PRA commit: `d880a4583df828744f6006976bd78ff66a05f926`
- Bundle build commit: `d880a4583df828744f6006976bd78ff66a05f926`
- Bundle schema: `2`
- PRA package: `0.2.0rc1`
- Component fingerprints and file checksums are recorded in `bundle.yaml`.

## Community and support

- [Canonical PRA Bundles Collection](https://huggingface.co/collections/EInnovator/pra-bundles-6a971e52093232f858e660f6)
- [EInnovator on Hugging Face](https://huggingface.co/EInnovator)
- [PRA documentation](https://einnovator.github.io/pdattention/)
- [Source repository](https://github.com/einnovator/pdattention)
- [Issues](https://github.com/einnovator/pdattention/issues)
- [Contribution guide](https://github.com/einnovator/pdattention/blob/main/CONTRIBUTING.md)
