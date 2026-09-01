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

# PRA bundle for mlx-community/Llama-3.1-8B-Instruct-4bit

## What this is

This repository contains a Progressive Retrieval Attention (PRA) structural adapter, learned adapters, runtime profiles, compatibility metadata, and qualification evidence. It does not contain or duplicate the base-model weights.

## Base model

- ID: `mlx-community/Llama-3.1-8B-Instruct-4bit`
- Immutable revision: `90215b22ec18e72f623dde2ea7af4097025160e2`
- Architecture: `LlamaForCausalLM`
- Parameters: `8B`
- Tokenizer revision: `90215b22ec18e72f623dde2ea7af4097025160e2`

## What PRA provides

PRA provides portable Selected Context, model-specific structural mapping, optional learned routing, and measured profiles. Native Memory and Native Serving are enabled only on engine/model combinations marked as qualified below.

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

## Bundle contents

| Component | Type | Status | Path |
| --- | --- | --- | --- |
| structural | structural | validated | `structural_adapter` |
| combined-router-d128 | routing | controlled-artifact | `learned_adapters/combined-router-d128` |

## Profiles

| Profile | Purpose | Routing | Consumer layers | Status |
| --- | --- | --- | --- | --- |
| reference | Training-free structural checks using generic cosine routing | generic cosine | all eligible | controlled |
| balanced | Portable default using generic cosine routing | generic cosine | all eligible | controlled-default |
| qasper-learned | Opt-in learned routing qualified for QASPER; not a HotpotQA default | combined-router-d128 | all eligible | controlled-dataset-specific |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| mlx | validated | structural mapping controlled for this exact MLX identity | engine-study evidence; not established by routing qualification | balanced generic; qasper-learned only for matched QASPER workloads |
| hf | portable | NOT_MEASURED for the full-precision HF counterpart | NOT_MEASURED | Selected Context; exact MLX artifact only |

## Expected metrics

| Engine | Hardware | Workload | Mode | Quality metric | Visible tokens | TTFT | Throughput | Status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| mlx-lm 0.31.3 | Apple M4 Pro, 48 GB | qasper (n=16) | Generic cosine routing | R@20%=0.3182 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| mlx-lm 0.31.3 | Apple M4 Pro, 48 GB | qasper (n=16) | Learned asymmetric routing | R@20%=0.4683 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| mlx-lm 0.31.3 | Apple M4 Pro, 48 GB | hotpotqa (n=16) | Generic cosine routing | R@20%=0.6158 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| mlx-lm 0.31.3 | Apple M4 Pro, 48 GB | hotpotqa (n=16) | Learned asymmetric routing | R@20%=0.4205 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| mlx-lm 0.31.3 | Apple M4 Pro, 48 GB | combined (n=32) | Generic cosine routing | R@20%=0.467 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |
| mlx-lm 0.31.3 | Apple M4 Pro, 48 GB | combined (n=32) | Learned asymmetric routing | R@20%=0.4444 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | CONTROLLED |

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate on your system

```bash
pra evaluate mlx-community/Llama-3.1-8B-Instruct-4bit -e mlx -a EInnovator/pra-llama3-1-8b-mlx-4bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## How to choose Selected Context vs Native Memory

Selected Context is the portable baseline and should be the first deployment. Native Memory is incremental, model-specific, and engine/workload dependent; include it in local qualification before promotion.

## Known limitations

- The learned router improves QASPER but is not uniformly positive on HotpotQA; it is opt-in rather than the bundle default.
- Routing evidence uses 16 held-out examples per dataset and does not establish generation quality or serving economics.
- The qualification identity is the exact 4-bit MLX model and revision; it does not transfer automatically to full-precision Hugging Face weights or another quantization.
- Base-model and dataset licenses apply separately to the router artifact.

## Training / creation

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

- PRA commit: `27b3ce12d8aec6b6f7855b65204de6b10c4aeb71`
- Bundle build commit: `27b3ce12d8aec6b6f7855b65204de6b10c4aeb71`
- Bundle schema: `2`
- PRA package: `0.2.0rc1`
- Component fingerprints and file checksums are recorded in `bundle.yaml`.

## Community and support

- [PRA documentation](https://einnovator.github.io/pdattention/)
- [Source repository](https://github.com/einnovator/pdattention)
- [Issues](https://github.com/einnovator/pdattention/issues)
- [Contribution guide](https://github.com/einnovator/pdattention/blob/main/CONTRIBUTING.md)
- [Canonical PRA Bundles Collection](https://huggingface.co/collections/EInnovator/pra-bundles-6a971e52093232f858e660f6)
- [EInnovator on Hugging Face](https://huggingface.co/EInnovator)
