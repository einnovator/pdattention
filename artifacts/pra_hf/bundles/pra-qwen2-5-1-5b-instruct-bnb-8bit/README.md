---
library_name: pra
base_model: Qwen/Qwen2.5-1.5B-Instruct
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

# PRA Runtime Bundle for Qwen/Qwen2.5-1.5B-Instruct · HF / 8bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Immutable revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Architecture: `Qwen2ForCausalLM`
- Parameters: `1.5B`
- Tokenizer revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Post-training: `general instruction tuning`

## Recommended configuration

- Engine: **hf**
- Recommended PRA mode: **Selected Context**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **CONTROLLED**
- Native Memory status: **CONTROLLED**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Headline results

| Workload | Baseline quality | PRA quality | Quality Δ | Input/context Δ | TTFT Δ | Completion Δ | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=15) | token_f1=0.3991 | token_f1=0.01786 | -0.3813 | -91.7% | NEEDS_RUN | +203.5% | 0/15 | ENGINE_QUALIFIED |

All headline rows use the same frozen selected evidence in the baseline and PRA paths. Deltas are PRA minus baseline; negative latency and context deltas are reductions.

Evidence receipt: `huggingface_eager 5.16.1`; NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB; selector-frozen natural QA; cold direct query (n=15); 2026-09-03; PRA commit `None`; artifact `qualification/matched_e0_e2_qasper.json, qualification/matched_e0_e2_hotpotqa.json, qualification/matched_e0_e2_2wikimultihopqa.json`; SHA-256 `d6a7b3eab5faf9c6641e4f9894893f41c87978329dcf1167e868a16ef93a1dcd,2f9866b196ac90c0937d74951998ec4b925dff2d312a20562c6f68159b3ea3dc,fb234090cf8d0d55eb725664d084b7e146077a00944d2220e9ef02cf61388737`.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB | 15.76 s | 4.841 s | 1.71 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

Runtime smoke does not establish end-task quality, Native Memory parity, routing quality, or serving economics. The coverage table below identifies the exact follow-up state.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | PRA - No Adaptor | PRA - Adaptor Bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| hf | Native Memory | QUALITY | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING |
| hf | Selected Context | BALANCED | NEEDS_RUN | NEEDS_RUN | NO_QUALIFIED_ADAPTER | NEEDS_RUN |
| hf | Native Memory | ECONOMY | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING | CALIBRATION_PENDING |

## Canonical three-condition evidence

Each table holds task, hardware, engine, model, mode, and profile fixed. Deltas are candidate minus No PRA and retain their mathematical sign.

### combined / huggingface_eager / balanced

Exact identity: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` on `NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB`.

#### Quality

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.399111 | 0.0178571 | -0.381254 (-95.53%) |
| Exact Match | fraction | higher_is_better | 0.333333 | 0 | -0.333333 (-100.00%) |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Context

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 404.6 | 33.6 | -371 (-91.70%) |
| Selected Native K/V Tokens | token | neutral | 0 | 371 | +371 |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Serving

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Output Tokens Per Second | output_token/s | higher_is_better | 2.30576 | 2.09692 | -0.208833 (-9.06%) |
| Completion Latency Mean (ms) | ms | lower_is_better | 3772.26 | 11448.4 | +7676.19 (+203.49%) |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Resources

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 1.06373e+07 | +1.06373e+07 |
| Retained Detail Bytes | byte | lower_is_better | 0 | 1.06373e+07 | +1.06373e+07 |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

#### Routing

| Metric | Unit | Direction | No PRA | PRA - No Adaptor | Delta No Adaptor |
| --- | --- | --- | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.75 | 0.75 | +0 (+0.00%) |

PRA - Adaptor Bundle: `NO_QUALIFIED_ADAPTER` for this metric group; the transport run did not evaluate an immutable learned-adaptor condition.

## Installation

```bash
pip install 'pra-hf[hf-hub,hf-runtime]'
pra doctor
```

## Quickstart

```bash
pra inspect Qwen/Qwen2.5-1.5B-Instruct -e hf -a EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit
pra evaluate Qwen/Qwen2.5-1.5B-Instruct -e hf -D qasper -a EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit
pra recommend .pra/runs/latest
pra serve Qwen/Qwen2.5-1.5B-Instruct -e hf -a EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit -p balanced
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
| hf | validated | CONTROLLED | NOT_APPLICABLE | Selected Context with BALANCED |

## End-to-end qualification

| Workload | Mode | Quality | Visible tokens | TTFT p50 | Completion mean | Hardware | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qasper (n=5) | Selected Context | token_f1=0.064 | 380.6 | NEEDS_RUN ms | 6854 ms | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB | ENGINE_QUALIFIED |
| qasper (n=5) | Native Memory | token_f1=0 | 27.4 | NEEDS_RUN ms | 1.13e+04 ms | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB | ENGINE_QUALIFIED |
| hotpotqa (n=5) | Selected Context | token_f1=0.3333 | 423.6 | NEEDS_RUN ms | 2297 ms | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB | ENGINE_QUALIFIED |
| hotpotqa (n=5) | Native Memory | token_f1=0.05357 | 40 | NEEDS_RUN ms | 1.146e+04 ms | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=5) | Selected Context | token_f1=0.8 | 409.6 | NEEDS_RUN ms | 2165 ms | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=5) | Native Memory | token_f1=0 | 33.4 | NEEDS_RUN ms | 1.158e+04 ms | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB | ENGINE_QUALIFIED |
| combined (n=15) | Selected Context | token_f1=0.3991 | 404.6 | NEEDS_RUN ms | 3772 ms | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB | ENGINE_QUALIFIED |
| combined (n=15) | Native Memory | token_f1=0.01786 | 33.6 | NEEDS_RUN ms | 1.145e+04 ms | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| qasper | 353.2 | 9.66 MiB | NEEDS_RUN | 1.649x |
| hotpotqa | 383.6 | 10.49 MiB | NEEDS_RUN | 4.988x |
| 2wikimultihopqa | 376.2 | 10.29 MiB | NEEDS_RUN | 5.349x |
| combined | 371 | 10.14 MiB | NEEDS_RUN | 3.035x |

## Research diagnostics

No separate routing diagnostic is packaged for this bundle.

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate Qwen/Qwen2.5-1.5B-Instruct -e hf -a EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed.
- Paired natural-QA evidence contains 5 examples per dataset and supports engine qualification, not production qualification.
- Native Memory is measured but remains a candidate because the exact-output equivalence gate did not pass.
- The qualification identity is the exact bnb-8bit HF model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.
- The selector-frozen natural-QA run qualifies the generic Native Memory path; an exact learned-adaptor arm still requires a separate run.
- Base-model and dataset licenses apply separately to the router artifact.

## Training/creation

The structural adapter is training-free. Learned-component training metadata is stored beside each component and summarized in `bundle.yaml`.

## Reproducibility

- PRA commit: `15b73e96201951fdd47e6925dda9415236ddc7b7`
- Bundle build commit: `15b73e96201951fdd47e6925dda9415236ddc7b7`
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
