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
- Bundle evidence tier: **ENGINE_QUALIFIED**
- Native Memory status: **QUALIFIED**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Precision qualification

Precision evidence is scoped to the exact model conversion, engine, mode, and profile. Qualification does not transfer automatically between BF16, INT8, INT4, or encoding-specific formats.

| Family | Encoding | Serving | Feature extraction | Adaptor parameters | Engine | Mode | Profile | Evidence | Datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 | MLX-8bit | INT8 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Native Memory | BALANCED | ENGINE_QUALIFIED | NOT_MEASURED |

## Headline results

| Workload | Selected Context quality | Native Memory quality | Delta NM vs SC | Visible-context delta NM vs SC | TTFT delta NM vs SC | Completion delta NM vs SC | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=60) | token_f1=0.1254 | token_f1=0.1254 | +0.0000 | -91.5% | -6.9% | -0.2% | 60/60 | ENGINE_QUALIFIED |

All headline rows freeze the PRA-selected evidence. Deltas are Native Memory minus Selected Context; negative latency and visible-context deltas are reductions. These rows contain no ordinary No-PRA arm.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA; cold direct query (n=60); 2026-09-03; PRA commit `None`; artifact `qualification/matched_e0_e2_qasper.json, qualification/matched_e0_e2_hotpotqa.json, qualification/matched_e0_e2_2wikimultihopqa.json`; SHA-256 `dbc4db9f1b5d80a0ba26eaa63344dcdcdc1989570e9d21cba6149c13d50e3119,00d5e4f84097041acc4d468774991392eee47525ca3e88b12afbee814b73e4a1,c1fd19cef98ccf91221a2cb7a12f500b4483c6df7c0d6467e04cdd1d67add8b5`.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | Model Name: MacBook Pro; Chip: Apple M4 Pro; Memory: 48 GB | 46.73 s | 0.4636 s | 1.24 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

Runtime smoke does not establish end-task quality, Native Memory parity, routing quality, or serving economics. The coverage table below identifies the exact follow-up state.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | Mode / no adaptor | Same mode / bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Native Memory | QUALITY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Native Memory | BALANCED | NEEDS_RUN | Native Memory: MEASURED (16) | Native Memory + Bundle: NO_QUALIFIED_ADAPTER | context, quality, resources, routing, serving |
| mlx | Native Memory | ECONOMY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |

## Canonical staged evidence

Each table holds task, hardware, engine, model, precision, and profile fixed. Every delta names its source and target; bundle use is held orthogonal to execution depth.

### combined / mlx-lm / balanced

Exact identity: `mlx-community/Llama-3.2-1B-Instruct-8bit` at `d48cdf0a4ea22d893b7c63a99d6a693e24822795` on `Apple M4 Pro (Mac16,7), 48 GB`; precision `INT8` / `MLX-8bit`.

#### Quality

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.125357 | 0.125357 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0 | 0 | NO_QUALIFIED_ADAPTER | +0 | NO_QUALIFIED_ADAPTER |
| Gold Answer Log Probability | log_probability | higher_is_better | -15.0841 | -15.0841 | NO_QUALIFIED_ADAPTER | +0 (-0.00%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 390.867 | 33.25 | NO_QUALIFIED_ADAPTER | -357.617 (-91.49%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 357.617 | NO_QUALIFIED_ADAPTER | +357.617 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 31.8044 | 29.6227 | NO_QUALIFIED_ADAPTER | -2.18167 (-6.86%) | NO_QUALIFIED_ADAPTER |
| TTFT p95 (ms) | ms | lower_is_better | 36.076 | 31.4482 | NO_QUALIFIED_ADAPTER | -4.62779 (-12.83%) | NO_QUALIFIED_ADAPTER |
| TTFT p99 (ms) | ms | lower_is_better | 87.1491 | 92.627 | NO_QUALIFIED_ADAPTER | +5.47783 (+6.29%) | NO_QUALIFIED_ADAPTER |
| ITL p50 (ms) | ms | lower_is_better | 6.13011 | 6.26277 | NO_QUALIFIED_ADAPTER | +0.132659 (+2.16%) | NO_QUALIFIED_ADAPTER |
| ITL p95 (ms) | ms | lower_is_better | 6.66693 | 6.42713 | NO_QUALIFIED_ADAPTER | -0.239795 (-3.60%) | NO_QUALIFIED_ADAPTER |
| ITL p99 (ms) | ms | lower_is_better | 7.20762 | 7.41378 | NO_QUALIFIED_ADAPTER | +0.206161 (+2.86%) | NO_QUALIFIED_ADAPTER |
| Output Tokens Per Second | output_token/s | higher_is_better | 138.47 | 138.864 | NO_QUALIFIED_ADAPTER | +0.394598 (+0.28%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 173.813 | 173.392 | NO_QUALIFIED_ADAPTER | -0.421136 (-0.24%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 1.17184e+07 | NO_QUALIFIED_ADAPTER | +1.17184e+07 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 1.17184e+07 | NO_QUALIFIED_ADAPTER | +1.17184e+07 | NO_QUALIFIED_ADAPTER |

#### Routing

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Evidence Recall | fraction | higher_is_better | 0.615972 | 0.615972 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |

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
| QUALITY | Candidate maximum-quality profile; held-out calibration is incomplete | generic cosine | all eligible | CALIBRATION_PENDING | Not promoted |
| BALANCED | Qualified default preserving the all-eligible consumer geometry | generic cosine | all eligible | QUALIFIED | Default |
| ECONOMY | Reduced-consumer candidate; the held-out quality gate has not passed | generic cosine | CALIBRATION_PENDING | CALIBRATION_PENDING | Not promoted |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| mlx | validated | QUALIFIED | NOT_APPLICABLE | Native Memory with BALANCED |

## End-to-end qualification

| Workload | Mode | Quality | Visible tokens | TTFT p50 | Completion mean | Hardware | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qasper (n=20) | Selected Context | token_f1=0.1452 | 397.4 | 26.08 ms | 173.9 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=20) | Native Memory | token_f1=0.1452 | 28.05 | 22.77 ms | 172.2 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Selected Context | token_f1=0.1367 | 410.5 | 33.6 ms | 175 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Native Memory | token_f1=0.1367 | 38.55 | 30.68 ms | 176.5 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Selected Context | token_f1=0.09413 | 364.8 | 31.51 ms | 172.5 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Native Memory | token_f1=0.09413 | 33.15 | 29.61 ms | 171.4 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Selected Context | token_f1=0.1254 | 390.9 | 31.8 ms | 173.8 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Native Memory | token_f1=0.1254 | 33.25 | 29.62 ms | 173.4 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| qasper | 369.3 | 11.54 MiB | NEEDS_RUN | 0.9903x |
| hotpotqa | 371.9 | 11.62 MiB | NEEDS_RUN | 1.009x |
| 2wikimultihopqa | 331.6 | 10.36 MiB | NEEDS_RUN | 0.9937x |
| combined | 357.6 | 11.18 MiB | NEEDS_RUN | 0.9976x |

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
