---
library_name: pra
base_model: mlx-community/gemma-3-1b-it-8bit
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
license: gemma
---

# PRA Runtime Bundle for mlx-community/gemma-3-1b-it-8bit · MLX / 8bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/gemma-3-1b-it-8bit`
- Immutable revision: `7b963136f21d05ca8b367c93a9c47a7944ad281a`
- Architecture: `Gemma3ForCausalLM`
- Parameters: `1B`
- Tokenizer revision: `7b963136f21d05ca8b367c93a9c47a7944ad281a`
- Serving precision: `INT8` / `MLX-8bit`
- Post-training: `pretrained and post-trained`

## Recommended configuration

- Engine: **mlx**
- Recommended PRA mode: **Selected Context**
- Recommended profile: **BALANCED**
- Bundle evidence tier: **CONTROLLED**
- Native Memory status: **CONTROLLED**

Availability, qualification, and recommendation are separate. A mode may be implemented without being qualified or recommended for this identity.

## Precision qualification

Precision evidence is scoped to the exact model conversion, engine, mode, and profile. Qualification does not transfer automatically between BF16, INT8, INT4, or encoding-specific formats.

| Family | Encoding | Serving | Feature extraction | Adaptor parameters | Engine | Mode | Profile | Evidence | Datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 | MLX-8bit | INT8 | NEEDS_RUN | NO_QUALIFIED_ADAPTER | mlx | Selected Context | BALANCED | ENGINE_QUALIFIED | NOT_MEASURED |

## Headline results

| Workload | Selected Context quality | Native Memory quality | Delta NM vs SC | Visible-context delta NM vs SC | TTFT delta NM vs SC | Completion delta NM vs SC | Paired parity | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| combined (n=60) | token_f1=0.07352 | token_f1=0.06991 | -0.0036 | -91.1% | -5.1% | +5.9% | 3/60 | ENGINE_QUALIFIED |

All headline rows freeze the PRA-selected evidence. Deltas are Native Memory minus Selected Context; negative latency and visible-context deltas are reductions. These rows contain no ordinary No-PRA arm.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA; cold direct query (n=60); 2026-09-03; PRA commit `None`; artifact `qualification/matched_e0_e2_qasper.json, qualification/matched_e0_e2_hotpotqa.json, qualification/matched_e0_e2_2wikimultihopqa.json`; SHA-256 `ca7c9a7929b5376b269b0b312d01dff3f5ab7f3263a5fba7c3ce084da6546fad,80cc399baab9fc1a500bb9375ef0d82ce47e8cb1bd552373970309cebcc1ea0d,0177e860677d0f12513bbf155bac3ab849710f5e6456ed1ab15528661ffa2384`.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | Model Name: MacBook Pro; Chip: Apple M4 Pro; Memory: 48 GB | 41.34 s | 1.174 s | 1.30 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

Runtime smoke does not establish end-task quality, Native Memory parity, routing quality, or serving economics. The coverage table below identifies the exact follow-up state.

## Evidence by engine, mode, and profile

Each row identifies the exact runtime surface for which metrics are available. `MEASURED` counts scalar metrics with real observations; missing profile/mode combinations are not inferred from another row.

| Engine | Mode | Profile | No PRA | Mode / no adaptor | Same mode / bundle | Measured metric groups |
| --- | --- | --- | --- | --- | --- | --- |
| mlx | Native Memory | QUALITY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |
| mlx | Selected Context | BALANCED | NEEDS_RUN | Selected Context: NEEDS_RUN | Selected Context + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RUN |
| mlx | Native Memory | ECONOMY | CALIBRATION_PENDING | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CALIBRATION_PENDING |

## Canonical staged evidence

Each table holds task, hardware, engine, model, precision, and profile fixed. Every delta names its source and target; bundle use is held orthogonal to execution depth.

### combined / mlx-lm / balanced

Exact identity: `mlx-community/gemma-3-1b-it-8bit` at `7b963136f21d05ca8b367c93a9c47a7944ad281a` on `Apple M4 Pro (Mac16,7), 48 GB`; precision `INT8` / `MLX-8bit`.

#### Quality

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.0735174 | 0.0699096 | NO_QUALIFIED_ADAPTER | -0.0036078 (-4.91%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0 | 0 | NO_QUALIFIED_ADAPTER | +0 | NO_QUALIFIED_ADAPTER |
| Gold Answer Log Probability | log_probability | higher_is_better | -23.0353 | -20.6684 | NO_QUALIFIED_ADAPTER | +2.36697 (-10.28%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 395.317 | 35.1667 | NO_QUALIFIED_ADAPTER | -360.15 (-91.10%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 360.15 | NO_QUALIFIED_ADAPTER | +360.15 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 32.6564 | 31.0044 | NO_QUALIFIED_ADAPTER | -1.65196 (-5.06%) | NO_QUALIFIED_ADAPTER |
| TTFT p95 (ms) | ms | lower_is_better | 39.1253 | 46.6731 | NO_QUALIFIED_ADAPTER | +7.54779 (+19.29%) | NO_QUALIFIED_ADAPTER |
| TTFT p99 (ms) | ms | lower_is_better | 46.3028 | 58.7391 | NO_QUALIFIED_ADAPTER | +12.4364 (+26.86%) | NO_QUALIFIED_ADAPTER |
| ITL p50 (ms) | ms | lower_is_better | 5.91003 | 6.36293 | NO_QUALIFIED_ADAPTER | +0.452904 (+7.66%) | NO_QUALIFIED_ADAPTER |
| ITL p95 (ms) | ms | lower_is_better | 6.44007 | 7.00265 | NO_QUALIFIED_ADAPTER | +0.562576 (+8.74%) | NO_QUALIFIED_ADAPTER |
| ITL p99 (ms) | ms | lower_is_better | 7.41584 | 7.40068 | NO_QUALIFIED_ADAPTER | -0.0151522 (-0.20%) | NO_QUALIFIED_ADAPTER |
| Output Tokens Per Second | output_token/s | higher_is_better | 141.269 | 133.394 | NO_QUALIFIED_ADAPTER | -7.87498 (-5.57%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 170.376 | 180.421 | NO_QUALIFIED_ADAPTER | +10.0445 (+5.90%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 9.58863e+06 | NO_QUALIFIED_ADAPTER | +9.58863e+06 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 9.58863e+06 | NO_QUALIFIED_ADAPTER | +9.58863e+06 | NO_QUALIFIED_ADAPTER |

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
pra inspect mlx-community/gemma-3-1b-it-8bit -e mlx -a EInnovator/pra-gemma3-1b-mlx-8bit
pra evaluate mlx-community/gemma-3-1b-it-8bit -e mlx -D qasper -a EInnovator/pra-gemma3-1b-mlx-8bit
pra recommend .pra/runs/latest
pra serve mlx-community/gemma-3-1b-it-8bit -e mlx -a EInnovator/pra-gemma3-1b-mlx-8bit -p balanced
```

## Profiles

| Profile | Purpose | Routing | Consumer layers | Status | Recommendation |
| --- | --- | --- | --- | --- | --- |
| QUALITY | Candidate maximum-quality profile; held-out calibration is incomplete | generic cosine | 5, 11, 17, 23 | CALIBRATION_PENDING | Not promoted |
| BALANCED | Qualified default preserving the all-eligible consumer geometry | generic cosine | 5, 11, 17, 23 | QUALIFIED | Default |
| ECONOMY | Reduced-consumer candidate; the held-out quality gate has not passed | generic cosine | CALIBRATION_PENDING | CALIBRATION_PENDING | Not promoted |

## Engine compatibility

| Engine | Selected Context | Native Memory | Native Serving | Recommended today |
| --- | --- | --- | --- | --- |
| mlx | validated | CONTROLLED | NOT_APPLICABLE | Selected Context with BALANCED |

## End-to-end qualification

| Workload | Mode | Quality | Visible tokens | TTFT p50 | Completion mean | Hardware | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qasper (n=20) | Selected Context | token_f1=0.06609 | 400.2 | 27.46 ms | 166 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=20) | Native Memory | token_f1=0.0443 | 30.05 | 24.88 ms | 175.7 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Selected Context | token_f1=0.08701 | 415.6 | 32.89 ms | 171 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Native Memory | token_f1=0.1083 | 40.5 | 31.13 ms | 180.7 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Selected Context | token_f1=0.06745 | 370.1 | 35.99 ms | 174.1 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Native Memory | token_f1=0.0571 | 34.95 | 34.54 ms | 184.8 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Selected Context | token_f1=0.07352 | 395.3 | 32.66 ms | 170.4 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Native Memory | token_f1=0.06991 | 35.17 | 31 ms | 180.4 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| qasper | 370.2 | 9.40 MiB | NEEDS_RUN | 1.059x |
| hotpotqa | 375.1 | 9.52 MiB | NEEDS_RUN | 1.057x |
| 2wikimultihopqa | 335.2 | 8.51 MiB | NEEDS_RUN | 1.062x |
| combined | 360.1 | 9.14 MiB | NEEDS_RUN | 1.059x |

## Research diagnostics

No separate routing diagnostic is packaged for this bundle.

These are qualification measurements, not guaranteed production performance. Run `pra evaluate` on your hardware and workload. Engine version, profile, cohort, evidence tier, date, and artifact provenance remain recorded in `qualification/` and `bundle.yaml`.

## How to evaluate locally

```bash
pra evaluate mlx-community/gemma-3-1b-it-8bit -e mlx -a EInnovator/pra-gemma3-1b-mlx-8bit -D qasper -o .pra/runs/qasper
pra recommend .pra/runs/qasper
pra report .pra/runs/qasper --format html
```

## Known limitations

- No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed.
- Paired natural-QA evidence contains 20 examples per dataset and supports engine qualification, not production qualification.
- Native Memory is measured but remains a candidate because the exact-output equivalence gate did not pass.
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
