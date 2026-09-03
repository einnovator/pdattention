---
library_name: pra
base_model: mlx-community/Qwen3-8B-8bit
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

# PRA Runtime Bundle for mlx-community/Qwen3-8B-8bit · MLX / 8bit

## What this PRA Runtime Bundle is

This repository packages the model-specific Progressive Retrieval Attention (PRA) structural mapping, runtime profiles, optional learned components, compatibility metadata, and measured qualification evidence. It does not contain the base-model weights and is not an ordinary LoRA quality fine-tune.

- Base model: `mlx-community/Qwen3-8B-8bit`
- Immutable revision: `48a0b75b1ae72503e21e1558d040bc227510ff06`
- Architecture: `Qwen3ForCausalLM`
- Parameters: `8B`
- Tokenizer revision: `48a0b75b1ae72503e21e1558d040bc227510ff06`
- Serving precision: `INT8` / `MLX-8bit`
- Post-training: `pretrained and post-trained`

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
| combined (n=60) | token_f1=0.144 | token_f1=0.144 | +0.0000 | -91.5% | -5.7% | +0.6% | 60/60 | ENGINE_QUALIFIED |

All headline rows freeze the PRA-selected evidence. Deltas are Native Memory minus Selected Context; negative latency and visible-context deltas are reductions. These rows contain no ordinary No-PRA arm.

Evidence receipt: `mlx-lm 0.31.3`; Apple M4 Pro (Mac16,7), 48 GB; selector-frozen natural QA; cold direct query (n=60); 2026-09-03; PRA commit `None`; artifact `qualification/matched_e0_e2_qasper.json, qualification/matched_e0_e2_hotpotqa.json, qualification/matched_e0_e2_2wikimultihopqa.json`; SHA-256 `312d6fc5e1df75d5e8c83598efa35ff960f5ccc078b7766523121cc795cc8bcf,a46a8d6b2b0951a225ff7745af8daf5b0ef11009e1b4175f419fcf50582cfb48,dd6044a32173ce41179b4b34c85883c8039bc7680138fd13bf91962e8772f06f`.

## Exact-identity runtime smoke

This bounded check loads the published quantized checkpoint, discovers the adapter projections, and performs one short generation. It is operational evidence, not an end-task benchmark.

| Status | Host hardware | Load | Generation | Peak model/runtime memory | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| RUNTIME_SMOKE_VALIDATED | Model Name: MacBook Pro; Chip: Apple M4 Pro; Memory: 48 GB | 279.2 s | 1.33 s | 8.15 GiB | exact checkpoint load, adapter projection discovery, and bounded generation |

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

Exact identity: `mlx-community/Qwen3-8B-8bit` at `48a0b75b1ae72503e21e1558d040bc227510ff06` on `Apple M4 Pro (Mac16,7), 48 GB`; precision `INT8` / `MLX-8bit`.

#### Quality

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Token F1 | fraction | higher_is_better | 0.144011 | 0.144011 | NO_QUALIFIED_ADAPTER | +0 (+0.00%) | NO_QUALIFIED_ADAPTER |
| Exact Match | fraction | higher_is_better | 0 | 0 | NO_QUALIFIED_ADAPTER | +0 | NO_QUALIFIED_ADAPTER |
| Gold Answer Log Probability | log_probability | higher_is_better | -19.1076 | -19.1076 | NO_QUALIFIED_ADAPTER | +0 (-0.00%) | NO_QUALIFIED_ADAPTER |

#### Context

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Visible Tokens | token | lower_is_better | 396.317 | 33.5167 | NO_QUALIFIED_ADAPTER | -362.8 (-91.54%) | NO_QUALIFIED_ADAPTER |
| Selected Native K/V Tokens | token | neutral | 0 | 362.8 | NO_QUALIFIED_ADAPTER | +362.8 | NO_QUALIFIED_ADAPTER |

#### Serving

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTFT p50 (ms) | ms | lower_is_better | 186.881 | 176.211 | NO_QUALIFIED_ADAPTER | -10.6696 (-5.71%) | NO_QUALIFIED_ADAPTER |
| TTFT p95 (ms) | ms | lower_is_better | 232.033 | 208.936 | NO_QUALIFIED_ADAPTER | -23.0971 (-9.95%) | NO_QUALIFIED_ADAPTER |
| TTFT p99 (ms) | ms | lower_is_better | 249.496 | 212.698 | NO_QUALIFIED_ADAPTER | -36.7977 (-14.75%) | NO_QUALIFIED_ADAPTER |
| ITL p50 (ms) | ms | lower_is_better | 34.8123 | 35.6742 | NO_QUALIFIED_ADAPTER | +0.861843 (+2.48%) | NO_QUALIFIED_ADAPTER |
| ITL p95 (ms) | ms | lower_is_better | 36.4766 | 36.8092 | NO_QUALIFIED_ADAPTER | +0.332536 (+0.91%) | NO_QUALIFIED_ADAPTER |
| ITL p99 (ms) | ms | lower_is_better | 36.8347 | 37.1582 | NO_QUALIFIED_ADAPTER | +0.323467 (+0.88%) | NO_QUALIFIED_ADAPTER |
| Output Tokens Per Second | output_token/s | higher_is_better | 24.4728 | 24.3286 | NO_QUALIFIED_ADAPTER | -0.144183 (-0.59%) | NO_QUALIFIED_ADAPTER |
| Completion Latency Mean (ms) | ms | lower_is_better | 981.993 | 987.449 | NO_QUALIFIED_ADAPTER | +5.4559 (+0.56%) | NO_QUALIFIED_ADAPTER |

#### Resources

| Metric | Unit | Direction | Selected Context | Native Memory | Native Memory + Bundle | Delta NM vs SC | Delta Bundle vs NM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Active Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NO_QUALIFIED_ADAPTER | +5.3497e+07 | NO_QUALIFIED_ADAPTER |
| Retained Detail Bytes | byte | lower_is_better | 0 | 5.3497e+07 | NO_QUALIFIED_ADAPTER | +5.3497e+07 | NO_QUALIFIED_ADAPTER |

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
| mlx | validated | QUALIFIED | NOT_APPLICABLE | Native Memory with BALANCED |

## End-to-end qualification

| Workload | Mode | Quality | Visible tokens | TTFT p50 | Completion mean | Hardware | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qasper (n=20) | Selected Context | token_f1=0.1681 | 398.6 | 154.3 ms | 973.7 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| qasper (n=20) | Native Memory | token_f1=0.1681 | 28.05 | 144.3 ms | 981.8 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Selected Context | token_f1=0.1594 | 415.4 | 221.6 ms | 1004 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| hotpotqa (n=20) | Native Memory | token_f1=0.1594 | 39.05 | 206.4 ms | 1008 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Selected Context | token_f1=0.1045 | 374.9 | 209.6 ms | 968.7 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| 2wikimultihopqa (n=20) | Native Memory | token_f1=0.1045 | 33.45 | 202.6 ms | 972.6 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Selected Context | token_f1=0.144 | 396.3 | 186.9 ms | 982 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |
| combined (n=60) | Native Memory | token_f1=0.144 | 33.52 | 176.2 ms | 987.4 ms | Apple M4 Pro (Mac16,7), 48 GB | ENGINE_QUALIFIED |

## Native Memory qualification

Native Memory uses the same selector output as Selected Context. It is recommended only where the profile and engine tables say so.

| Workload | Selected native K/V tokens | Active detail | Peak memory | Completion cost vs Selected Context |
| --- | ---: | ---: | ---: | ---: |
| qasper | 370.6 | 52.12 MiB | NEEDS_RUN | 1.008x |
| hotpotqa | 376.3 | 52.92 MiB | NEEDS_RUN | 1.004x |
| 2wikimultihopqa | 341.5 | 48.02 MiB | NEEDS_RUN | 1.004x |
| combined | 362.8 | 51.02 MiB | NEEDS_RUN | 1.006x |

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
