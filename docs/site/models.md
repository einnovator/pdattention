# Model Support

PRA support has two distinct boundaries. **Selected Context** works with any
model that accepts ordinary text through a supported runtime or endpoint; it
does not require an attention adapter. **Native Memory** changes model execution
and therefore requires a known structural mapping plus model-specific validation.

| Family | Selected Context | Native Memory | Structural adapter | Evidence |
| --- | --- | --- | --- | --- |
| [Qwen](#qwen) | ✅ Available | ℹ️ Validated mapping | **Optional** | Qwen3-8B, 14B, and 32B 4-bit checkpoints have exact-identity paired MLX Native Memory qualification: 15/15 output parity, unchanged F1, and 89.1% fewer visible tokens. Qwen3-4B 8-bit adds exact five-seed routing evidence: learned QASPER R@20% improves by 0.120, while HotpotQA declines by 0.227, so generic routing remains default. Other 6-bit/8-bit bundles have runtime smoke only. Matched 1.5B general-instruction and code-instruction HF checkpoints show the same dataset-dependent routing pattern. |
| [Llama](#llama) | ✅ Available | ℹ️ Validated mapping | **Optional** | Llama-3.1-8B 4-bit has a five-seed, held-out MLX routing comparison. The Llama-3.2-1B 8-bit bundle has exact structural validation only. Learned routing improves QASPER MRR but reduces HotpotQA recall on the measured 4-bit identity, so generic routing remains the default. |
| [Gemma 3 text](#gemma3) | ✅ Available | 🧪 Partial topology | **Required for native production** | Gemma-3-1B 4-bit has a five-seed, held-out MLX comparison under its mixed sliding/global topology. Its 8-bit bundle has exact structural validation only. Learned routing helps QASPER and combined MRR on the measured 4-bit identity, while HotpotQA recall remains mixed. |
| [Other Hugging Face causal decoders](#other-hf) | ✅ Available | ⏳ Qualification pending | **Required** | No family-wide native claim is made for unregistered architectures. |
| [Models behind ordinary API endpoints](#endpoint-only) | ✅ Available | 🧪 Engine dependent | **Not required for Selected Context** | Selected Context is the portable cross-engine path. |

**Key:** ✅ available/validated · 🧪 partial or engine-dependent · ⏳ qualification pending.

## Discover published bundles

List the immutable catalog packaged with this PRA release, or search current
Hugging Face metadata without downloading model or bundle weights:

```bash
pra hf list
pra hf list --family qwen --engine mlx
pra hf search llama --author EInnovator
```

The list command is the authority for `-a auto`. Live search is broader and
marks whether each result is registry-backed and automatically resolvable.
Community search results remain explicit opt-in artifacts.

## Qwen { #qwen }

**Model types:** `qwen2, qwen3`  
**Adapter:** Optional

The SDK includes Qwen2/Qwen3 structural mappings. Export a declarative adapter when the deployment needs a pinned, reviewable artifact.

**Known examples and published bundles**

| Model | PRA bundle/model card | Status | Validated engines | Recommended mode | Last qualification |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-1.5B-Instruct` | [EInnovator/pra-qwen2-5-1-5b-instruct](https://huggingface.co/EInnovator/pra-qwen2-5-1-5b-instruct) | Controlled | HF routing qualification; portable Selected Context | Generic balanced; learned router only for matched QASPER | 2026-09-03 |
| `Qwen/Qwen2.5-1.5B-Instruct (bitsandbytes 8-bit runtime)` | [EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit](https://huggingface.co/EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit) | Controlled | HF bitsandbytes int8 CUDA measured; Native Memory candidate failed quality/parity gates | Selected Context with BALANCED | 2026-09-03 |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | [EInnovator/pra-qwen2-5-coder-1-5b-instruct](https://huggingface.co/EInnovator/pra-qwen2-5-coder-1-5b-instruct) | Controlled | HF routing qualification; portable Selected Context | Generic balanced; learned router only for matched QASPER | 2026-09-03 |
| `Qwen/Qwen3-0.6B` | [EInnovator/pra-qwen3-0.6b](https://huggingface.co/EInnovator/pra-qwen3-0.6b) | Research/reference | HF Native Memory; portable Selected Context | Selected Context; qualify Native Memory locally | 2026-09-01 |
| `Qwen/Qwen3-1.7B` | Not published | NOT_MEASURED | NOT_MEASURED | Inspect and qualify locally | NOT_MEASURED |
| `mlx-community/Qwen3-4B-4bit` | [EInnovator/pra-qwen3-4b-mlx-4bit](https://huggingface.co/EInnovator/pra-qwen3-4b-mlx-4bit) | Controlled | MLX routing qualification; portable Selected Context | Generic balanced; learned router only for matched QASPER | 2026-09-01 |
| `mlx-community/Qwen3-4B-8bit` | [EInnovator/pra-qwen3-4b-mlx-8bit](https://huggingface.co/EInnovator/pra-qwen3-4b-mlx-8bit) | Engine qualified | MLX Native Memory: 60/60 exact paired outputs; 91.5% fewer visible tokens | Native Memory with BALANCED; learned router remains opt-in | 2026-09-03 |
| `mlx-community/Qwen3-8B-4bit` | [EInnovator/pra-qwen3-8b-mlx-4bit](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-4bit) | Engine qualified | MLX paired natural-QA Native Memory qualification | Native Memory with BALANCED | 2026-09-01 |
| `mlx-community/Qwen3-8B-6bit` | [EInnovator/pra-qwen3-8b-mlx-6bit](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-6bit) | Engine-qualified | MLX paired natural-QA Native Memory qualification | Native Memory with BALANCED | 2026-09-03 |
| `mlx-community/Qwen3-8B-8bit` | [EInnovator/pra-qwen3-8b-mlx-8bit](https://huggingface.co/EInnovator/pra-qwen3-8b-mlx-8bit) | Engine-qualified | MLX paired natural-QA Native Memory qualification | Native Memory with BALANCED | 2026-09-03 |
| `mlx-community/Qwen3-14B-4bit` | [EInnovator/pra-qwen3-14b-mlx-4bit](https://huggingface.co/EInnovator/pra-qwen3-14b-mlx-4bit) | Engine qualified | MLX paired natural-QA Native Memory qualification | Native Memory with BALANCED | 2026-09-01 |
| `mlx-community/Qwen3-14B-8bit` | [EInnovator/pra-qwen3-14b-mlx-8bit](https://huggingface.co/EInnovator/pra-qwen3-14b-mlx-8bit) | Smoke | MLX 8-bit load/generation smoke; end-task qualification pending | Selected Context with BALANCED | 2026-09-03 |
| `mlx-community/Qwen3-32B-4bit` | [EInnovator/pra-qwen3-32b-mlx-4bit](https://huggingface.co/EInnovator/pra-qwen3-32b-mlx-4bit) | Engine qualified | MLX paired natural-QA Native Memory qualification | Native Memory with BALANCED | 2026-09-01 |

**Inspect and launch**

```bash
pra inspect Qwen/Qwen2.5-1.5B-Instruct --engine hf --pra-bundle auto
pra model validate Qwen/Qwen2.5-1.5B-Instruct --suite smoke
pra serve Qwen/Qwen2.5-1.5B-Instruct --engine hf --mode auto --profile recommended --pra-bundle auto
```

**Evidence boundary**

Qwen3-8B, 14B, and 32B 4-bit checkpoints have exact-identity paired MLX Native Memory qualification: 15/15 output parity, unchanged F1, and 89.1% fewer visible tokens. Qwen3-4B 8-bit adds exact five-seed routing evidence: learned QASPER R@20% improves by 0.120, while HotpotQA declines by 0.227, so generic routing remains default. Other 6-bit/8-bit bundles have runtime smoke only. Matched 1.5B general-instruction and code-instruction HF checkpoints show the same dataset-dependent routing pattern.

**Limitations**

- Model revision, tokenizer, quantization, engine, and profile still require qualification together.
- The code-tuned checkpoint was compared on matched QASPER/HotpotQA routing, not code retrieval or coding-agent task success.

## Llama { #llama }

**Model types:** `llama`  
**Adapter:** Optional

The SDK includes the conventional Llama decoder mapping. A declarative adapter is optional for discovery and recommended for pinned releases.

**Known examples and published bundles**

| Model | PRA bundle/model card | Status | Validated engines | Recommended mode | Last qualification |
| --- | --- | --- | --- | --- | --- |
| `unsloth/Llama-3.2-1B` | Not published | NOT_MEASURED | NOT_MEASURED | Inspect and qualify locally | NOT_MEASURED |
| `meta-llama/Llama-3.2-1B-Instruct` | Not published | NOT_MEASURED | NOT_MEASURED | Inspect and qualify locally | NOT_MEASURED |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | [EInnovator/pra-llama3-2-1b-mlx-8bit](https://huggingface.co/EInnovator/pra-llama3-2-1b-mlx-8bit) | Engine qualified | MLX Native Memory: 60/60 exact Selected Context/Native Memory outputs; 91.5% fewer visible tokens | Native Memory with BALANCED | 2026-09-03 |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | [EInnovator/pra-llama3-1-8b-mlx-4bit](https://huggingface.co/EInnovator/pra-llama3-1-8b-mlx-4bit) | Controlled | MLX routing qualification; portable Selected Context | Generic balanced; learned router only for matched QASPER | 2026-09-01 |

**Inspect and launch**

```bash
pra inspect unsloth/Llama-3.2-1B --engine hf
pra model validate unsloth/Llama-3.2-1B --suite smoke
pra serve unsloth/Llama-3.2-1B --engine hf --mode auto --profile recommended
```

**Evidence boundary**

Llama-3.1-8B 4-bit has a five-seed, held-out MLX routing comparison. The Llama-3.2-1B 8-bit bundle has exact structural validation only. Learned routing improves QASPER MRR but reduces HotpotQA recall on the measured 4-bit identity, so generic routing remains the default.

**Limitations**

- Instruction tuning and quantized derivatives are separate qualification identities.

## Gemma 3 text { #gemma3 }

**Model types:** `gemma3_text`  
**Adapter:** Required for native production

A built-in mapping discovers Gemma 3 text attention, but its heterogeneous topology must be exported, reviewed, and validated for the exact revision before Native Memory is promoted.

**Known examples and published bundles**

| Model | PRA bundle/model card | Status | Validated engines | Recommended mode | Last qualification |
| --- | --- | --- | --- | --- | --- |
| `google/gemma-3-1b-it` | Not published | NOT_MEASURED | NOT_MEASURED | Inspect and qualify locally | NOT_MEASURED |
| `mlx-community/gemma-3-1b-it-4bit` | [EInnovator/pra-gemma3-1b-mlx-4bit](https://huggingface.co/EInnovator/pra-gemma3-1b-mlx-4bit) | Controlled | MLX mixed/sliding routing qualification; portable Selected Context | Generic balanced; learned router for matched QASPER or validated mixed workloads | 2026-09-01 |
| `mlx-community/gemma-3-1b-it-8bit` | [EInnovator/pra-gemma3-1b-mlx-8bit](https://huggingface.co/EInnovator/pra-gemma3-1b-mlx-8bit) | Controlled | MLX Native Memory: 60 measured cases; 3/60 exact outputs and token-F1 delta -0.0036 | Selected Context with BALANCED | 2026-09-03 |

**Inspect and launch**

```bash
pra inspect google/gemma-3-1b-it --engine hf
pra model adapt google/gemma-3-1b-it -o .pra/adapters/gemma3
pra model validate google/gemma-3-1b-it --adapter .pra/adapters/gemma3 --suite standard
```

**Evidence boundary**

Gemma-3-1B 4-bit has a five-seed, held-out MLX comparison under its mixed sliding/global topology. Its 8-bit bundle has exact structural validation only. Learned routing helps QASPER and combined MRR on the measured 4-bit identity, while HotpotQA recall remains mixed.

**Limitations**

- Sliding/global attention topology and multimodal wrappers require explicit review.

## Other Hugging Face causal decoders { #other-hf }

**Model types:** `unregistered`  
**Adapter:** Required

Selected Context needs no attention adapter. Native Memory requires a generated declarative mapping or reviewed Python plugin plus the complete validation ladder.

**Known examples and published bundles**

| Model | PRA bundle/model card | Status | Validated engines | Recommended mode | Last qualification |
| --- | --- | --- | --- | --- | --- |
| `Any AutoConfig-compatible causal decoder` | Not published | NOT_MEASURED | NOT_MEASURED | Inspect and qualify locally | NOT_MEASURED |

**Inspect and launch**

```bash
pra model inspect ORG/MODEL
pra model adapt ORG/MODEL -o .pra/adapters/model
pra model validate ORG/MODEL --adapter .pra/adapters/model --suite standard
```

**Evidence boundary**

No family-wide native claim is made for unregistered architectures.

**Limitations**

- Projection-name discovery alone does not establish mask, position, cache, or generation correctness.

## Models behind ordinary API endpoints { #endpoint-only }

**Model types:** `openai-compatible`  
**Adapter:** Not required for Selected Context

The gateway can select and render context for an ordinary endpoint without modifying the model. Native modes require an explicit backend capability handshake and model qualification.

**Known examples and published bundles**

| Model | PRA bundle/model card | Status | Validated engines | Recommended mode | Last qualification |
| --- | --- | --- | --- | --- | --- |
| `vLLM, SGLang, Ollama, llama.cpp, OpenVINO or custom OpenAI-compatible endpoints` | Not published | NOT_MEASURED | NOT_MEASURED | Inspect and qualify locally | NOT_MEASURED |

**Inspect and launch**

```bash
pra gateway serve --mode selected-context --backend custom --backend-url http://127.0.0.1:8000/v1
```

**Evidence boundary**

Selected Context is the portable cross-engine path.

**Limitations**

- Do not infer detached Native Memory from prefix caching or a paged K/V implementation.

## Adapter decision

1. Start with `pra inspect MODEL --engine ENGINE`.
2. If Selected Context is the target, no model structural adapter is required.
3. For a built-in family, run `pra model validate`; exporting an adapter pins the mapping.
4. For a partial or unknown family, run `pra model adapt` and the full validation ladder.
5. Promote Native Memory only after quality, geometry, lifecycle, and economics pass
   for the exact model revision, tokenizer, quantization, engine, and hardware.

_Generated from the model registry; evidence current through 2026-09-03._
