# Model Support

PRA support has two distinct boundaries. **Selected Context** works with any
model that accepts ordinary text through a supported runtime or endpoint; it
does not require an attention adapter. **Native Memory** changes model execution
and therefore requires a known structural mapping plus model-specific validation.

| Family | Selected Context | Native Memory | Structural adapter | Evidence |
| --- | --- | --- | --- | --- |
| [Qwen](#qwen) | ✅ Available | ℹ️ Validated mapping | **Optional** | Qwen/Qwen3-0.6B has checked-in model-backed profile rows; larger Qwen configurations have engine-specific evidence. |
| [Llama](#llama) | ✅ Available | ℹ️ Validated mapping | **Optional** | unsloth/Llama-3.2-1B has checked-in model-backed profile rows. |
| [Gemma 3 text](#gemma3) | ✅ Available | 🧪 Partial topology | **Required for native production** | google/gemma-3-1b-it has checked-in model-backed profile rows; native deployment remains topology specific. |
| [Other Hugging Face causal decoders](#other-hf) | ✅ Available | ⏳ Qualification pending | **Required** | No family-wide native claim is made for unregistered architectures. |
| [Models behind ordinary API endpoints](#endpoint-only) | ✅ Available | 🧪 Engine dependent | **Not required for Selected Context** | Selected Context is the portable cross-engine path. |

**Key:** ✅ available/validated · 🧪 partial or engine-dependent · ⏳ qualification pending.

## Qwen { #qwen }

**Model types:** `qwen2, qwen3`  
**Adapter:** Optional

The SDK includes Qwen2/Qwen3 structural mappings. Export a declarative adapter when the deployment needs a pinned, reviewable artifact.

**Known examples**

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `mlx-community/Qwen3-4B-4bit`

**Inspect and launch**

```bash
pra inspect Qwen/Qwen3-1.7B --engine hf
pra model validate Qwen/Qwen3-1.7B --suite smoke
pra serve Qwen/Qwen3-1.7B --engine hf --mode auto --profile recommended
```

**Evidence boundary**

Qwen/Qwen3-0.6B has checked-in model-backed profile rows; larger Qwen configurations have engine-specific evidence.

**Limitations**

- Model revision, tokenizer, quantization, engine, and profile still require qualification together.

## Llama { #llama }

**Model types:** `llama`  
**Adapter:** Optional

The SDK includes the conventional Llama decoder mapping. A declarative adapter is optional for discovery and recommended for pinned releases.

**Known examples**

- `unsloth/Llama-3.2-1B`
- `meta-llama/Llama-3.2-1B-Instruct`

**Inspect and launch**

```bash
pra inspect unsloth/Llama-3.2-1B --engine hf
pra model validate unsloth/Llama-3.2-1B --suite smoke
pra serve unsloth/Llama-3.2-1B --engine hf --mode auto --profile recommended
```

**Evidence boundary**

unsloth/Llama-3.2-1B has checked-in model-backed profile rows.

**Limitations**

- Instruction tuning and quantized derivatives are separate qualification identities.

## Gemma 3 text { #gemma3 }

**Model types:** `gemma3_text`  
**Adapter:** Required for native production

A built-in mapping discovers Gemma 3 text attention, but its heterogeneous topology must be exported, reviewed, and validated for the exact revision before Native Memory is promoted.

**Known examples**

- `google/gemma-3-1b-it`
- `mlx-community/gemma-3-1b-it-4bit`

**Inspect and launch**

```bash
pra inspect google/gemma-3-1b-it --engine hf
pra model adapt google/gemma-3-1b-it -o .pra/adapters/gemma3
pra model validate google/gemma-3-1b-it --adapter .pra/adapters/gemma3 --suite standard
```

**Evidence boundary**

google/gemma-3-1b-it has checked-in model-backed profile rows; native deployment remains topology specific.

**Limitations**

- Sliding/global attention topology and multimodal wrappers require explicit review.

## Other Hugging Face causal decoders { #other-hf }

**Model types:** `unregistered`  
**Adapter:** Required

Selected Context needs no attention adapter. Native Memory requires a generated declarative mapping or reviewed Python plugin plus the complete validation ladder.

**Known examples**

- `Any AutoConfig-compatible causal decoder`

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

**Known examples**

- `vLLM, SGLang, Ollama, llama.cpp, OpenVINO or custom OpenAI-compatible endpoints`

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

_Generated from the model registry; evidence current through 2026-09-01._
