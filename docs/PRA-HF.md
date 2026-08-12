# PRA-HF

PRA-HF retrofits supported frozen causal language models with bounded,
URI-addressed native-K/V memory. A compact semantic index selects parent chunks;
only selected post-RoPE layer-native K/V enters attention.

## Install

```bash
pip install -e .
```

Research and router-training dependencies are optional:

```bash
pip install -e ".[research]"
```

## Quick Start

```python
import torch
from pra_hf import PRAConfig, PRAForCausalLM

model = PRAForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    routing_adapter="artifacts/pra_hf/routers/smollm2-135m-qasper-d128",
    pra_config=PRAConfig(selected_fraction=0.20),
    torch_dtype=torch.float16,
)
model.model.to("cuda")
reference = model.add_reference_file("paper.txt")
result = model.generate("What is the main finding?", return_details=True)
print(result.text)
print(result.stats["materialized_kv_token_fraction"])
model.remove_reference(reference)
```

`selected_fraction` takes precedence over `top_k`. It controls the number of
parent chunks requested from the complete exact ranking. The hard native-context
budget can materialize fewer chunks; request statistics report both fractions.

## Supported Families

PRA-HF v0.2.0rc1 supports Qwen2/Qwen2.5/Qwen3, Llama-family, and Gemma 3 eager attention.
The disabled path delegates to the original attention module exactly. The enabled
path preserves each family's projections, RoPE, native K/V head count, cache
updates, masks, eager kernel, and output projection.
For Gemma 3, PRA is restricted to native global-attention layers; local sliding-window
layers and their hybrid-cache semantics remain untouched.

The release routers target:

| Router | Base model | Parameters | Training data |
|---|---|---:|---|
| `qwen3-0.6b-qasper-d128` | `Qwen/Qwen3-0.6B` | 262,144 | QASPER |
| `smollm2-135m-qasper-d128` | `HuggingFaceTB/SmolLM2-135M` | 147,456 | QASPER |
| `gemma3-1b-qasper-d128` | `google/gemma-3-1b-it` | 294,912 | QASPER |

Meta Llama 3.2-1B uses the same thin Llama adapter, but its official weights are
license-gated and are not redistributed by PRA-HF. Paper 2's pinned validation
runner is:

```powershell
hf auth login
python -m experiments.paper2_hf.llama.run_llama32_1b
```

The runner refuses to substitute another Llama-family checkpoint. A distributable
Meta router is published only after the official checkpoint passes exact parity,
native GQA/RoPE, bounded-reference, five-seed routing, systems, and causal-use gates.

Router directories contain `adapter_model.pt`, `config.json`, and a model card.
They never contain base-model weights.

## References And Long Prompts

```python
model.add_reference("plain text")
model.add_reference("doc://manual", text="plain text")
model.add_reference_file("manual.txt")
model.clear_references()
```

Prompts longer than `max_direct_context` publish their displaced prefix as the
implicit `#__head` reference. Encoding blocks and every attention operation remain
within `native_operation_limit`; references can reside on CPU and selected K/V is
transferred per request.

## CLI

```bash
pra-hf inspect HuggingFaceTB/SmolLM2-135M
pra-hf router inspect artifacts/pra_hf/routers/smollm2-135m-qasper-d128
pra-hf ask HuggingFaceTB/SmolLM2-135M "What is the conclusion?" \
  --routing-adapter artifacts/pra_hf/routers/smollm2-135m-qasper-d128 \
  --reference paper.txt
```

`pra-hf router train` accepts frozen train and validation feature files. Router
evaluation automatically emits `R@5%`, `R@10%`, `R@20%`, `R@30%`, `f70`, `f80`,
`f90`, `AUC0-30`, and fixed `R@3/8/16`, including exact K/V-token fractions.
MRR is retained as a first-relevant-evidence diagnostic.

## Current Limits

- The v1 implementation uses exact Python/Torch routing, not ANN search or a
  serving-optimized kernel.
- QASPER routing recall is useful but not yet high at 10--20% selection.
- Evidence retrieval does not guarantee that a frozen decoder will use the memory
  to produce a correct answer. The release evaluation reports this gap directly.
- Flash Attention, SDPA, vLLM serving, recurrent routing, semantic-D positioning,
  multi-layer rerouting, and learned materialization are deferred.
