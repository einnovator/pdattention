"""Build the executable, user-facing PRA-HF model-family notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "pra_hf_model_families.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    code(
        r'''
# Source import cell: replace this with `pip install pra-hf` when a package is published.
from pathlib import Path
import sys

DEMO_DIR = Path.cwd().resolve()
PROJECT_ROOT = DEMO_DIR.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if not (SOURCE_ROOT / "pra_hf").is_dir():
    raise RuntimeError(f"PRA-HF source package not found under {SOURCE_ROOT}")
sys.path.insert(0, str(SOURCE_ROOT))

from pra_hf import PRAConfig, PRAForCausalLM, PRARouter

print(f"Using PRA-HF source from: {SOURCE_ROOT}")
'''
    ),
    md(
        r'''
# PRA-HF from a User's Perspective

This notebook shows the same PRA workflow across three Hugging Face decoder families:

1. **Qwen 3**
2. **Llama**
3. **Gemma 3**

The default cells build tiny random Hugging Face models locally. They are **correctness and API
smokes**, not language-quality demonstrations. This keeps the complete notebook executable on a
CPU, without credentials or multi-gigabyte downloads. Near the end, opt-in examples show how the
same API maps to real pretrained repositories.

For each family we will:

- construct or load a native Hugging Face causal LM;
- wrap it with `PRAForCausalLM`;
- register URI-addressed references;
- route once from the query to a bounded subset of reference chunks;
- generate with selected layer-native K/V;
- inspect routing, memory, and native-operation statistics.
'''
    ),
    code(
        r'''
import platform
from types import SimpleNamespace

import torch
import transformers
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

torch.set_grad_enabled(False)
DEVICE = torch.device("cpu")  # Tiny offline examples are fastest and most portable on CPU.

print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "device": str(DEVICE),
})
'''
    ),
    md(
        r'''
## The PRA Mental Model

PRA keeps two representations with different jobs:

- A compact **routing index** ranks candidate chunks.
- The selected chunks' full **native K/V payload** is materialized for attention.

The gist is therefore an address, not a compressed replacement for the selected content. A normal
request follows this path:

```text
reference text -> bounded encoding blocks -> routing chunks/gists -> selected chunk identities
               -> selected layer-native K/V -> ordinary model-family attention
```

The model remains frozen in this notebook. With no trained router supplied, routing uses the
configured native hidden-state representation directly. A production application normally loads a
small model-matched `PRARouter` trained for its retrieval domain.
'''
    ),
    md(
        r'''
## Key Configuration Parameters

### Placement

| Parameter | Meaning |
|---|---|
| `routing_layer` | Layer whose attention-input hidden state represents the query and chunks. `-1` means the last eligible layer. |
| `consumption_layers` | Layers that may attend to selected memory K/V. More layers increase memory influence and transfer/materialization work. |
| `routing_representation` | The search representation. The public default is the position-independent attention-input hidden state. |

Negative layer IDs are resolved from the end of the decoder. Gemma is special: its local
sliding-window layers remain untouched, so PRA routing and consumption must use Gemma's native
global-attention layers.

### Chunking and selection

| Parameter | Meaning |
|---|---|
| `encoding_block_tokens` | Maximum reference tokens encoded together. It controls contextualization cost. |
| `chunk_tokens` | Addressable routing unit cut from encoded state. Smaller chunks improve addressability but create more candidates. |
| `chunk_overlap_tokens` | Shared tokens between adjacent routing chunks. It can reduce boundary misses at extra index/detail cost. |
| `selected_fraction` | Fraction of candidate chunks requested after ranking. It takes precedence over `top_k`. |
| `top_k` | Fixed candidate count used only when `selected_fraction=None`. |

Encoding granularity and routing granularity are deliberately separate: a source can be encoded in
larger contextual blocks while still being addressed through smaller routing chunks.

### Native-operation and memory budgets

| Parameter | Meaning |
|---|---|
| `native_operation_limit` | Maximum direct plus materialized K/V allowed in one model operation. |
| `max_direct_context` | Direct prompt tail retained in the native operation. Older prompt tokens can become the implicit `#__head` reference. |
| `max_materialized_tokens` | Upper bound for selected reference K/V admitted to attention. |
| `context_safety_reserve_tokens` | Tokens held back for generation/runtime bookkeeping. |
| `reference_device` | Persistent detail-K/V residency (`cpu` or `gpu`). CPU saves GPU memory but requires transfer. |
| `pin_reference_memory` / `non_blocking_transfer` | CUDA transfer controls; useful only with CPU-resident references and CUDA execution. |

The invariant is approximately:

```text
direct tokens + admitted memory tokens + safety reserve <= native operation limit
```
'''
    ),
    md(
        r'''
## Reusable Offline Helpers

The tokenizer below is intentionally simple: one deterministic ID per character. The models are
real Hugging Face `Qwen3ForCausalLM`, `LlamaForCausalLM`, and `Gemma3ForCausalLM` classes with tiny
random configurations. That is enough to exercise family adapters, native Q/K/V layouts, routing,
reference lifecycle, generation, and diagnostics.
'''
    ),
    code(
        r'''
class TinyTokenizer:
    """Minimal tokenizer implementing the methods used by the public PRA-HF API."""

    chat_template = None
    truncation_side = "left"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False, **_kwargs):
        values = [2 + (ord(char) % 93) for char in text]
        if add_special_tokens:
            values.insert(0, 1)
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{row['role']}: {row['content']}" for row in messages)


def make_tiny_hf_model(family: str):
    """Return a tiny native HF model while preserving each family's attention structure."""
    common = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=128,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=96,
        pad_token_id=0,
    )
    if family == "qwen3":
        config = Qwen3Config(num_hidden_layers=2, head_dim=8, **common)
        model = Qwen3ForCausalLM(config)
    elif family == "llama":
        config = LlamaConfig(num_hidden_layers=2, **common)
        model = LlamaForCausalLM(config)
    elif family == "gemma3":
        config = Gemma3TextConfig(
            num_hidden_layers=6,
            head_dim=8,
            sliding_window=16,
            sliding_window_pattern=3,
            **common,
        )
        model = Gemma3ForCausalLM(config)
    else:
        raise ValueError(f"Unsupported demo family: {family}")
    config._attn_implementation = "eager"
    return model.eval().to(DEVICE)


def make_demo_config(family: str) -> PRAConfig:
    """Use matched budgets while respecting Gemma's local/global schedule."""
    placement = (
        {} if family == "gemma3"
        else {"routing_layer": -1, "consumption_layers": (-2, -1)}
    )
    return PRAConfig(
        **placement,
        chunk_tokens=8,
        chunk_overlap_tokens=0,
        selected_fraction=0.25,
        top_k=4,
        encoding_block_tokens=16,
        max_direct_context=24,
        native_operation_limit=64,
        max_materialized_tokens=24,
        context_safety_reserve_tokens=0,
        reference_device="cpu",
    )
'''
    ),
    code(
        r'''
REFERENCES = {
    "kb://cities/lisbon": (
        "Lisbon is the capital of Portugal. The city stands beside the Tagus river."
    ),
    "kb://cities/porto": (
        "Porto is a city in northern Portugal. It is associated with the Douro river."
    ),
}
PROMPT = "Which city is Portugal's capital? Answer with one city."


def compact_stats(pra, result):
    stats = pra.stats()
    request = result.stats
    return {
        "family": stats["family"],
        "routing_layer": pra.routing_layer,
        "consumption_layers": list(pra.consumption_layers),
        "references": len(stats["references"]),
        "candidate_chunks": request["candidate_chunks"],
        "requested_chunks": request["requested_chunks"],
        "requested_chunk_fraction": round(request["requested_chunk_fraction"], 3),
        "materialized_kv_tokens": request["materialized_kv_tokens"],
        "materialized_kv_token_fraction": round(request["materialized_kv_token_fraction"], 3),
        "routing_index_bytes": stats["routing_index_bytes"],
        "resident_detail_kv_bytes": stats["resident_detail_kv_bytes"],
        "max_native_operation_tokens": stats["max_native_operation_tokens"],
        "native_limit_violations": stats["native_limit_violations"],
        "generated_token_ids": result.text,
    }


def run_family_session(family: str):
    torch.manual_seed(100 + {"qwen3": 1, "llama": 2, "gemma3": 3}[family])
    pra = PRAForCausalLM.from_model(
        make_tiny_hf_model(family),
        TinyTokenizer(),
        pra_config=make_demo_config(family),
    )
    handles = [pra.add_reference(uri, text=text) for uri, text in REFERENCES.items()]
    result = pra.generate(PROMPT, max_new_tokens=3, do_sample=False, return_details=True)
    return pra, handles, result, compact_stats(pra, result)
'''
    ),
    md(
        r'''
## Session 1: Qwen 3

Qwen uses native grouped-query attention (GQA) and RoPE. PRA captures and stores each selected
layer's native K/V without permanently expanding KV heads to the query-head count. In a typical
pretrained setup, use the last layer for routing and a conservative late band for memory
consumption.

The offline model has two layers, so both consume memory. For a real Qwen model the equivalent
configuration might use `routing_layer=-1` and `consumption_layers=range(-8, 0)`. Treat depth as a
model/application parameter: adding layers increases the opportunity to use memory, but also
increases K/V transfer and attention work.
'''
    ),
    code(
        r'''
qwen_pra, qwen_refs, qwen_result, qwen_stats = run_family_session("qwen3")
qwen_stats
'''
    ),
    md(
        r'''
### Reading the Qwen output

`candidate_chunks` is the number of addressable chunks across both references.
`requested_chunks` reflects the 25% selection policy. `materialized_kv_tokens` can be lower than
the requested logical total because the native-operation budgeter is authoritative. The random
model's decoded token IDs have no linguistic meaning; the important smoke assertions are finite
generation, bounded materialization, native family identification, and zero limit violations.
'''
    ),
    md(
        r'''
## Session 2: Llama

The Llama adapter exposes the same public API and preserves native GQA/RoPE tensors. The user-level
workflow does not change:

```python
pra = PRAForCausalLM.from_pretrained(model_id, routing_adapter=router_dir,
                                    pra_config=config, revision=revision)
pra.add_reference("docs://manual", text=manual)
answer = pra.generate(question)
```

The routing adapter must match the base model's hidden width and, for released artifacts, should
record the exact base revision. A router can be replaced only after `clear_references()`, because
already cached gists belong to the old routing geometry.
'''
    ),
    code(
        r'''
llama_pra, llama_refs, llama_result, llama_stats = run_family_session("llama")
llama_stats
'''
    ),
    md(
        r'''
### Choosing fraction versus fixed top-k

This session uses `selected_fraction=0.25`, which scales the requested set with reference size.
For a fixed-size service budget, set `selected_fraction=None` and choose `top_k`. The fraction is a
routing request, not permission to exceed `max_materialized_tokens` or the native-operation limit.
Those budgets remain final.
'''
    ),
    code(
        r'''
fraction_config = make_demo_config("llama")
fixed_k_config = PRAConfig(
    **{**fraction_config.to_dict(), "selected_fraction": None, "top_k": 3}
)
{
    "fraction_policy": fraction_config.selection_policy,
    "fixed_policy": fixed_k_config.selection_policy,
    "fixed_top_k": fixed_k_config.top_k,
}
'''
    ),
    md(
        r'''
## Session 3: Gemma 3

Gemma 3 alternates local sliding-window and global-attention layers. PRA preserves this host-model
contract: local layers stay native and only global layers are eligible for external memory.

The tiny six-layer configuration has global layers 2 and 5. The default public config resolves the
routing layer to 5 and intersects the late consumption band with `(2, 5)`. An explicit request for
a local layer raises an error instead of silently converting Gemma into a full-attention model.

For the official `google/gemma-3-1b-it`, global layers are 5, 11, 17, and 23; the validated
conservative placement routes at 23 and consumes at 17 and 23. Access to the official repository
requires accepting Google's Gemma terms on Hugging Face.
'''
    ),
    code(
        r'''
gemma_pra, gemma_refs, gemma_result, gemma_stats = run_family_session("gemma3")
gemma_stats
'''
    ),
    code(
        r'''
gemma_layer_types = list(gemma_pra.model.config.layer_types)
{
    "layer_types": gemma_layer_types,
    "global_layers": [i for i, kind in enumerate(gemma_layer_types) if kind == "full_attention"],
    "resolved_routing_layer": gemma_pra.routing_layer,
    "resolved_consumption_layers": list(gemma_pra.consumption_layers),
}
'''
    ),
    md(
        r'''
## Compare the Three Sessions

The table below is a mechanics comparison, not a quality benchmark. All models are random and use
the same references, prompt, routing fraction, chunk size, and native budget. Differences in
placement reflect each host architecture.
'''
    ),
    code(
        r'''
comparison = {
    row["family"]: {
        key: row[key]
        for key in (
            "routing_layer",
            "consumption_layers",
            "candidate_chunks",
            "requested_chunks",
            "materialized_kv_tokens",
            "routing_index_bytes",
            "resident_detail_kv_bytes",
            "max_native_operation_tokens",
            "native_limit_violations",
        )
    }
    for row in (qwen_stats, llama_stats, gemma_stats)
}
comparison
'''
    ),
    md(
        r'''
## Reference Lifecycle

References can be in-memory text, explicit `(URI, text)` pairs, or local text files. Their handles
are stable identities for removal. Clearing references also removes the implicit `#__head` memory
created when a prompt exceeds `max_direct_context`.

Disable/enable controls are useful for paired comparisons. Disabling PRA leaves the base model's
native attention path active; it does not delete cached references.
'''
    ),
    code(
        r'''
lifecycle = llama_pra
before = len(lifecycle.stats()["references"])
lifecycle.disable()
disabled = lifecycle.generate(PROMPT, max_new_tokens=1, do_sample=False, return_details=True)
lifecycle.enable()
lifecycle.remove_reference(llama_refs[0])
after_remove = len(lifecycle.stats()["references"])
lifecycle.clear_references()
after_clear = len(lifecycle.stats()["references"])

{
    "before": before,
    "disabled_generated_tokens": disabled.generated_tokens,
    "after_remove": after_remove,
    "after_clear": after_clear,
}
'''
    ),
    md(
        r'''
## Opt In to Real Pretrained Models

The next cell is deliberately disabled by default. Set `RUN_REMOTE_MODELS=True` only after choosing
one model, confirming its license/access conditions, and ensuring enough RAM/VRAM and disk space.

Suggested starting repositories:

| Family | Example repository | Notes |
|---|---|---|
| Qwen 3 | `Qwen/Qwen3-0.6B` | Public; use eager attention for the current adapter path. |
| Llama family | `HuggingFaceTB/SmolLM2-135M` | Small public Llama-family smoke. Official Meta Llama repositories are gated separately. |
| Gemma 3 | `google/gemma-3-1b-it` | Requires accepting Gemma terms; PRA uses only native global layers. |

Always pin a revision for research or production. Router metadata must identify the same base model
and revision. A router trained for one family or hidden width is not interchangeable with another.
'''
    ),
    code(
        r'''
RUN_REMOTE_MODELS = False
REMOTE_FAMILY = "qwen3"  # one of: qwen3, llama, gemma3
ROUTER_DIRECTORY = None  # Set to a model-matched PRARouter directory when available.

REMOTE_MODELS = {
    "qwen3": {"model": "Qwen/Qwen3-0.6B", "revision": None},
    "llama": {"model": "HuggingFaceTB/SmolLM2-135M", "revision": None},
    "gemma3": {
        "model": "google/gemma-3-1b-it",
        "revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    },
}

if RUN_REMOTE_MODELS:
    selected = REMOTE_MODELS[REMOTE_FAMILY]
    placement = (
        {"routing_layer": 23, "consumption_layers": (17, 23)}
        if REMOTE_FAMILY == "gemma3"
        else {"routing_layer": -1, "consumption_layers": tuple(range(-8, 0))}
    )
    remote_config = PRAConfig(
        **placement,
        chunk_tokens=32,
        encoding_block_tokens=128,
        selected_fraction=0.20,
        max_direct_context=128,
        native_operation_limit=512,
        max_materialized_tokens=128,
        context_safety_reserve_tokens=4,
        reference_device="cpu",
        pin_reference_memory=torch.cuda.is_available(),
        non_blocking_transfer=torch.cuda.is_available(),
    )
    model_kwargs = {
        "revision": selected["revision"],
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
    }
    remote_pra = PRAForCausalLM.from_pretrained(
        selected["model"],
        routing_adapter=ROUTER_DIRECTORY,
        pra_config=remote_config,
        **model_kwargs,
    )
    if torch.cuda.is_available():
        remote_pra.model.to("cuda")
    remote_pra.add_reference("kb://portugal", text=REFERENCES["kb://cities/lisbon"])
    remote_result = remote_pra.generate(PROMPT, max_new_tokens=16, return_details=True)
    print(remote_result.text)
    print(remote_pra.stats())
else:
    print("Remote loading skipped. The three offline family sessions above were executed.")
'''
    ),
    md(
        r'''
## Adding a Trained Router

A trained router supplies a compact semantic geometry while the attention payload remains native
model K/V:

```python
pra = PRAForCausalLM.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    routing_adapter=ROUTER_DIRECTORY,
    pra_config=config,
)
```

Important checks before deployment:

1. Router `base_model` and revision match the loaded model.
2. Router input width matches `model.config.hidden_size`.
3. Routing layer and feature representation match training metadata.
4. Chunk size and evidence mapping match the evaluated protocol.
5. Recall-sparsity metrics use the intended evidence definition.
6. `native_limit_violations` remains zero under representative requests.
7. Systems metrics distinguish routing-index bytes from materialized detail-K/V bytes.

Do not infer answer quality from routing recall alone. Retrieval, context composition, and the
frozen model's ability to use selected memory are separate measurements.
'''
    ),
    md(
        r'''
## Practical Starting Points

- Start with `chunk_tokens=32`, `encoding_block_tokens=128`, and no overlap.
- Use `selected_fraction=0.20` while studying recall-sparsity curves; switch to fixed `top_k` when a
  service needs a fixed routing request.
- Keep detail K/V on CPU when GPU memory is scarce, then measure transfer latency.
- Begin with one or two late eligible consumption layers. Expand only with causal-use evidence.
- For Gemma, preserve the native local/global schedule.
- Pin base and tokenizer revisions and package router metadata alongside results.
- Inspect `stats()` after real requests; budgets and zero native-limit violations matter more than
  nominal configuration alone.

The core user workflow is intentionally small: load, configure, add references, generate, inspect.
The careful work is choosing a model-matched router and validating retrieval and downstream use for
the application.
'''
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
