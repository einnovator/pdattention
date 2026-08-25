"""Build the executable unified PRA runtime notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "pra_runtime_productization.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    code(
        r'''
from pathlib import Path
import sys

DEMO_DIR = Path.cwd().resolve()
PROJECT_ROOT = DEMO_DIR.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from pra_hf import (
    DiscoveryRequest,
    ExecutionAuthorization,
    KVInterval,
    KVMaterializer,
    MaterializationPlan,
    NativeKV,
    PersistentResourceIndex,
    PRAConfig,
    PRARuntime,
    PRARuntimeConfig,
    ResourceDiscoveryEngine,
    VLLMThinBackend,
    runtime_capabilities,
)

print(f"Using source package: {SOURCE_ROOT}")
'''
    ),
    md(
        r'''
# One PRA SDK from Memory to Tools

This notebook uses the unified runtime interface introduced for Paper 4.5. The interface layers
systems controls over the existing Paper 2 model API rather than replacing it:

```text
PRARuntime
  -> PRAForCausalLM              model loading, routing, generation
  -> ExternalMemoryManager       authenticated cold/warm/hot resources
  -> ResourceDiscoveryEngine     typed capability identities
  -> SafeToolExecutor            schema and host-authorization boundary
```

The examples are deliberately offline. A tiny random Llama exercises the real Hugging Face/PRA
integration, while deterministic in-memory tools demonstrate execution safety without external
side effects. Generated language is therefore meaningless; lifecycle and tensor behavior are the
things being tested.
'''
    ),
    code(
        r'''
import platform
from types import SimpleNamespace

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

torch.set_grad_enabled(False)
print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    **runtime_capabilities(),
})
'''
    ),
    md(
        r'''
## Configure semantics and physical execution separately

`PRAConfig` controls which memory is logically selected and consumed. `PRARuntimeConfig` controls
how selected K/V is packed, cached, compiled, and handed to a backend. Changing the runtime layout
or compiler mode must preserve the selected identities and generated semantics.
'''
    ),
    code(
        r'''
runtime_config = PRARuntimeConfig(
    pra=PRAConfig(
        routing_layer=-1,
        consumption_layers=(-1,),
        chunk_tokens=8,
        selected_fraction=0.5,
        max_direct_context=32,
        native_operation_limit=64,
        max_materialized_tokens=16,
        context_safety_reserve_tokens=0,
        encoding_block_tokens=32,
    ),
    backend="huggingface",
    compilation="eager",
    kv_layout="layer_major",
    cache_max_bytes=1 << 20,
)
runtime_config.to_dict()
'''
    ),
    md(
        r'''
## Load once, then reuse references

`PRARuntime.from_model` wraps an already loaded model. In a deployed application,
`PRARuntime.from_pretrained` performs the equivalent Hugging Face load. Stable URIs let logs,
caches, authorization, and selected K/V refer to the same resource identity.
'''
    ),
    code(
        r'''
class TinyTokenizer:
    all_special_ids = (0, 1)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False, **_kwargs):
        values = [2 + (ord(char) % 61) for char in text]
        if add_special_tokens:
            values.insert(0, 1)
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        return [str(int(value)) for value in token_ids]


model_config = LlamaConfig(
    vocab_size=67,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    bos_token_id=1,
    eos_token_id=66,
    pad_token_id=0,
)
model_config._attn_implementation = "eager"
model = LlamaForCausalLM(model_config).eval()
runtime = PRARuntime.from_model(model, TinyTokenizer(), runtime_config=runtime_config)
handle = runtime.add_reference("memory://demo/facts", text="Paris is the capital of France.")
result = runtime.generate("Question: capital of France? Answer:", max_new_tokens=1, return_details=True)
{
    "reference": handle,
    "generated_text": result.text,
    "selected": result.stats["selected"],
    "runtime": runtime.inspect(),
}
'''
    ),
    md(
        r'''
## Inspect physical native-K/V packing

The portable materializer consumes exact half-open intervals. Native K/V uses shape
`[batch, kv_heads, tokens, head_dim]`; grouped-query heads are not expanded in storage. Overlapping
intervals are merged before the hard token budget is applied.
'''
    ),
    code(
        r'''
key = torch.arange(1 * 2 * 16 * 4, dtype=torch.float32).reshape(1, 2, 16, 4)
source = NativeKV(key, key + 1000)
plan = MaterializationPlan.build(
    [
        KVInterval("memory://demo/facts", 1, 0, 8),
        KVInterval("memory://demo/facts", 1, 4, 12),
    ],
    max_tokens=10,
)
packed = KVMaterializer().materialize({("memory://demo/facts", 1): source}, plan)
{
    "plan": plan,
    "packed_shape": tuple(packed.layers[1].key.shape),
    "physical_bytes": packed.physical_bytes,
    "transfer_bytes": packed.transfer_bytes,
}
'''
    ),
    md(
        r'''
## Discovering a tool never authorizes it

Paper 6.5's typed resources now share the SDK. Discovery returns stable URIs. The model may then
propose a JSON call, but a host-provided `ExecutionAuthorization` independently decides whether
that disclosed identity can execute. Write and destructive permissions are separate flags.
'''
    ),
    code(
        r'''
from data.agent_workflows import realistic_tool_catalog, workflow_executor, workflow_tasks

resources = realistic_tool_catalog()
task = workflow_tasks()[0]
runtime.discovery = ResourceDiscoveryEngine(
    PersistentResourceIndex(resources),
    select_threshold=0.0,
    ask_threshold=0.0,
    margin_threshold=0.0,
)
runtime.executor = workflow_executor(resources, task)
trace = runtime.discover_resources(
    DiscoveryRequest(query="search documents", tenant_id="paper6_5", top_k=1)
)
search = next(resource for resource in resources if resource.name == "search_document")
proposal = '<tool_call>{"name":"search_document","arguments":{"title":"quarterly"}}</tool_call>'
denied = runtime.execute_tool(
    proposal,
    selected_uris=(search.uri,),
    authorization=ExecutionAuthorization(frozenset()),
    call_id="denied",
)
accepted = runtime.execute_tool(
    proposal,
    selected_uris=(search.uri,),
    authorization=ExecutionAuthorization(frozenset((search.uri,))),
    call_id="accepted",
)
{
    "discovered": trace.selected_uris,
    "denied_reason": denied.reason,
    "accepted": accepted.executed,
    "typed_observation": accepted.observation.uri,
}
'''
    ),
    md(
        r'''
## Thin serving-engine handoff

The Paper 4.5 vLLM boundary intentionally keeps semantic routing outside the scheduler. It passes
only selected stable identities, materialized-token accounting, and ordinary request metadata.
This notebook does not claim a measured vLLM integration when vLLM is not installed.
'''
    ),
    code(
        r'''
handoff = VLLMThinBackend().prepare(
    "Question: capital of France?",
    selected_uris=(handle.uri,),
    materialized_tokens=result.stats["materialized_kv_tokens"],
    metadata={"kv_layout": runtime_config.kv_layout},
)
handoff
'''
    ),
    md(
        r'''
## Production checklist

The SDK now supports versioned config artifacts, model loading, direct and external-memory
lifecycle hooks, stable typed resources, safe tool execution, physical K/V accounting, cache
inspection, runtime capability reports, and a benchmark command. Triton, custom CUDA, and deep
retrieval-aware scheduler integration remain later gates and must be reported as unsupported until
they are actually installed and measured.
'''
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
