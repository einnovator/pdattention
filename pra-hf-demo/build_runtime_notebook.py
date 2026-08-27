"""Build the executable, comprehensive Paper 4.5 PRA runtime notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "pra_runtime_productization.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    md(
        r'''
# PRA Runtime Productization: All Mechanisms

This is the Paper 4.5 systems notebook. It follows selected memory from a stable reference URI to
physical native K/V, cache residency, measurement, typed resource disclosure, safe execution, and
a thin serving-engine handoff.

The notebook is deliberately offline. A tiny random Llama exercises the real Hugging Face/PRA
adapter, while deterministic in-memory resources exercise lifecycle and safety boundaries. The
generated language is not meaningful; tensor parity, state transitions, accounting, and API
behavior are the objects under test.
'''
    ),
    md(
        r'''
## How this differs from the Paper 2 model-family demo

The two notebooks are complementary, not replacements.

| Question | `pra_hf_model_families.ipynb` (Paper 2) | This notebook (Paper 4.5) |
|---|---|---|
| Primary concern | Does PRA attach to supported HF model families? | Does sparse selection become a controllable physical runtime? |
| Models exercised | Qwen 3, Llama, and Gemma 3 sessions | One tiny Llama, keeping attention on systems mechanisms |
| References | Direct text references through `PRAForCausalLM` | Direct references plus authenticated external cold/warm/hot resources |
| K/V internals | Model-level routing and generation statistics | Exact intervals, budgets, GQA/MQA shape, layouts, transfer and temporary bytes |
| Runtime state | Per-model examples | Versioned config, sessions, LRU reuse, eviction, and stage profiler |
| Tools | Outside scope | Typed discovery, capability-graph disclosure, and independent execution authority |
| Serving | Ordinary HF generation | Capability gates and scheduler-unaware vLLM handoff |

Use the Paper 2 notebook to validate a model family. Use this notebook to understand and reproduce
the unified SDK and the boundary between logical selection and physical execution.
'''
    ),
    code(
        r'''
from pathlib import Path
import json
import platform
import sys
import tempfile
from types import SimpleNamespace

DEMO_DIR = Path.cwd().resolve()
PROJECT_ROOT = DEMO_DIR.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from pra_hf import (
    AgentConfig,
    InMemorySessionService,
    AuthContext,
    CapabilityEncodingPolicy,
    CapabilitySDK,
    ContextPolicy,
    CursorAction,
    DiscoveryRequest,
    EncodingContext,
    ExecutionAuthorization,
    ExternalMemoryManager,
    HotMemoryHandle,
    KVInterval,
    KVMaterializer,
    MaterializationPlan,
    NativeEncoding,
    NativeKV,
    PackedNativeKVStore,
    PersistentResourceIndex,
    PRAConfig,
    PRAAgent,
    PRAAgentConfig,
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAGateway,
    PRAWireRequest,
    PRAWireResource,
    PRARuntime,
    PRARuntimeConfig,
    ResolverRegistry,
    ResourceDiscoveryEngine,
    ResourceStat,
    RecordCapabilities,
    RecordType,
    RecordViewName,
    RuntimeKVCache,
    RuntimeProfiler,
    SafeToolExecutor,
    SelectedKVGather,
    Skill,
    Tool,
    Toolset,
    TypeContextPolicy,
    VLLMThinBackend,
    default_toolset,
    runtime_capabilities,
)
from pra_hf.agent_disclosure import ToolCapabilityGraph, disclosure_policy_for_profile
from pra_hf.runtime_benchmark import run_runtime_microbenchmark
from data.agent_workflows import realistic_tool_catalog, workflow_executor, workflow_tasks

torch.manual_seed(7)
torch.set_grad_enabled(False)

print("Using the repository source package")
print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
})
'''
    ),
    md(
        r'''
## 1. Capability discovery is a claim boundary

The runtime reports optional systems without importing them. Availability is not performance:
`torch.compile`, Triton, vLLM, SGLang, TensorRT-LLM, and MLX remain unsupported or contract-only
until the corresponding path is actually executed and measured on the current host.
'''
    ),
    code(
        r'''
capabilities = runtime_capabilities()
capabilities
'''
    ),
    md(
        r'''
## 2. Separate PRA semantics from systems policy

`PRAConfig` controls routing and memory consumption semantics. `PRARuntimeConfig` layers physical
choices over it: backend, compilation mode, K/V layout, page size, cache limits, prefetch policy,
and profiler behavior. A runtime optimization is valid only when changing these systems fields
preserves model semantics.

The config artifact contains no credentials or model weights.
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
    page_tokens=4,
    cache_max_bytes=1 << 20,
    cache_max_entries=8,
)

with tempfile.TemporaryDirectory() as directory:
    config_path = runtime_config.save_pretrained(directory)
    restored_config = PRARuntimeConfig.from_pretrained(directory)
    config_artifact = json.loads(config_path.read_text(encoding="utf-8"))

{
    "round_trip_equal": restored_config == runtime_config,
    "schema_version": config_artifact["schema_version"],
    "systems_policy": {
        key: config_artifact[key]
        for key in ("backend", "compilation", "kv_layout", "page_tokens", "cache_max_bytes")
    },
}
'''
    ),
    md(
        r'''
## 3. One facade over the Paper 2 model API

`PRARuntime.from_model` wraps an existing HF model through `PRAForCausalLM`. In deployment,
`from_pretrained` performs the equivalent model and tokenizer load. Stable URIs connect references,
routing traces, caches, authorization, and physical K/V without using filenames as identity.

The tiny tokenizer is only an offline fixture. The model path itself is the real Llama adapter,
including grouped-query attention with fewer physical K/V heads than query heads.
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


def build_tiny_runtime(config, **components):
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
    return PRARuntime.from_model(
        model, TinyTokenizer(), runtime_config=config, **components
    )


runtime = build_tiny_runtime(runtime_config)
handle = runtime.add_reference(
    "memory://demo/facts",
    text="Paris is the capital of France.",
)
generation = runtime.generate(
    "Question: capital of France? Answer:",
    max_new_tokens=1,
    return_details=True,
)
{
    "reference_uri": handle.uri,
    "generated_token_fixture": generation.text,
    "selected": generation.stats["selected"],
    "materialized_kv_tokens": generation.stats["materialized_kv_tokens"],
    "backend": runtime.inspect()["backend"],
}
'''
    ),
    md(
        r'''
## 4. Authenticated external memory: cold, warm, and hot

Direct text is convenient, but production memory often starts as a URI. The external-memory
manager keeps three distinct states:

1. **Cold:** authorized descriptor and cheap external gist; source bytes are not encoded.
2. **Warm:** source is fetched and encoded into model-specific native memory.
3. **Hot:** the selected native K/V is materialized for immediate use.

Authorization is rechecked at the resolver boundary. Credentials remain opaque and are absent from
session snapshots and metrics. The example resolver is in-memory so it has no external side effect.
'''
    ),
    code(
        r'''
class NotebookMemoryResolver:
    name = "mem"

    def __init__(self):
        self.documents = {}
        self.fetches = 0

    def put(self, uri, text, version="v1"):
        self.documents[uri] = {"text": text, "version": version}

    def _authorized(self, uri, auth_context):
        if uri not in auth_context.authorization_scopes:
            raise PermissionError(f"Resource not authorized: {uri}")
        auth_context.credentials_for(self.name, uri)
        return self.documents[uri]

    async def stat(self, uri, auth_context, session):
        document = self._authorized(uri, auth_context)
        payload = document["text"].encode()
        return ResourceStat(
            uri=uri,
            resolver=self.name,
            version=document["version"],
            size_bytes=len(payload),
            title=document["text"],
        )

    async def fetch(self, uri, auth_context, session, byte_range=None):
        document = self._authorized(uri, auth_context)
        self.fetches += 1
        payload = document["text"].encode()
        return payload if byte_range is None else payload[slice(*byte_range)]

    async def external_gist(self, uri, metadata, auth_context, session):
        return metadata.title or uri


def notebook_encoder(source, metadata, context):
    tokens = tuple(source.decode().split())
    return NativeEncoding(
        uri=metadata.uri,
        source_version=metadata.version,
        model_fingerprint=context.model_fingerprint,
        tokenizer_fingerprint=context.tokenizer_fingerprint,
        config_fingerprint=context.config_fingerprint,
        token_count=len(tokens),
        byte_count=len(source),
        pra_gists=(" ".join(tokens[:3]),),
        logical_offsets=((0, len(tokens)),),
        payload=tokens,
    )


def notebook_materializer(encoding, selected_token_ids):
    tokens = encoding.payload
    if selected_token_ids is not None:
        tokens = tuple(tokens[index] for index in selected_token_ids)
    return HotMemoryHandle(
        uri=encoding.uri,
        source_version=encoding.source_version,
        selected_token_count=len(tokens),
        byte_count=sum(len(token.encode()) for token in tokens),
        payload=tokens,
    )


resolver = NotebookMemoryResolver()
registry = ResolverRegistry()
registry.register("mem", resolver)
manager = ExternalMemoryManager(
    encoding_context=EncodingContext(
        model_fingerprint="tiny-llama",
        tokenizer_fingerprint="tiny-tokenizer",
        encoding_config={"block_tokens": 32, "position": "rope"},
    ),
    encoder=notebook_encoder,
    materializer=notebook_materializer,
    resolvers=registry,
)
'''
    ),
    code(
        r'''
uri = "mem://documents/alpha"
resolver.put(uri, "alpha evidence connects beta")
auth = AuthContext(
    tenant_id="demo-tenant",
    user_id="demo-user",
    session_id="demo-session",
    authorization_scopes=frozenset({uri}),
    credential_provider=lambda _resolver, _uri: "opaque-demo-token",
)
memory_runtime = PRARuntime(
    config=runtime.config,
    backend=runtime.backend,
    external_memory=manager,
)
session = memory_runtime.open_session(
    session_id="demo-session",
    user_id="demo-user",
    tenant_id="demo-tenant",
    auth_context=auth,
)
record = await memory_runtime.add_external_reference(session, uri=uri, encoding_mode="lazy")
cold_tier = record.tier
candidates = await manager.route_candidates(session, "alpha evidence")
await manager.admit(session, candidates, max_admitted=1)
warm_tier = record.tier
first_hot = await manager.ensure_hot(session, uri, selected_token_ids=(0, 2))
hot_tier = record.tier
second_hot = await manager.ensure_hot(session, uri)
third_hot = await manager.ensure_hot(session, uri)

safe_state = memory_runtime.inspect()
{
    "tier_transitions": [cold_tier, warm_tier, hot_tier],
    "selected_hot_payload": first_hot.payload,
    "reused_full_hot_handle": second_hot == third_hot,
    "resolver_fetches": resolver.fetches,
    "lifecycle_metrics": safe_state["external_memory"],
    "session_snapshot": safe_state["sessions"][0],
    "credential_leaked": "opaque-demo-token" in json.dumps(safe_state),
}
'''
    ),
    md(
        r'''
## 5. Exact interval planning before physical K/V work

Routing produces logical selections. `MaterializationPlan` turns them into stable half-open
`[start, end)` intervals grouped by URI and layer. Overlaps are merged before the global token
budget is enforced, preventing repeated K/V from consuming capacity twice.
'''
    ),
    code(
        r'''
plan = MaterializationPlan.build(
    [
        KVInterval("memory://demo/a", 0, 0, 8),
        KVInterval("memory://demo/a", 0, 4, 12),
        KVInterval("memory://demo/b", 0, 0, 8),
    ],
    max_tokens=16,
)
{
    "requested_tokens": plan.requested_tokens,
    "unique_tokens_after_merge_and_budget": plan.unique_tokens,
    "dropped_tokens": plan.dropped_tokens,
    "physical_intervals": plan.intervals,
}
'''
    ),
    md(
        r'''
## 6. Native K/V shape and grouped-query attention

Warm K/V is stored as `[batch, physical_kv_heads, tokens, head_dim]`. It is not expanded to the
larger number of query heads. For GQA and MQA, preserving physical K/V heads is essential to the
memory claim; head expansion belongs at attention consumption, not in persistent memory.
'''
    ),
    code(
        r'''
def native_kv(tokens, offset=0.0):
    key = torch.arange(1 * 2 * tokens * 4, dtype=torch.float32).reshape(1, 2, tokens, 4)
    key = key + offset
    return NativeKV(key, key + 1000)


sources = {
    ("memory://demo/a", 0): native_kv(12),
    ("memory://demo/b", 0): native_kv(8, 100),
}
materialized = KVMaterializer().materialize(sources, plan)
{
    "source_shape": tuple(sources[("memory://demo/a", 0)].key.shape),
    "packed_shape": tuple(materialized.layers[0].key.shape),
    "logical_tokens": materialized.logical_tokens,
    "physical_bytes": materialized.physical_bytes,
    "transfer_bytes": materialized.transfer_bytes,
    "temporary_bytes": materialized.temporary_bytes,
}
'''
    ),
    md(
        r'''
## 7. Four physical layouts, one logical result

The store can be ordered layer-major, reference-major, chunk-major, or block-major. A placement
index remaps logical URI/layer/page coordinates into contiguous physical ranges. Layout is a
systems choice only: every layout must reconstruct identical per-layer K/V before attention.
'''
    ),
    code(
        r'''
layout_rows = []
reference = KVMaterializer().materialize(sources, plan)
for layout in ("layer_major", "reference_major", "chunk_major", "block_major"):
    store = PackedNativeKVStore(sources, layout=layout, page_tokens=4)
    restored = KVMaterializer(layout=layout).materialize(store, plan)
    parity = all(
        torch.equal(restored.layers[layer].key, reference.layers[layer].key)
        and torch.equal(restored.layers[layer].value, reference.layers[layer].value)
        for layer in reference.layers
    )
    layout_rows.append({
        "layout": layout,
        "parity": parity,
        "store_bytes": store.nbytes,
        "placement_index_bytes": store.index_bytes,
        "placements": len(store.placements),
    })
layout_rows
'''
    ),
    md(
        r'''
## 8. Portable eager selected-token gather

`SelectedKVGather` gathers token positions from both K and V along dimension 2. The eager path is
the correctness baseline. A compiled wrapper is a separate capability gate: failure to compile is
reported, never relabeled as eager performance.
'''
    ),
    code(
        r'''
indices = torch.tensor([0, 3, 7, 11], dtype=torch.long)
gather = SelectedKVGather("eager")
gathered = gather(sources[("memory://demo/a", 0)], indices)
{
    "gather_state": gather.inspect(),
    "indices": indices.tolist(),
    "output_shape": tuple(gathered.key.shape),
    "exact_key_parity": torch.equal(
        gathered.key,
        sources[("memory://demo/a", 0)].key.index_select(2, indices),
    ),
}
'''
    ),
    md(
        r'''
## 9. Byte-bounded hot-cache behavior

The runtime LRU limits both bytes and entries. It reports loaded and reused bytes separately, so a
high hit rate cannot conceal reload amplification or an oversized resident set.
'''
    ),
    code(
        r'''
cache = RuntimeKVCache(max_bytes=10, max_entries=3)
cache.put("reference-a", "A", nbytes=6)
first_reuse = cache.get("reference-a")
cache.put("reference-b", "B", nbytes=6)  # evicts A to restore the byte budget
evicted_lookup = cache.get("reference-a")
{
    "first_reuse": first_reuse,
    "evicted_lookup": evicted_lookup,
    "accounting": cache.snapshot(),
}
'''
    ),
    md(
        r'''
## 10. Stage-level profiling and physical accounting

The profiler records stage latency, input/output bytes, metadata, and peak CUDA allocation. CUDA
timings synchronize only around explicitly profiled regions. Routing quality and runtime costs stay
in separate records.
'''
    ),
    code(
        r'''
profiler = RuntimeProfiler(device="cpu")
with profiler.stage(
    "selected_kv_materialization",
    input_bytes=sum(memory.nbytes for memory in sources.values()),
    metadata={"layout": "layer_major", "selected_tokens": plan.unique_tokens},
) as accounting:
    profiled_materialized = KVMaterializer().materialize(sources, plan)
    accounting["output_bytes"] = profiled_materialized.physical_bytes

profiler.snapshot()
'''
    ),
    md(
        r'''
## 11. Structured mechanism benchmark

This small run exercises indexed gather, interval packing, physical layouts, hierarchy rows where
available, and cache accounting. It is a mechanism benchmark, not end-to-end TTFT. The paper uses
larger repeated runs and stores every sample as JSON/CSV before plotting.
'''
    ),
    code(
        r'''
benchmark = run_runtime_microbenchmark(
    device="cpu",
    candidate_tokens=256,
    selected_tokens=32,
    batches=(1, 2),
    kv_heads=2,
    head_dim=16,
    warmups=1,
    repeats=2,
    include_compile=False,
)
[
    {
        "study": row["study"],
        "mode": row["mode"],
        "batch": row["batch"],
        "status": row["status"],
        "median_ms": (
            round(row["median_seconds"] * 1000, 4)
            if row["median_seconds"] is not None
            else None
        ),
        "parity": row["parity"],
        "error": row["error"],
    }
    for row in benchmark["summary"]
]
'''
    ),
    md(
        r'''
## 12. Typed resource discovery

Paper 6.5 resources are now part of the same SDK. Each capability has a stable URI, tenant, input
schema, output schema, side-effect class, and searchable metadata. Discovery chooses identities;
it does not disclose every schema or authorize execution.
'''
    ),
    code(
        r'''
resources = realistic_tool_catalog()
task = workflow_tasks()[0]
discovery = ResourceDiscoveryEngine(
    PersistentResourceIndex(resources),
    select_threshold=0.0,
    ask_threshold=0.0,
    margin_threshold=0.0,
)
tool_runtime = PRARuntime(
    config=runtime.config,
    backend=runtime.backend,
    discovery=discovery,
    executor=workflow_executor(resources, task),
)
discovery_trace = tool_runtime.discover_resources(
    DiscoveryRequest(query="search documents", tenant_id="paper6_5", top_k=3)
)
{
    "decision": discovery_trace.decision,
    "selected_uris": discovery_trace.selected_uris,
    "top_scores": [
        {"uri": row.uri, "score": round(row.selected_score, 3), "mode": row.selected_mode}
        for row in discovery_trace.candidates[:3]
    ],
}
'''
    ),
    md(
        r'''
## 13. Capability-graph disclosure

Disclosure is a second, bounded step. Starting from direct discovery roots, a policy may add local
family neighbors or schema-compatible predecessors/successors. Provenance records why each tool
became visible. Destructive capabilities remain suppressed unless policy explicitly allows them.
'''
    ),
    code(
        r'''
graph = ToolCapabilityGraph(resources)
root_uris = discovery_trace.selected_uris[:1]
disclosure = graph.disclose(
    root_uris,
    disclosure_policy_for_profile("planning", max_tools=6),
    root_confidence=1.0,
)
{
    "graph_density": round(graph.density, 4),
    "roots": disclosure.root_uris,
    "disclosed": disclosure.disclosed_uris,
    "graph_expansions": disclosure.graph_expansions,
    "unsafe_suppressed": disclosure.unsafe_suppressed,
    "provenance": [
        {"uri": row.uri, "source": row.source, "depth": row.graph_depth}
        for row in disclosure.provenance
    ],
}
'''
    ),
    md(
        r'''
## 14. Discovery and disclosure still do not authorize execution

The model may emit a structured call only to a disclosed identity. `SafeToolExecutor` then checks
selection, host authorization, side-effect permission, and argument schema. This keeps model
visibility and host authority independent.
'''
    ),
    code(
        r'''
search = next(resource for resource in resources if resource.name == "search_document")
proposal = '<tool_call>{"name":"search_document","arguments":{"title":"quarterly"}}</tool_call>'

not_selected = tool_runtime.execute_tool(
    proposal,
    selected_uris=(),
    authorization=ExecutionAuthorization(frozenset({search.uri})),
    call_id="not-selected",
)
not_authorized = tool_runtime.execute_tool(
    proposal,
    selected_uris=(search.uri,),
    authorization=ExecutionAuthorization(frozenset()),
    call_id="not-authorized",
)
accepted = tool_runtime.execute_tool(
    proposal,
    selected_uris=(search.uri,),
    authorization=ExecutionAuthorization(frozenset({search.uri})),
    call_id="accepted",
)
{
    "not_selected": not_selected.reason,
    "not_authorized": not_authorized.reason,
    "accepted": accepted.executed,
    "typed_observation_uri": accepted.observation.uri,
}
'''
    ),
    md(
        r'''
## 15. Lazy callable and skill records

The product SDK now accepts ordinary Python callables, explicit `Skill` objects, and a parent
directory containing OpenAI- or Anthropic-style `SKILL.md` folders. Discovery sees only compact
selection views by default. Full tool schemas and skill instructions are encoded only after exact
identity selection, with count and token budgets applied to the candidate palette.
'''
    ),
    code(
        r'''
def lookup_incidents(service: str) -> dict[str, object]:
    """Return recent incidents for one service."""

    return {
        "columns": ["service", "status", "latency_ms"],
        "rows": [
            {"service": service, "status": "ok", "latency_ms": 12},
            {"service": service, "status": "failed", "latency_ms": 950},
            {"service": service, "status": "ok", "latency_ms": 18},
        ],
    }


skills_root = Path(tempfile.mkdtemp(prefix="pra-runtime-skills-"))
openai_folder = skills_root / "release-review"
openai_folder.mkdir()
(openai_folder / "SKILL.md").write_text(
    "---\nname: release-review\ndescription: Review a release before publication.\n"
    "metadata:\n  short-description: Release review\n---\n\n"
    "Check tests, migration risk, monitoring, and rollback evidence.\n",
    encoding="utf-8",
)
(openai_folder / "agents").mkdir()
(openai_folder / "agents" / "openai.yaml").write_text("interface: {}\n", encoding="utf-8")

agent_config = AgentConfig(
    tools=(lookup_incidents,),
    skills=(Skill(
        name="incident-triage",
        description="Prioritize operational incidents.",
        when_to_use="Use when service health degrades.",
        instructions="Inspect evidence, assess impact, and assign the next safe action.",
        namespace="runtime-demo",
        tenant_id="runtime-demo",
    ),),
    skills_path=skills_root,
    namespace="runtime-demo",
    tenant_id="runtime-demo",
    max_candidates=4,
    selection_view_token_budget=256,
    encoding=CapabilityEncodingPolicy(lazy_selection=True, lazy_full=True),
)
capability_sdk = CapabilitySDK(agent_config)
capability_resources = capability_sdk.resources()
palette = capability_sdk.activate_candidates(
    [resource.uri for resource in capability_resources]
)
skill_uri = next(resource.uri for resource in capability_resources if resource.kind == "skill")
tool_uri = next(resource.uri for resource in capability_resources if resource.kind == "tool")
skill_activation = capability_sdk.activate_selected(skill_uri)
tool_activation = capability_sdk.activate_selected(tool_uri)
{
    "capability_kinds": sorted({resource.kind for resource in capability_resources}),
    "skill_folder_formats": sorted(skill.metadata.get("source_format", "object") for skill in capability_sdk.skills),
    "palette_records": palette.admitted_record_ids,
    "selection_tokens": palette.selection_tokens,
    "full_skill_cache_hit": skill_activation.cache_hit,
    "full_tool_cache_hit": tool_activation.cache_hit,
    "semantic_rediscovery_calls": tool_activation.semantic_rediscovery_calls,
}
'''
    ),
    md(
        r'''
## 16. Compact typed result records

A successful tool result no longer has to re-enter the prompt as one flat payload.
`execute_tool_and_record` preserves the exact result in a scoped, hash-verified backing store and
returns a type-aware compact view. Tool/API payloads infer tabular, log, graph, or terminal shape;
applications can search retrieval-only addresses, materialize selected rows or fields, or use a
bounded cursor. These operations retain the original record identity and authorization scope.
'''
    ),
    code(
        r'''
tool_resource = capability_sdk.tools[0].to_agent_resource()
tool_executor = SafeToolExecutor(
    (tool_resource,),
    {tool_resource.uri: lambda arguments, _observations: lookup_incidents(**arguments)},
)
result_store = Path(tempfile.mkdtemp(prefix="pra-runtime-results-"))
capability_runtime = PRARuntime(
    config=runtime.config,
    backend=runtime.backend,
    capability_sdk=capability_sdk,
    executor=tool_executor,
    context_policy=ContextPolicy(
        local_store=result_store,
        persistent_store=False,
        record_policies={RecordType.TOOL_RESPONSE: TypeContextPolicy(unit_limit=2)},
    ),
)
capability_session = capability_runtime.open_session(
    session_id="capability-demo",
    user_id="notebook-user",
    tenant_id="runtime-demo",
)
execution = capability_runtime.execute_tool_and_record(
    '<tool_call>{"name":"lookup_incidents","arguments":{"service":"billing"}}</tool_call>',
    session=capability_session,
    selected_uris=(tool_resource.uri,),
    authorization=ExecutionAuthorization(frozenset((tool_resource.uri,))),
    call_id="incident-call-1",
    capabilities=RecordCapabilities(
        searchable=True,
        partial_selectors=("rows", "fields"),
    ),
)
record_id = execution.record.record_id
compact = capability_runtime.compact_result(capability_session, record_id)
address_hits = capability_runtime.search_results(
    capability_session, "failed latency", top_k=2
)
selected_rows = capability_runtime.materialize_result(
    capability_session,
    record_id,
    level=RecordViewName.SELECTED,
    selector={"rows": [1, 2]},
)
cursor = capability_runtime.open_result_cursor(
    capability_session, record_id, collection="rows"
)
cursor_page = capability_runtime.execute_result_cursor(
    capability_session, CursorAction(cursor.cursor_id, "next")
)
{
    "executed": execution.execution.executed,
    "record_id": record_id,
    "compression_strategy": execution.record.compression_strategy,
    "compact_row_count": compact["row_count"],
    "compact_rows_retained": len(compact["representative_rows"]),
    "address_hit_ids": [record.record_id for record in address_hits],
    "selected_status": selected_rows.payload["rows"][0]["status"],
    "cursor_items": len(cursor_page.payload.items),
    "runtime_accounting": capability_runtime.inspect()["result_contexts"][capability_session.session_id],
}
'''
    ),
    md(
        r'''
## 17. Opt-in native PRA routing over exact result backing

Compact views and deterministic address search work without changing model memory. For an isolated
model session, `native_result_routing=True` enables Paper 7's stronger path: explicitly register
exact backing through the production PRA encoder, call the route-only model API, and decode only
the selected original spans. The runtime does not register compact descriptors or full backing
implicitly, and session teardown removes every reference it registered.
'''
    ),
    code(
        r'''
native_result_runtime = build_tiny_runtime(
    runtime_config,
    context_policy=ContextPolicy(
        local_store=Path(tempfile.mkdtemp(prefix="pra-native-results-")),
        persistent_store=False,
    ),
    native_result_routing=True,
)
native_session = native_result_runtime.open_session(
    session_id="native-result-demo",
    user_id="notebook-user",
    tenant_id="runtime-demo",
)
native_record = native_result_runtime.ingest_result(
    native_session,
    "alpha beta exact evidence says deploy canary after validation omega",
    record_type=RecordType.GENERIC_TEXT,
    capabilities=RecordCapabilities(searchable=True),
)
native_handle = native_result_runtime.register_result_backing(
    native_session, native_record.record_id
)
native_selection = native_result_runtime.route_result_backing(
    native_session, "what should deploy after validation"
)
native_detail = native_result_runtime.materialize_routed_result(
    native_session, native_selection
)
native_summary = {
    "backing_uri": native_handle.uri,
    "selected_record_ids": native_selection.record_ids,
    "requested_kv_tokens": native_selection.routing.stats["requested_kv_tokens"],
    "materialized_detail": native_detail.success,
    "selected_chunk_count": len(native_detail.payload["selected_chunks"]),
}
native_result_runtime.close_session(native_session)
native_summary["references_after_close"] = len(native_result_runtime.backend.model.stats()["references"])
native_summary
'''
    ),
    md(
        r'''
## 18. Thin vLLM handoff

Paper 4.5 stops before a retrieval-aware serving scheduler. The thin boundary carries only stable
selected identities, the materialized-token count, and ordinary request metadata. Semantic scores
remain outside vLLM, so the scheduler does not acquire hidden retrieval policy.

Calling `prepare` is a contract demonstration, not a vLLM speed claim.
'''
    ),
    code(
        r'''
handoff = VLLMThinBackend().prepare(
    "Question: capital of France?",
    selected_uris=(handle.uri,),
    materialized_tokens=generation.stats["materialized_kv_tokens"],
    metadata={"kv_layout": runtime_config.kv_layout},
)
{
    "request_id_present": bool(handoff.request_id),
    "selected_uris": handoff.selected_uris,
    "materialized_tokens": handoff.materialized_tokens,
    "scheduler_receives_semantic_scores": "score" in repr(handoff),
}
'''
    ),
    md(
        r'''
## 19. Unified inspection and session teardown

`inspect()` exposes non-secret configuration, backend state, external-memory counters, installed
boundaries, and cache accounting. Closing a session removes its ephemeral state while preserving
the runtime object and broader caches according to policy.
'''
    ),
    code(
        r'''
before_close = memory_runtime.inspect()
memory_runtime.close_session(session)
capability_runtime.close_session(capability_session)
after_close = memory_runtime.inspect()
{
    "before_close_sessions": len(before_close["sessions"]),
    "after_close_sessions": len(after_close["sessions"]),
    "session_closed": session.closed,
    "typed_discovery_installed_on_tool_runtime": tool_runtime.inspect()["typed_discovery_installed"],
    "safe_executor_installed_on_tool_runtime": tool_runtime.inspect()["safe_executor_installed"],
}
'''
    ),
    md(
        r'''
## 20. Persistent tasks and reusable toolsets

Paper 8 adds the logical state needed by an actual long-running agent. A `SessionService` resolves
state by user and session, stores an ordered typed-record stream, and keeps a versioned task DAG.
Closing `PRARuntime` releases physical model and K/V state; reopening the same logical session does
not erase tasks or compact observations. `InMemorySessionService` is useful for tests, while
`LocalSessionService` uses atomic JSON manifests for process-to-process persistence.

`Toolset` keeps each callable beside its declared side-effect class. The built-in workspace bundle
provides bounded read/search/edit/Git/command tools. Discovery never grants execution authority:
write and destructive calls still require a host policy or one interactive approval.
'''
    ),
    code(
        r'''
def uppercase(value: str) -> dict[str, str]:
    """Uppercase one value for the SDK demonstration."""

    return {"value": value.upper()}


demo_toolset = Toolset((Tool(uppercase, tenant_id="runtime-demo"),), name="notebook")
task_sdk = CapabilitySDK(AgentConfig(
    tools=demo_toolset.records,
    tenant_id="runtime-demo",
))
task_runtime = PRARuntime(
    config=runtime.config,
    backend=runtime.backend,
    capability_sdk=task_sdk,
    executor=demo_toolset.executor(),
    session_service=InMemorySessionService(),
    context_policy=ContextPolicy(
        local_store=Path(tempfile.mkdtemp(prefix="pra-task-results-")),
        persistent_store=False,
    ),
)
task_agent = PRAAgent(
    task_runtime,
    config=PRAAgentConfig(user_id="notebook-user", tenant_id="runtime-demo"),
    toolset=demo_toolset,
)
initial = task_agent.start_session("paper4-5-demo", task_description="Inspect runtime status")
updated = task_agent.create_task(
    "Summarize the selected evidence",
    task_id="task-2",
    parent_task_id="task-1",
)
physical_id = task_agent.session.session_id
task_agent.close()
resumed = task_agent.start_session(physical_id, resume=True)
{
    "resumed_session": resumed.session_id,
    "active_task": resumed.active_task_id,
    "task_count": len(resumed.tasks.tasks),
    "toolset": demo_toolset.inspect(),
    "default_tool_names": [
        resource.name for resource in default_toolset(PROJECT_ROOT).resources
    ],
}
'''
    ),
    md(
        r'''
## 21. Agent CLI and terminal UI

`pra-hf agent chat` assembles the model, PRA runtime, local session service, optional OpenAI- or
Anthropic-style skills, and workspace toolset. Slash commands expose `/sessions`, `/tasks`,
`/task new`, `/task use`, `/task done`, `/context`, and `/tools`. Writes are denied by default and
confirmed one call at a time in the terminal. `--allow-writes` is an explicit unattended-host
override; it does not change the tool's recorded side-effect class.
'''
    ),
    md(
        r'''
## 22. Multi-axis execution policy

Selection timing, layer scope, materialization lifetime, and residency are independent. The model
default is request/shared/request/keep. A request override is isolated and appears with field-level
provenance in `pra_execution`. Token/shared selects once at the routing layer in each forward;
earlier layers receive no current-token memory, while later layers resolve the same logical IDs to
their own native K/V. Phase selection intentionally raises until HF has a cache-correct prefill to
completion handoff.
'''
    ),
    code(
        r'''
policy_model = runtime.backend.model
request_per_layer = policy_model.generate(
    "Where is the runtime fact?",
    max_new_tokens=2,
    return_details=True,
    do_sample=False,
    pra_policy={"selection_layer_scope": "per_layer"},
)
token_shared = policy_model.generate(
    "Where is the runtime fact?",
    max_new_tokens=2,
    return_details=True,
    do_sample=False,
    pra_policy={
        "selection_stage": "token",
        "selection_layer_scope": "shared",
        "materialization_scope": "token",
    },
)
{
    "request_per_layer": request_per_layer.stats["pra_execution"],
    "token_shared": token_shared.stats["pra_execution"],
}
'''
    ),
    md(
        r'''
## 23. Standalone gateway and explicit downgrade

The wire carries structured messages and stable logical IDs, never raw tensors. This offline cell
uses a recording E0 adapter to show G10. The gateway inserts labeled text context only because the
request opts into fallback, and the trace states that native K/V was not used. G01 performs the
opposite migration: it derives typed resources from ordinary system and tool-result records before
calling a logical-reference-capable engine.
'''
    ),
    code(
        r'''
class NotebookEngine:
    def __init__(self):
        self.last_request = None

    def capabilities(self):
        return PRAEngineCapabilities(adapter="notebook-e0")

    def prepare_session(self, request):
        return request.session_id

    def generate(self, request):
        self.last_request = request
        return PRAEngineResult("offline gateway response")

    def stream(self, request):
        raise NotImplementedError

    def close_session(self, session_id):
        return None


engine = NotebookEngine()
gateway = PRAGateway(engine, mode="G10")
gateway_result = gateway.generate(PRAWireRequest(
    model="offline/tiny-llama",
    messages=({"role": "user", "content": "Use the selected fact."},),
    tenant_id="runtime-demo",
    resources=(PRAWireResource(
        resource_id="fact-1",
        uri="pra://runtime-demo/fact-1",
        text="The selected runtime fact is exact.",
        metadata={"tenant_id": "runtime-demo"},
    ),),
    required_capabilities=("native_kv",),
    allow_text_fallback=True,
))
{
    "gateway": gateway.capabilities(),
    "materialized_message": engine.last_request.messages[0],
    "trace": gateway_result.trace,
}
'''
    ),
    md(
        r'''
## CLI equivalents

The same systems surface is available without notebook state:

```powershell
python -m pra_hf.cli runtime init ./runtime-config --kv-layout block_major
python -m pra_hf.cli runtime inspect ./runtime-config
python -m pra_hf.cli runtime capabilities
python -m pra_hf.cli runtime benchmark --output ./runtime-results
python -m pra_hf.cli runtime prepare-vllm "A user prompt" --selected-uri memory://demo/facts
python -m pra_hf.cli agent chat Qwen/Qwen3-0.6B --workspace . --task "Inspect the repository"
python -m pra_hf.cli agent chat Qwen/Qwen3-0.6B --resume --user-id local-user
pra gateway serve --mode G10 --backend sglang --backend-url http://localhost:30000
```

## What this notebook proves, and what it does not

Demonstrated here:

- one model/memory/resource/execution facade;
- versioned non-secret systems configuration;
- real HF/PRA model wrapping and direct references;
- authenticated cold-to-warm-to-hot external memory;
- exact overlap deduplication and hard materialization budgets;
- native `[B, Hkv, T, D]` storage and parity across four physical layouts;
- eager gather, byte-bounded LRU reuse, stage profiling, and structured benchmarking;
- typed discovery, bounded graph disclosure, and separate execution authorization;
- lazy callable and skill records with exact full-view activation;
- scoped type-aware result compaction, address search, selective replay, and cursors;
- opt-in route-only native PRA retrieval over exact result backing;
- durable user/session resolution, versioned task DAGs, and task-scoped typed records;
- reusable workspace toolsets and a resumable coding-agent terminal UI;
- request/per-layer and token/shared execution with request-owned traces;
- a logical gateway request and explicitly labeled text fallback;
- a scheduler-unaware vLLM request contract.

Not demonstrated here:

- meaningful language quality from the random tiny model;
- a supported `torch.compile` result on every host;
- Triton/custom CUDA fusion;
- asynchronous transfer overlap or production prefetch;
- continuous batching, p95/p99 serving latency, or a deep vLLM scheduler integration;
- streaming gateway proxying or phase-level HF selection;
- native or measured SGLang, FreeToken, TensorRT-LLM, MLX, llama.cpp, TGI, or Ollama adapters.

Those omissions are explicit capability gates, not hidden eager fallbacks.
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
