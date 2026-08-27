# Getting Started

## Install

Create a Python 3.10 or newer environment and install the project in editable mode:

```bash
python -m pip install -e .
```

Install documentation tooling as well:

```bash
python -m pip install -e ".[docs]"
```

## Inspect configuration

The Click CLI loads defaults from `config/config.yml`. A named model profile can override
model, training, and dataset sections.

```bash
python -m pra_torch.cli config show
python -m pra_torch.cli dataset show -g stage0_synthetic_memory -m 2
```

## Train and evaluate

Run the standalone entry points:

```bash
python scripts/train_standalone.py --model standalone_tiny
python scripts/eval_standalone.py \
  --model standalone_tiny \
  --checkpoint out/standalone_tiny/checkpoints/best.pt
```

Use `--device cuda` when CUDA-enabled PyTorch is installed and a compatible GPU is
available. The generic runtime also accepts `device="auto"` and selects CUDA when present.

## Run tests

```bash
python -m pytest
```

Focused suites for the core architecture are:

```bash
python -m pytest tests/test_pra_routing.py tests/test_pra_batching.py
```

## Configure tools and skills

`AgentConfig` accepts Python callables, explicit `Skill` objects, and a parent
directory whose child folders contain OpenAI- or Anthropic-style `SKILL.md`
files. Compact selection views and complete schemas or instructions are encoded
lazily by default.

```python
from pra_hf import AgentConfig, CapabilityEncodingPolicy, Skill

agent = AgentConfig(
    tools=(lookup_incidents,),
    skills=(Skill(
        name="incident-triage",
        description="Prioritize operational incidents.",
        when_to_use="Use when service health degrades.",
        instructions="Inspect evidence and assign the next safe action.",
    ),),
    skills_path="./skills",
    max_candidates=24,
    selection_view_token_budget=2048,
    encoding=CapabilityEncodingPolicy(lazy_selection=True, lazy_full=True),
)
```

Pass `agent_config=agent` to `PRARuntime`. Discovery returns stable record IDs;
`activate_capability_candidates()` admits a bounded selection palette and
`activate_capability()` resolves one exact full definition locally.

## Compact result records

Each runtime session owns a scoped exact backing store. Successful tool output
can be compacted automatically, or any supported result can be ingested directly:

```python
from pra_hf import ContextPolicy, RecordType, RecordViewName, TypeContextPolicy

runtime = PRARuntime.from_pretrained(
    model_id,
    agent_config=agent,
    context_policy=ContextPolicy(
        max_native_index_tokens=4096,
        max_native_index_bytes=65536,
        record_policies={
            RecordType.TOOL_RESPONSE: TypeContextPolicy(
                unit_limit=8,
                max_native_index_tokens=2048,
            ),
        },
    ),
)
session = runtime.open_session(
    session_id="request-7", user_id="user-1", tenant_id="tenant-1"
)
record = runtime.ingest_result(
    session, payload, record_type=RecordType.API_RESULT
)
compact = runtime.compact_result(session, record.record_id)
selected = runtime.materialize_result(
    session,
    record.record_id,
    level=RecordViewName.SELECTED,
    selector={"rows": [20, 30]},
)
```

Tool/API results infer tabular, log, graph, terminal, or generic structured
shape. Full bytes stay hash-verified and tenant/session scoped. Address search,
bounded cursors, TTLs, storage placement, native-index ingestion limits, and
per-record overrides are configured through `ContextPolicy`.

`PRARuntime.from_pretrained()` enables size-adaptive native result routing by
default. In-budget records enter `BUILT`; oversized records enter
`SKIPPED_SIZE_LIMIT`; policy-deferred records enter `DEFERRED`. A skipped record
is not truncated or mislabeled as native. Its compact view, cheap addresses,
search, and cursor remain available. After cheap selection, promote only an
authorized bounded region with:

```python
audit = runtime.result_native_index_audit(session, record.record_id)
region = runtime.encode_result_region_native(
    session, record.record_id, {"rows": [20, 30]}
)
```

Use `prepare_result_native_index(..., force=True)` to resolve a deliberately
deferred index. Pass `native_result_routing=False` when model-resident result
references are not appropriate, such as a shared model process without tenant
partitioning.

## Run a persistent PRA agent

`PRAAgent` combines the same runtime with durable typed sessions, a versioned task DAG,
lazy skills, and a reusable `Toolset`. The default local service resolves sessions by
`user_id` and `session_id`; closing the model session releases ephemeral K/V without
deleting the logical record stream.

```python
from pra_hf import PRAAgent, PRAAgentConfig

agent = PRAAgent.from_pretrained(
    model_id,
    config=PRAAgentConfig(user_id="user-1"),
    workspace=".",
    sessions_path=".pra/sessions",
    skills_path="./skills",
)
state = agent.start_session(task_description="Inspect the failing tests")
turn = agent.run_turn("Find the relevant test and explain the failure")
agent.close()
```

The coding-agent terminal exposes task, context, tool, and session inspection:

```bash
pra-hf agent chat Qwen/Qwen3-0.6B --workspace . --task "Inspect this repository"
pra-hf agent chat Qwen/Qwen3-0.6B --resume --user-id local-user
```

The built-in toolset can list, read, search, edit, inspect Git, and run a command inside
the configured workspace. Writes are denied by default and require one interactive host
approval. `--allow-writes` is intended only for explicitly trusted unattended runs.

## Build this site

Generate the static HTML site under `site/`:

```bash
python -m mkdocs build --strict
```

Run a local development server with automatic source and documentation reloads:

```bash
python -m mkdocs serve
```

MkDocs serves the site at `http://127.0.0.1:8000/` by default.
