# Getting Started

Choose the path closest to your application. All three begin with Selected
Context; Native Memory is a later, measured optimization.

## Install

Use Python 3.10 or newer:

```bash
python -m pip install -e .
pra doctor
```

`pra doctor` inspects the Python environment containing the CLI. It is not a
machine-wide accelerator scan. Install the intended PyTorch build in that
environment before diagnosing CUDA or Apple Metal support.

## Path 1: five-minute evaluation

Inspect a model, inspect the execution plan, run a local benchmark, and compare
the evidence recorded for its profiles:

```bash
pra doctor
pra model inspect Qwen/Qwen3-1.7B
pra runtime inspect Qwen/Qwen3-1.7B -e hf
pra runtime benchmark Qwen/Qwen3-1.7B -e hf -o .pra/bench
pra profiles compare Qwen/Qwen3-1.7B
```

The benchmark writes a structured report under `.pra/bench`. For a Full Context
versus Selected Context evaluation, keep the task examples and model fixed,
freeze the selected record IDs/intervals, and compare quality alongside input
tokens and serving metrics. Follow the [qualification contract](metrics.md);
never interpret token reduction alone as cost savings.

## Path 2: existing OpenAI-compatible app

Start a gateway in front of the existing endpoint:

```bash
pra gateway serve \
  --mode selected-context \
  --backend vllm \
  --backend-url http://127.0.0.1:8000/v1
```

Point the application's OpenAI base URL to `http://127.0.0.1:8080/v1`. Ordinary
chat requests continue to work. PRA-aware requests can add typed records,
session identity, task metadata, and resource deltas; the gateway selects and
renders authorized context for the ordinary downstream engine.

Use `typed-transport` only when the immediate backend advertises the required
typed-resource capabilities:

```bash
pra gateway serve \
  --mode typed-transport \
  --backend sglang \
  --backend-url http://127.0.0.1:30000
```

See [Gateway deployment](deployment/gateway.md) and the [wire
protocol](protocol.md).

## Path 3: embedded Python or agent

Create the runtime, open a scoped session, ingest a typed record, and generate:

```python
from pra_hf import PRARuntime, RecordType

runtime = PRARuntime.from_pretrained("Qwen/Qwen3-1.7B")
session = runtime.open_session(
    session_id="incident-42",
    user_id="user-1",
    tenant_id="tenant-1",
)
record = runtime.ingest_result(
    session,
    "Restart the worker only after draining its queue.",
    record_type=RecordType.GENERIC_TEXT,
)
match = runtime.search_results(session, "worker restarts", top_k=1)[0]
detail = runtime.materialize_result(session, match.record_id)
answer = runtime.generate(
    f"Context:\n{detail.payload}\n\n"
    "Question: What should happen before the worker restarts?"
)
print(answer)
```

For an agent with durable sessions, tasks, skills, and tools:

```bash
pra agent chat Qwen/Qwen3-0.6B -w . -t "Inspect this repository"
pra agent run -p work "Summarize the current task state."
```

See [Agent deployment](deployment/agent.md) and [Runtime / SDK](deployment/runtime-sdk.md).

## Choose an engine

Do not infer Native Memory support from ordinary prefix caching. Check the
[engine support matrix](engines/overview.md), run `pra runtime doctor` where a
provider exists, and require explicit capability negotiation through a gateway.

## Build the documentation

Install the documentation dependencies and build strict static HTML:

```bash
python -m pip install -e ".[docs]"
python -m experiments.paper4_5_runtime.build_technical_site --check
python -m mkdocs build --strict
```

Open `site/index.html` directly. The build uses explicit `.html` links, so local
navigation does not require a web server. For authoring with live reload:

```bash
python -m mkdocs serve
```

Standalone model training and paper reproduction live in [Research /
Evidence](research/index.md), not in the product quickstart.
