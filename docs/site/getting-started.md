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

## Register tools and skills lazily

`CapabilitySDK` accepts ordinary Python callables, explicit `Skill` objects,
and a directory whose immediate child folders contain `SKILL.md` files:

```python
from pra_hf import AgentConfig, CapabilitySDK, Skill

sdk = CapabilitySDK(AgentConfig(
    tools=(my_python_tool,),
    skills=(Skill(
        name="incident-triage",
        description="Prioritize an operational incident.",
        when_to_use="Use when service health degrades.",
        instructions="Assess impact, evidence, ownership, and the next safe action.",
    ),),
    skills_path="./skills",
    max_candidates=24,
))
```

Discovery supplies stable record identities. `activate_candidates()` exposes
only bounded selection views; `activate_selected()` activates the exact full
record locally without rerunning semantic discovery.

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
