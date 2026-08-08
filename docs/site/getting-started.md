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
