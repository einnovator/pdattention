# URI-Addressed Progressive Retrieval Attention (PRA) Research Prototype

This repository is a Codex handoff package for experimenting with **URI-addressed Progressive Retrieval Attention**.

The core idea: instead of placing all long context directly in the prompt, the prompt may contain explicit lightweight reference tokens such as:

```text
<REF_1>
<REF_2>
```

A PRA-enabled transformer treats those references as latent memory handles. At runtime, an explicit local `ReferenceTable` maps each token to a URI. The resolver partitions content into deterministic chunks and caches, for every PRA layer, full token K/V plus configurable `[G,D]` routing-gist sets for chunks and URIs. The default remains one mean gist per chunk and no URI-level routing stage. Each batch row independently uses its projected last-token query to select references and then chunks; only selected chunk K/V is materialized for cross-attention. Textual summaries are optional routing metadata and are disabled by default.

Recursive construction is child-first and follows only local reference tables. Shared depth/reference/token budgets, cycle and missing-reference policies, build states, and cache fingerprints keep expansion bounded and prevent partial or stale entries from becoming searchable. Variable selected-memory lengths can run independently or in deterministic masked buckets without sharing one example's memory with another.

## Repository goals

1. Provide a **standalone PyTorch TinyGPT** using PRA for local training/evaluation.
2. Provide a synthetic benchmark for long-context / reference QA.
3. Provide a notebook version for easier interactive work.
4. Provide separate experimental folders for Hugging Face wrappers and model-family-specific patches:
   - `hf_wrappers/`
   - `llama_patch/`
   - `mistral_patch/`
   - `qwen_patch/`
5. Keep the standalone PyTorch path clean so Codex can focus there first.

## Recommended development order

### Phase 1 — standalone PyTorch

Focus here first:

```text
pra_torch/
experiments/
tests/
notebooks/
```

Implement and validate:

- Tiny decoder-only transformer.
- PRA attention layer.
- `<REF_n>` token parser and runtime reference table.
- In-memory URI resolver.
- Layer-specific chunk K/V and routing-gist cache.
- Independent reference/chunk top-k routing.
- Bounded recursive anchor expansion through explicit local tables.
- Batch-isolated variable-length memory attention.
- Synthetic QA dataset.
- Training and evaluation scripts.

## CLI usage

The standalone path is driven by the Click CLI:

```bash
python -m pra_torch.cli config show
python -m pra_torch.cli dataset show -g stage0_synthetic_memory -m 2
python -m pra_torch.cli train -d cpu -s 200 -g stage0_synthetic_memory
python -m pra_torch.cli eval -d cpu -g stage0_synthetic_memory -K pra_tiny.pt
```

Defaults live in:

```text
config/config.yml
```

Use `-c/--config` to layer another YAML file on top of the default config. Explicit command-line options override both config files. If a config file is missing, the CLI prints a warning and continues with built-in/default values plus any command-line overrides.

## Long prompts

PRA can preserve history beyond the direct self-attention window as a request-local implicit
reference. Configure the `pra` section as follows:

```yaml
pra:
  max_prompt_direct_tokens: 4096
  prompt_overflow_mode: implicit_reference
  max_prompt_gists: null
```

For a prompt longer than 4096 tokens, the latest 4096 remain in direct causal attention.
Earlier token IDs become `pra://implicit/prompt/head`, displayed as `#__head`, and reuse the
ordinary chunk, gist, cache, routing, and K/V materialization pipeline. `max_prompt_gists`
limits only this implicit reference; `null` preserves every displaced chunk regardless of
the smaller explicit-reference cap. Explicit references remain in the same row-local cache.

`prompt_overflow_mode: truncate` is the backward-compatible default, while `error` rejects
oversized prompts. Generation supports long initial prompts; generated tokens that later
roll out of the direct window are not yet migrated into `#__head`.

Installed environments also get the console entry point:

```bash
pra train
pra eval
pra config show
pra dataset show
```

The product SDK also includes a persistent task-aware agent shell:

```bash
pra-hf agent chat Qwen/Qwen3-0.6B --workspace . --task "Inspect this repository"
pra-hf agent chat Qwen/Qwen3-0.6B --resume --user-id local-user
```

`PRAAgent` resolves durable typed sessions by user and session ID. `Toolset` registers
custom callables or the workspace-bounded default tools, while tool visibility and host
authorization remain separate. Subagents and learned long-term memory are intentionally
reserved for Papers 9 and 10.

## HTML documentation

The MkDocs site combines architecture guides with API reference generated from Python
docstrings by `mkdocstrings-python`.

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build --strict
```

The static site is written to `site/`. For local development with automatic reloads:

```bash
python -m mkdocs serve
```

## Research trainer

The professional standalone trainer uses the `standalone_tiny` profile in:

```text
config/config.yml
```

Both the Click CLI and standalone scripts use the model-independent infrastructure in
`src/common`, with the canonical optimizer loop in `src/common/train.py`. PRA-specific
cache construction, evaluation, and retrieval metrics remain in
`src/pra_torch/pra_train.py`. Generic infrastructure should be imported directly from
`common`; `PRAStandaloneTrainer` remains a thin shell for code that prefers an
object-oriented `train()`, `validate()`, and `test()` API.

Train:

```bash
python scripts/train_standalone.py --model standalone_tiny
```

Override the output directory or device:

```bash
python scripts/train_standalone.py \
  --model standalone_tiny \
  --output-dir out \
  --device cpu
```

Validate/test from a checkpoint:

```bash
python scripts/eval_standalone.py \
  --model standalone_tiny \
  --checkpoint out/standalone_tiny/checkpoints/best.pt \
  --predictions-jsonl out/standalone_tiny/predictions.jsonl
```

Resume training:

```bash
python scripts/train_standalone.py \
  --model standalone_tiny \
  --resume-from out/standalone_tiny/checkpoints/latest.pt
```

Checkpoints are saved under:

```text
out/<experiment_name>/checkpoints/latest.pt
out/<experiment_name>/checkpoints/best.pt
```

Retrieval traces are saved under:

```text
out/<experiment_name>/traces/*.jsonl
```

TensorBoard logging is enabled by default. View logs with:

```bash
tensorboard --logdir out/<experiment_name>/tensorboard
```

If the `tensorboard` package is missing, the logger falls back to a simple event-like file so experiments still run. To enable W&B or ClearML, set these in the YAML config:

```yaml
use_wandb: true
use_clearml: true
```

Both integrations are optional; missing packages do not crash the run.

## Dataset architecture

The standalone research path uses a PyTorch-native data pipeline under `src/data/`:

```text
src/data/
  schemas.py       # QuestionSample, ReferenceSample, DatasetMetadata
  tokenizer.py     # PRATokenizer with atomic <REF_n> handling
  datasets.py      # PRADataset and concrete dataset classes
  collators.py     # PRACollator for tensor batches
  datamodules.py   # PRADataModule for train/val/test loaders
  generators/      # JSONL generators for each stage
```

Every dataset derives from `PRADataset`, which itself derives from `torch.utils.data.Dataset`.
Dataset items are educational Python dataclasses, not tensors:

```python
QuestionSample(
    id=...,
    question="What is the JWT expiration? <REF_1>",
    answer="37 minutes",
    references=[ReferenceSample(...)],
    target_reference_ids=[1],
    metadata={...},
)
```

`PRACollator` is responsible for turning samples into model batches:

```python
{
    "input_ids": ...,
    "labels": ...,
    "attention_mask": ...,
    "reference_tables": ...,
    "metadata": ...,
}
```

Reference tokens such as `<REF_1>` are preserved as atomic tokenizer entries and are never split into characters.

## DataLoader and PRA

Training and evaluation use `PRADataModule`, which hides dataset-specific details:

```python
dm = PRADataModule(
    dataset_stage="stage0_synthetic_memory",
    data_dir="data",
    batch_size=8,
    max_seq_len=96,
).load()

for batch in dm.train_loader():
    ...
```

The training loop builds the model memory cache from `batch["metadata"]`, while the model itself remains independent from any dataset implementation.

## Adding a dataset

1. Add a dataset class in `src/data/datasets.py` deriving from `PRADataset`.
2. Return `QuestionSample` objects from `__getitem__`.
3. Add the class to `DATASET_REGISTRY`.
4. Add or generate JSONL files with:
   - `documents.jsonl`
   - `references.jsonl`
   - `questions.jsonl`
5. Optionally add a generator under `src/data/generators/`.

Generator modules expose:

```python
generate(out_dir, ...)
save_jsonl(path, rows)
```

Existing stages cover synthetic memory, hierarchical documents, code repos, Wikipedia-style articles, books, technical docs, and GitHub-like repos.

### Phase 2 — Hugging Face wrapper

Use `hf_wrappers/` only after the standalone PyTorch model works.

Start with small models such as:

- `Qwen/Qwen2.5-0.5B`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Do **not** start by replacing Llama 3.x attention directly. Use a compatibility adapter first.

### Phase 3 — model-family patches

The folders `llama_patch/`, `mistral_patch/`, and `qwen_patch/` contain specs and scaffolding for later work.

## Baselines to compare

The evaluation should compare at least:

1. No retrieval / no refs.
2. Full context, if sequence length allows.
3. Standard RAG-style prompt retrieval.
4. PRA content-gist routing, with optional summary-only/hybrid/augment ablations.
5. PRA recursive anchor expansion and oracle-chunk controls.

## Core claim to test

> Long-context inference can be represented as explicit latent references. Layer-specific content gists route to bounded reference chunks, while selected full-token K/V supplies evidence through a batch-isolated memory path and explicit recursive runtime.

## Important caution

The first standalone version can use simple learned absolute position embeddings. For real models such as Llama/Qwen/Mistral, RoPE, GQA/MQA, cache layout, and attention masks must be preserved. For pretrained models, prefer a **cross-attention adapter branch** first instead of direct K/V injection.
