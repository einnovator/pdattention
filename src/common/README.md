# Common Experiment Infrastructure

`common` is the model-independent runtime shared by PRA and unrelated Transformer
research. It does not import a model, retrieval implementation, or dataset package.

## Local compatibility

No distributed configuration is required. An implicit `local` worker, `local` cluster,
and filesystem storage backend preserve the original one-process behavior. Ordinary
`pra train` still calls the same optimizer loop and writes the requested checkpoint.

## Configuration

`common.config` recursively merges YAML files and directories in argument order. A
directory is searched recursively for non-hidden `.yml` and `.yaml` files and loaded in
normalized lexical order. Later sources win recursively. `PATH=YAML_VALUE` overrides are
typed and applied last.

The validated top-level registries are independent:

- `workers`: logical compute and a `local`, `process`, or `ssh` command transport;
- `clusters`: worker allocations and default distribution/storage policy;
- `storage`: named local, S3, or GCS artifact locations;
- `experiments`: arbitrary module/function, file/function, or file-only entrypoints.

See `config/distributed.example.yml` for a local process pool and future remote Macs.
Cloud SDKs are optional and imported only when a cloud backend is selected.

## Experiments

A callable receives plain parameters and an `ExperimentContext`:

```python
def run(params: dict, context: ExperimentContext) -> dict:
    # Write extra files below context.output_dir / "artifacts".
    return {"loss": 1.25, "accuracy": 0.8}
```

Sweeps use a deterministic Cartesian product and SHA-256 trial fingerprints. The
coordinator writes `run.json`; each trial writes `experiment.json`, `metric.json`,
`status.json`, logs, checkpoints, and artifacts. Successful matching trials are skipped
on resume, while incompatible fingerprints are rejected. Numeric metrics are aggregated
with count, mean, standard deviation, minimum, and maximum.

Independent `seeds`/`sweep` jobs use the capacity-aware scheduler. Cooperative `ddp` and
`fsdp` trials reserve multiple workers and use PyTorch process groups; these concepts are
kept separate deliberately. Pipeline wrapping is an explicit future hook.

## Training

`common.train` owns training state, optimization, logging, metrics, and checkpoints.
Architecture packages inject `batch_step`, `eval_step`, and model-specific checkpoint
metadata. Under DDP/FSDP, the common state wraps the model before optimizer creation,
uses distributed samplers, reduces metrics, and limits canonical checkpoint writes to
rank zero.

The portable cooperative baseline is CPU/Gloo. CUDA uses NCCL when available. Apple MPS
can run independent trials over SSH, but MPS distributed collectives are not required or
claimed; use Gloo/CPU for portable multi-process validation.

## CLI examples

```bash
pra config validate -c config/distributed.example.yml
pra worker ping local
pra experiment run common-five-seeds -c config/distributed.example.yml
pra train --seeds 0:5
pra train -C local
```

Existing `-P` remains PRA layer IDs, so generic dotted values use `--param` on `pra train`.
Dedicated existing short aliases are unchanged.
