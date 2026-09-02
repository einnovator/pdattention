# CLI End-to-End Testing

The end-to-end suite launches the installed PRA command through a fresh Python
subprocess. It is separate from Click's in-process unit tests and covers every
public leaf command in two ways:

1. Every command must expose a working help contract.
2. Every command must have an executable semantic classification: offline
   success, launched-and-probed service, expected policy rejection, guarded
   authentication, or an explicitly skipped external prerequisite.

The controlled fixtures include a tiny local Llama checkpoint and tokenizer,
synthetic routing features, qualification measurements, a checksummed PRA
bundle, and a local OpenAI-compatible endpoint. No language-model download is
required. Gateway proxying and Agent Web start/stop are exercised as real
services and cleaned up by the harness.

## Run offline

```bash
python experiments/paper4_5_runtime/run_cli_e2e.py \
  --output .pra/evidence/cli_e2e.json
```

The offline run still executes `pra hf list`, local bundle inspection, and Hub
publication dry runs. Live search and pull are recorded as external
prerequisites rather than silently passing.

## Include Hugging Face

```bash
python experiments/paper4_5_runtime/run_cli_e2e.py \
  --live-hub \
  --output .pra/evidence/cli_e2e_live.json
```

Live mode searches the canonical `EInnovator` catalog and pulls the immutable
`EInnovator/pra-qwen3-0.6b` bundle. It never downloads the base model.

The equivalent pytest entry point is:

```bash
PRA_RUN_CLI_E2E=1 PRA_CLI_E2E_LIVE_HUB=1 \
  python -m pytest -q -m e2e tests/e2e/test_cli_end_to_end.py
```

In PowerShell, set the variables with `$env:PRA_RUN_CLI_E2E='1'` and
`$env:PRA_CLI_E2E_LIVE_HUB='1'` before invoking pytest.

## Evidence contract

The JSON receipt records the Git commit, operating system, architecture, Python
and Torch versions, CUDA/MPS availability, duration, command arguments, exit
status, semantic classification, and a bounded output excerpt. Tokens,
credentials, prompts, and downloaded payloads are not included.

A run passes only when the discovered Click command tree and the semantic
coverage set are identical. Adding a new public subcommand therefore makes the
suite fail until a real expected behavior is assigned.
