# Using a Bundle

## Automatic selection

```bash
pra serve Qwen/Qwen3-0.6B -e hf -a auto -p balanced
```

The registry resolves an immutable bundle commit matching the base model,
revision, engine compatibility, PRA version, schema, profile, and qualification
status. If no trusted match exists, the CLI reports that result and keeps generic
PRA available.

## Explicit Hub or local selection

```bash
pra hf pull EInnovator/pra-qwen3-0.6b
pra bundle validate EInnovator/pra-qwen3-0.6b
pra serve Qwen/Qwen3-0.6B -e hf -a EInnovator/pra-qwen3-0.6b -p balanced

pra serve Qwen/Qwen3-0.6B -e hf -a ./.pra/bundles/qwen3 -p balanced
```

For an exact MLX catalog identity:

```bash
pra hf pull EInnovator/pra-qwen3-14b-mlx-4bit
pra inspect mlx-community/Qwen3-14B-4bit -e mlx \
  -a EInnovator/pra-qwen3-14b-mlx-4bit

# Portable parameter-free routing is the default.
pra serve mlx-community/Qwen3-14B-4bit -e mlx \
  -a EInnovator/pra-qwen3-14b-mlx-4bit -p balanced

# The learned router is opt-in and qualified only for matched QASPER use.
pra serve mlx-community/Qwen3-14B-4bit -e mlx \
  -a EInnovator/pra-qwen3-14b-mlx-4bit -p qasper-learned
```

The learned profile is not a generic "higher quality" switch. In the held-out
catalog comparison it improves QASPER for all four models but reduces HotpotQA
recall. Evaluate the target workload before promoting it.

An explicit bundle must match the requested base-model identity. The managed HF
runtime loads profile-selected routing and memory adapters from the resolved
snapshot. Remote engines consume the bundle at the PRA gateway boundary unless
their adapter explicitly implements a deeper engine-native contract. Direct
engine launch rejects a bundle it cannot consume rather than ignoring it.

## Disable learned bundle components

```bash
pra serve Qwen/Qwen3-0.6B -e hf -a none
```

This is useful for structural-reference and Selected Context controls. It does
not disable typed context, routing policy, or PRA as a whole.
