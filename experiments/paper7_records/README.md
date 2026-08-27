# Paper 7 progressive-context experiments

`run_progressive_context_iteration.py` is the primary balanced context-control
benchmark. It creates 30 typed-result identities across six classes: compact
sufficiency, bounded full recovery, partial materialization, cursor use,
in-record search, and information that requires a new tool call. Five decoding
seeds compare:

- full context;
- compact-only context;
- automatic PRA address selection;
- model-controlled escalation;
- PRA selection followed by model escalation; and
- a binary retrieve-original CCR baseline.

The experiment reports answer/evidence success separately from mechanism
correctness, under- and over-expansion, materialized tokens and bytes, model
passes, latency, class-level results, and paired case-cluster bootstrap
intervals. The model emits only an action class in this controlled benchmark;
the host binds the sole active authorized record, cursor, selector, query, or
tool. The public runtime separately supports validated structured decisions.

Run from the repository root with a local Ollama `qwen3:0.6b` model:

```powershell
python experiments/paper7_records/run_progressive_context_iteration.py
```

Artifacts are written under
`docs/papers/shared/results/paper7_records/progressive_context/`.

## Secondary mechanism studies

`run_inception_experiments.py` runs the bounded mechanism study used by the
paper. It covers five seeds and keeps model-dependent claims out of the first
implementation checkpoint:

- type-specific compact-view savings;
- exact backing-state recovery;
- compact, explicit-address, latent-query, and proactive-probe trigger reachability;
- native-event, tool, mixed-cursor, and proactive materialization accounting;
- cursor aggregate/drill-down correctness;
- adaptive transport decisions across payload sizes and deployment topologies.

Run from the repository root:

```powershell
python experiments/paper7_records/run_inception_experiments.py
```

Artifacts are written under
`docs/papers/shared/results/paper7_records/inception/`.

## Native-index size gate

`run_native_index_size_gate.py` profiles the final Paper 7 ingestion safeguard
over `1K`, `4K`, `16K`, `64K`, and `256K` token payloads and five seeds.
It compares eager full-body native indexing, size-gated cheap indexing,
size-gated lazy selected-region native encoding, and search/cursor-only
handling.

The script uses an instrumented fixed-width Torch encoder through the production
PRA reference API. Its results characterize lifecycle scaling, TTUC, state
transitions, and selected-region work. They are not pretrained-model quality or
Qwen latency measurements.

Run from the repository root:

```powershell
python experiments/paper7_records/run_native_index_size_gate.py
```

Artifacts are written under
`docs/papers/shared/results/paper7_records/native_index_size_gate/`.
