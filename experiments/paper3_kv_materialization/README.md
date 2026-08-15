# Paper 3 Materialization Experiments

`run_toy_materialization.py` is the primary causal experiment. It encodes each
controlled source once as contextual native K/V, fixes selection to one oracle
parent, and evaluates exact evidence cores, local radii, complete facts, the
whole parent, matched wrong memory, fixed budgets, and gist controls.

`summarize_toy_materialization.py` freezes the validation-selected sufficient
radius, writes the causal diagnosis and next-action gates, and produces the
controlled plots.

`run_oracle_frontier.py --study confirmation` reuses the original Qwen runner
with a smaller toy-motivated policy set and separate checkpoints. Validation
selects the local radius; held-out evaluation does not alter it. Pass an
explicit confirmation output directory and policy-selection path so these
artifacts cannot overwrite the preserved oracle pilot.

`summarize_pretrained_confirmation.py` computes paired example-level bootstrap
intervals, writes the pretrained frontier, updates `paper3_findings.json`, and
compares the controlled predictions with MuSiQue and 2Wiki.

No runner changes conceptual routing, trains model weights, or implements a
serving kernel.
