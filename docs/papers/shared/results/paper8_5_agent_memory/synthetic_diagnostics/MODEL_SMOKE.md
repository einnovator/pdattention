# Qwen3-14B synthetic model smoke

We evaluated one predeclared probe decision per task on the M5 16 GB host with
Ollama 0.30.8, `qwen3:14b`, the `Qwen/Qwen3-14B` tokenizer, temperature zero,
seed zero, and a 1,024-token output ceiling. A contemporaneous FULL repeat was
exact in content and command on all 3/3 probes. These are synthetic next-action
diagnostics, not autonomous task-success evidence.

| Task / arm | Materialized retention | Exact command vs FULL | Diagnostic action |
|---|---:|---:|---|
| Discovery / strict H1 | 85.75% | no | Equivalent terminal no-op (`echo`) after verified fix |
| Versioned state / H2b | 78.53% | no | Equivalent terminal no-op (`echo`) after verified fix |
| Versioned state / H3 | 84.18% | yes | Exact FULL command |
| Long failure / fixed head-tail | 60.78% | no | Wrong phase: inspected diff before making the edit |
| Long failure / lexical matched span | 80.02% | no | Targeted edit on the correct file, different expression |
| Long failure / structured evidence | 66.88% | no | Targeted edit on the correct file, different expression |

The negative-rule controls show that rule activation is functioning: H3 can be
exactly inert when a same-version read is genuinely duplicated. H1 and H2b
still change wording/commands after removing a whole causal group, but in these
two probes both commands are terminal no-ops with the requested change already
verified. This demonstrates why exact command equality must be reported beside
completion-state and task-oriented equivalence rather than used alone.

The long-result diagnostic is the clearest materialization result. The decisive
assertion and source location occur in the middle of 856 tokenizer tokens of
tool output. Fixed head-tail omits that evidence and produces the wrong-phase
action. The existing lexical matched-span policy and the new structured-evidence
policy both produce targeted edits on the correct file rather than the exact
FULL command. Structured evidence retains less context: it removes 315 of 951
total request tokens (33.12%), compared with 190 (19.98%) for lexical matching.
It keeps traceback/failure lines, paths, diff structure, source symbols, and
task/query matches. Unknown output fails closed to the whole record instead of
silently reverting to positional sampling. The edit commands were not executed,
so this is action-phase/target evidence rather than task correctness.

Raw model artifacts and the machine-readable reduction are adjacent in the
task directories and in [`model_smoke_summary.json`](model_smoke_summary.json).
