# Paper 2.5: Associative Memory in Transformers

The reviewer-facing associative-memory narrative lives in
`main_associative_memory.tex`; `paper.tex` supplies the common preamble and the
complete chronological/negative-result appendices. The experiment program is
frozen against additional graph-search mechanisms.

Build from this directory with:

```powershell
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Primary experiment:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_closure.py --device cpu
python ../../../experiments/paper2_5_iterative_pra/summarize_results.py
```

Controlled receptive-field and causal activation diagnosis:

```powershell
python -m experiments.paper2_5_iterative_pra.run_controlled_local_sa `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6 `
  --device cuda --windows 16,32,64,128,global --seeds 17,29,41,53,67 `
  --steps 800 --d-model 96 --layers 6
python -m experiments.paper2_5_iterative_pra.run_controlled_pra `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6 `
  --device cuda --windows 16,32,64,128,global --seeds 17,29,41,53,67 `
  --examples 32 --d-model 96 --layers 6
python -m experiments.paper2_5_iterative_pra.run_toy_mechanistic `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6 `
  --device cuda --windows 16,32,64,128,global --seeds 17,29,41,53,67 `
  --examples 16
python -m experiments.paper2_5_iterative_pra.summarize_outcome_b `
  --output-dir docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6
```

The final two commands reuse frozen checkpoints. They do not train a router,
consumer, readout, or memory projection. Their matched causal cache conditions
and intermediate readouts delimit topology, traversal, controlled activation,
consumption, and later-layer preservation. The oracle intervention establishes
frozen consumption capacity; it is not an executable routing policy.

The compact routing sweep is faster on CPU for these small matrices.  The separate
`run_downstream_smoke.py` runner validates full native-K/V execution on CUDA.

Projection-correct and hierarchical-local gates:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_local_associative_closure.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/precompute_local_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_gate2_local_closure.py --device cuda
```

Gate 2 encodes 256-token contextual parents once and derives eight 32-token
local means without re-encoding subwindows. Results and schema-v2 graph traces
are under `../shared/results/paper2_5_iterative_pra/local_associative_closure/`.

Native local Q/K gate:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_native_qk_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_gate3_native_qk_closure.py --device cuda
```

Gate 3 uses the exact pinned Qwen revision and captures tokenwise layer-27
pre-RoPE Q/K from each single contextual parent encoding. The large regenerable
feature tensor is ignored; its manifest hash and all result/graph/plot artifacts
are under `../shared/results/paper2_5_iterative_pra/native_qk_closure/`.

Oracle convergence and edge-rank diagnostic:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_oracle_convergence.py --device cuda
```

This additive diagnostic imports the Paper-2 annotated-evidence oracle unchanged,
evaluates all five frozen routing methods at 5/10/20/30/40% budgets, and measures
the rank of true Hotpot transitions after forcing only the source evidence group.
It also runs a validation-only offline margin policy; it does not change the SDK
or train a router. Results are under
`../shared/results/paper2_5_iterative_pra/oracle_convergence/`.

Root displacement and score calibration diagnostic:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_displacement_calibration.py --device cuda
```

This runner requires row-for-row identity parity with the oracle-convergence
artifact, classifies one-shot oracle-hit displacement, computes a matched-budget
oracle-protected upper bound, audits root/semantic/native score paths, and tests
validation-only family z-score and empirical-quantile controls on held-out
examples. It remains offline and additive. Results are under
`../shared/results/paper2_5_iterative_pra/oracle_competition_diagnostics/`.

Monotonic root persistence and adaptive competition:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_monotonic_adaptive_competition.py --device cuda
```

This final frozen-policy gate computes full root Top-B before locking, selects
transparent lock and transition rules on validation only, and reports held-out
QASPER preservation plus the Hotpot entry/locking/propagation decomposition.
The policy implementation is opt-in and does not change one-shot SDK defaults.
Artifacts are under
`../shared/results/paper2_5_iterative_pra/monotonic_adaptive_competition/`.

Contextual query-facet and native-head entry gate:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_query_entry_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_query_entry_facets.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_query_entry_propagation.py --device cuda
```

The first command captures one complete contextual query pass and no independent
window encodings. The second freezes a validation-selected facet/head policy and
reports the matched-budget root matrix. The third runs only because the Hotpot
entry gain cleared the predeclared materiality gate; it reconnects the winning
root to the previously frozen monotonic propagation policy. Artifacts are under
`../shared/results/paper2_5_iterative_pra/query_entry_facets/`.

Facet-type, prompt-support, and grounded-propagation gates:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_grounded_query_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_grounded_facet_gate.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_grounded_propagation_gate.py --device cuda
```

These commands compare parameter-free facet families, capture controlled stale
chat history with an explicit latest-message boundary, and conditionally test
query validation inside native-QK Top-4 successor proposals. The conditional
gate fails on held-out edges, so no end-to-end grounded-closure experiment is
run. Artifacts are under
`../shared/results/paper2_5_iterative_pra/grounded_query_facets/`.

Frozen dynamic-query reconstruction gate:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_dynamic_query_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_dynamic_query_gate.py --device cuda
```

The first command re-encodes the current question with oracle-conditioned A in
three deterministic orderings. The second freezes an ordering and support scope
on validation identities, compares Q0 and Q1 over K=1/2/3/4/5/6/8/11, and
checks exact reproduction of the previous static and native-candidate
baselines. Gate 1 fails its held-out +0.10 R@1 criterion, so the conditional
F/K/B/theta surface is not run. Artifacts are under
`../shared/results/paper2_5_iterative_pra/dynamic_query_discovery/`.

Terminal-query semantic graph search:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_semantic_graph_search.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/review_semantic_graph_false_goals.py
```

The first command builds native local-QK edges on CUDA, calibrates the K/B/hop/
edge/goal surfaces from frozen score caches, and reruns the selected condition
on CUDA. Intermediate admission never uses query similarity. The terminal
query-goal gate fails, so routed roots are intentionally not run. The second
command decodes and classifies the validation q95 false terminals. Results are
under `../shared/results/paper2_5_iterative_pra/semantic_graph_search/`.

Hotpot chunk-granularity and oracle-discovery control:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_chunk_granularity_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_chunk_granularity.py --device cuda
```

The first command captures an ignored 52 MB token-hidden cache under the same
frozen 256-token contextual encoder. The second remaps evidence over
16/32/64/128/256-token zero-overlap parents and runs the oracle-root K/H/B
surface with terminal stopping disabled. It records actual routing cost and
counterfactual K/V payload separately. Tracked rows, tables, plots, and the
feature manifest are under
`../shared/results/paper2_5_iterative_pra/chunk_granularity/`.

MuSiQue/2Wiki annotated natural-graph gate:

```powershell
python ../../../experiments/paper2_5_iterative_pra/audit_natural_graph_datasets.py
python ../../../experiments/paper2_5_iterative_pra/precompute_natural_graph_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_natural_graph_depth.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_natural_multiscale_query_audit.py --device cuda
```

The audit freezes 36 MuSiQue and 48 2Wiki identities after checking official
schemas and mapping semantics. The ignored 1.32 GB cache contains one frozen
layer-27 source/query capture per example; its tracked manifest pins the hash.
The graph runner preserves the frozen 128-token operating point while sweeping
16/32/64/128/256-token search parents under one 256-token contextual encoder.
It scans the oracle-root K/H/B surface and asserts exact reproduction of the
canonical 2Wiki transition curve. The query audit then scores every valid
stride-one question span at scales 1/2/4/8/16 plus global, records post-hoc
root/terminal ceilings, and evaluates one globally bounded union. Raw mappings,
transitions, hop survival, systems metrics, facet diagnostics, routed rows,
plots, and bootstrap summaries are under
`../shared/results/paper2_5_iterative_pra/natural_graph_depth/`.

Layerwise native-graph and contextualization extension:

```powershell
python ../../../experiments/paper2_5_iterative_pra/precompute_layerwise_graph_features.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_layerwise_graph_exploration.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_layerwise_granularity_cross.py --device cuda
```

The capture streams exact pre-RoPE Q/K from layers 0/4/8/12/16/20/24/27 and
measures residual contributions plus position-preserving self/16/32-token
context interventions. The analysis holds the canonical 128-token graph fixed,
then conditionally runs the a-priori 0/12/27 by 32/128/256-token cross. Large
per-example feature files are ignored; their hashes and all metric rows, plots,
and result JSON are tracked under
`../shared/results/paper2_5_iterative_pra/layerwise_graph/`.

Final measurement-only synthesis:

```powershell
python ../../../experiments/paper2_5_iterative_pra/run_final_metrics.py
```

This command consumes the frozen experiment rows and caches without training,
generation, new graph search, or K/V materialization. It writes the facet
diagnostics, role visibility, matched layer correlations, edge/search
decomposition, sparse quality/payload frontiers, cross-dataset table, plots,
and negative-results registry under
`../shared/results/paper2_5_iterative_pra/final_metrics/`.

Final reviewer-response synthesis from frozen traces:

```powershell
python -m experiments.paper2_5_iterative_pra.analyze_final_reviewer_patch
```

This command joins the canonical 400 controlled model--example units, emits the
59-versus-341 pre-decision and post-treatment comparisons, and summarizes the
existing cross-dataset routing geometry and parameter directions. It performs
no training, routing, generation, or model inference. The derived artifacts are
under `../shared/results/paper2_5_iterative_pra/final_reviewer_patch/`.

Gate-3 end-to-end output validation:

```powershell
python ../../../experiments/paper2_5_iterative_pra/build_gate3_output_selections.py --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_gate3_output_validation.py --phase layer_sweep --device cuda
python ../../../experiments/paper2_5_iterative_pra/run_gate3_output_validation.py --phase heldout --device cuda
python ../../../experiments/paper2_5_iterative_pra/analyze_gate3_output_validation.py
python ../../../experiments/paper2_5_iterative_pra/build_gate3_judge_package.py
```

The first command freezes one-shot, sparse, balanced, broad, and oracle identity
sets before generation. The layer sweep chooses one consumption band per
dataset from validation balanced-graph outputs only; the held-out command then
runs C0--C6 with deterministic decoding. Analysis produces paired identity
bootstraps and source/KV/layer/depth figures under
`../shared/results/paper2_5_iterative_pra/output_validation/`. The final command
uses the existing Paper-2 A/B-reversal and calibration harness. Blinded items
are public, while `behavioral_judge_truth.json` is intentionally gitignored.
The optional `run_gate3_ollama_judge.py` runner is a supplementary local
protocol diagnostic and never makes a headline SOTA-judge result eligible.

External responses are validated and unblinded with the shared Paper-2 scorer:

```powershell
python ../../../experiments/paper2_hf/score_behavioral_judge_results.py `
  --truth ../shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge/behavioral_judge_truth.json `
  --responses ../shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge/responses/behavioral_judge_response_gpt56sol.json ../shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge/responses/behavioral_judge_response_claude_sonnet5_partial.json `
  --output ../shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge/behavioral_judge_external_metrics.json `
  --allow-partial
```

Strict full-ID coverage remains the default. `--allow-partial` is required here because the
Claude response contains 152 complete A/B-reversed pairs (51.7% coverage); cross-judge metrics
use only shared pairs and the report records coverage explicitly. GPT-5.6 Sol covers all 294
pairs and is the only judge that passes both identity and corruption calibration anchors.
