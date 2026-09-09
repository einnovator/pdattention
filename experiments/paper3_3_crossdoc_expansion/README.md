# Paper 3.3 cross-document retrieval expansion

Install the research dependencies with `pip install -e '.[research]'`. Natural
runs use deterministic BM25-v2 receipts; v2 counts document occurrence for IDF
and sorts query terms before floating-point accumulation.

This experiment keeps first-stage retrieval, reranking, selected records, and
initial selected spans frozen. A selected span may search only peer selected
records. The runtime then enforces authorization, interval deduplication, and a
global extra-token budget before emitting source-addressed materialization
spans and a hash-linked receipt.

Run the correctness fixture:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.run_expansion_frontier \
  --dataset fixture --max-examples 5 --candidate-count 25 \
  --chunk-tokens 32 --chunk-overlap 4 --max-resources 2 \
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/fixture_smoke
```

Run the frozen natural validation cohort with the inherited strong reranker:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.run_expansion_frontier \
  --dataset multihoprag --split-name validation --max-examples 150 \
  --selector cross_encoder --reranker BAAI/bge-reranker-v2-m3 \
  --dense-model BAAI/bge-base-en-v1.5 \
  --selection-cache .runs/paper3_3_validation_selection.jsonl \
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/validation_n150
```

The dense and RRF arms use the pinned sentence-transformer's distinct query and
document encodings. The runtime retains a dependency-free signed-hash fallback,
but powered semantic results must identify the model and immutable revision in
their summary. The output records support-span/chunk recovery, distractor fraction, requested
and deduplicated expansion tokens, already-resident versus newly materialized
native tokens, search latency, and complete policy receipts. These are
retrieval/mechanism measurements. They are not answer F1 or an SDK promotion
until the generation, sparse-SA, five-seed, scale, and family gates pass.

After selecting a policy and budget on validation, replay that frozen choice
through the native MLX path:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.run_expansion_generation \
  --cache-dir .cache/rag_eval --split-name test --max-examples 150 \
  --model mlx-community/Qwen3-1.7B-4bit \
  --mode dense --no-query-conditioned --top-k 1 \
  --cross-token-budget 64 \
  --selection-cache .runs/paper3_3_test_selection.jsonl \
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/generation_test_qwen17b
```

The JSONL cache freezes tokenizer-independent source intervals and validates
the candidate receipt, selector revision, token budget, and resource limit on
every replay. Model-specific native token counts are measured only when those
fixed intervals are encoded. This keeps policy and cross-model comparisons
paired while avoiding repeated cross-encoder inference.

The dense selected-only, k=1, 64-token policy above is fixed by the corrected
BM25-v2 validation frontier. Do not replace it using test retrieval or answer
metrics. Summarize a held-out retrieval replay with `summarize_expansion
--frozen-config <validation-publication-summary>` and a generation run with
`summarize_generation`; both commands preserve the cache digest and distinguish
bootstrap resampling seeds from independent model trials.

The generation runner reports eight separately named conditions. Four form an
explicit expansion-by-interaction factorial:

| Condition | Added evidence | Cross-record interaction |
| --- | --- | --- |
| `INDEPENDENT_PRA` | no | no |
| `PAIR_SA_ONLY` | no | retrieval-linked original record pairs |
| `EXPANSION_ONLY` | yes | no |
| `CROSSDOC_EXPANSION_PAIR_SA` | yes | retrieval-linked pairs |

Packed RAG is the dense causal reference. `PAIR_SA_ONLY_BOUNDARY` and
`CROSSDOC_EXPANSION_PAIR_SA_BOUNDARY` are matched boundary-token controls, and
the linked-region physical-edge oracle is a mechanism control. The interaction
conditions reuse Paper 3.3's host attention instrumentation and are not
sparse-kernel speed measurements.

Run the same frozen test identities at both audit budgets by changing only
`--token-budget 512` to `--token-budget 1024`. Each output row records
`source_token_budget` and immutable selection and expansion receipt identities.
For the larger-budget interaction matrix, `--interaction-audit-only` skips the
already-qualified expansion-only and linked-expansion arms while retaining the
same frozen link discovery, packed reference, independent baseline, pair-SA,
and boundary-SA conditions.

Reduce the two canonical runs and retain an optional second-host replay with:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.summarize_factorial \
  --run-512 <same-host-512-run> \
  --run-1024 <same-host-1024-run> \
  --replication-512 <optional-second-host-512-run> \
  --output docs/papers/shared/results/paper3_3_crossdoc_expansion/factorial_budget_audit
```

The reducer rejects duplicate rows, incomplete condition cohorts, mismatched
frozen identities, and model mismatches. It records each run's environment and
immutable revisions and emits absolute-quality and paired-effect plots with
conservative bootstrap envelopes.

## Task-aware boundary-region-layer oracle

The coarse document-pair selector is closed by the powered Paper 3.3 gate. The
next mechanism audit therefore holds retrieval links fixed and scores a finer
consumption unit:

```text
equal-width source token region
  x equal-width target token region
  x contiguous decoder-layer band
```

Prefix, middle, and suffix windows are geometrically matched. In particular,
source-suffix to target-prefix is the boundary hypothesis, while the other
eight region pairs are equal-cost controls. Each singleton cell executes the
original model attention and is scored by its incremental gold-answer NLL gain
from the no-cross-document packed state. The best singleton is decoded as
`TASK_ORACLE_REGION_LAYER_SINGLETON`. Independently positive cells are also
unioned and decoded as `TASK_ORACLE_REGION_LAYER_UNION`; this is a separate
consumption test because singleton utilities need not compose additively.

Run the first locked 1,024-token validation smoke on Apple Silicon:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.run_expansion_generation \
  --cache-dir .cache/rag_eval --split-name validation --max-examples 1 \
  --model mlx-community/Qwen3-1.7B-4bit \
  --mode dense --no-query-conditioned --top-k 1 --cross-token-budget 64 \
  --token-budget 1024 \
  --interaction-audit-only --region-layer-audit \
  --region-tokens 8 --layer-band-count 4 --oracle-cells 4 \
  --selection-cache .runs/paper3_3_region_layer_validation_1024.jsonl \
  --resume --output .runs/paper3_3_region_layer_validation_smoke
```

This is a gold-answer oracle headroom experiment, not selector training or a
deployable policy. `region_layer_diagnostics.jsonl` keeps selection-signal
measurements separate from the generated consumption outcomes in `rows.jsonl`.
Gold-answer log probabilities are reduced in float32 even when the quantized
model emits lower-precision logits, preventing small causal effects from being
rounded into coarse NLL steps. Per-question checkpoints are configuration-bound
and make `--resume` safe after a host or SSH interruption.
The logical masks currently execute through a dense host kernel, so latency is
diagnostic only. Do not begin learned-policy training unless this finer oracle
recovers a useful fraction of the locked 1,024-token packed deficit. Independent
training seeds apply only after that gate; the oracle uses paired question
bootstrap intervals and must later transfer to the reduced Qwen3-4B/Qwen3-8B
and Llama-3.1-8B cohorts before any systems claim.

Reduce a completed audit with strict cell/condition completeness checks:

```bash
PYTHONPATH=src python -m experiments.paper3_3_crossdoc_expansion.summarize_region_layer \
  --run-dir .runs/paper3_3_region_layer_validation_n10 \
  --output .runs/paper3_3_region_layer_validation_n10/publication
```

For the long remote test cohort,
`reduce_region_layer_after_audit.sh` waits for the runner's final
`summary.json`, refuses a partial/failed run, and then invokes the same strict
reducer.

The reducer reports `selection_quality` (singleton NLL utility and the
suffix-to-prefix boundary contrast against eight equal-cost controls) separately
from `consumption_quality` (realized NLL, F1, and official-score changes for the
singleton and ranked union). Bootstrap seeds are resampling seeds, not model
training trials. Its predeclared `controller_training_gate` passes only when
the primary singleton condition is evaluated on at least 150 identities from
the frozen test partition and the conservative paired-bootstrap 95% interval
for token-F1 gain lies strictly above zero. NLL is explicitly ineligible to
satisfy this gate. A pass authorizes multi-seed compact-controller training; a
failure keeps training locked.
