# Paper 3.3 Oracle Sparse Cross-Document Contextualization

This experiment asks a deliberately narrow first question: can a small subset
of the packed teacher's cross-document attention recover packed-RAG quality?
Retrieval, reranking, record order, token budget, prompt text, model revision,
and generation settings are frozen. Only causal document-to-document edges are
changed.

The implementation stores one score per physical
`(layer, head, target token, source token)` edge, restricted to causal
cross-record pairs. It therefore avoids the irrelevant within-record part of
the dense `[layer, head, token, token]` tensor while preserving the exact
head-level oracle unit required by the study.

## Commands

Freeze the future learned-policy split:

```bash
PYTHONPATH=src python -m experiments.paper3_3_sparse_crossdoc.freeze_splits \
  --cache-dir .cache/rag_eval \
  --output docs/papers/shared/results/paper3_3_sparse_crossdoc/splits.json
```

Run a fixture mechanism smoke:

```bash
PYTHONPATH=src python -m experiments.paper3_3_sparse_crossdoc.run_oracle_sparsity \
  --dataset fixture --max-examples 1 --token-budget 384 \
  --output docs/papers/shared/results/paper3_3_sparse_crossdoc/oracle_fixture
```

Run the small natural inception gate on Apple Silicon:

```bash
PYTHONPATH=src python -m experiments.paper3_3_sparse_crossdoc.run_oracle_sparsity \
  --dataset multihoprag --max-examples 3 --token-budget 512 \
  --model mlx-community/Qwen3-1.7B-4bit \
  --split-name validation \
  --output docs/papers/shared/results/paper3_3_sparse_crossdoc/oracle_natural_seed11
```

Each output contains immutable candidate and selection receipts, canonical
`ContextRecord` identities, compressed teacher graphs, graph/plan digests,
host-path parity diagnostics, rows, summaries, localization, and the
prespecified gate decision.

Run the interventional diagnostic on the frozen test split:

```bash
PYTHONPATH=src:. python -m experiments.paper3_3_sparse_crossdoc.run_oracle_sparsity \
  --dataset multihoprag --max-examples 10 --token-budget 512 \
  --model mlx-community/Qwen3-1.7B-4bit \
  --ranking-targets attention,pair_nll,pair_js,pair_nll_x_attention,layer_nll,layer_js \
  --edge-percentages 0,0.01,0.05,0.1,0.5,1,100 \
  --skip-mass-frontier --resume \
  --output docs/papers/shared/results/paper3_3_sparse_crossdoc/interventional_n10
```

The document-pair and layer rankings use leave-one-group-out interventions.
`pair_nll_x_attention` multiplies the finite-difference NLL effect by teacher
attention within the pair; it is not an autograd attribution. `layer_head_nll`
and `layer_head_js` are available for targeted diagnostics but are intentionally
excluded from the default command because a full layer/head ablation is much
more expensive.

For the powered gate, use the same command with `--split-name test`,
`--max-examples 100`, and omit the two layer targets after the ten-question
validation localization identifies the useful hierarchy. The runner restricts
sampling to the named frozen split in `splits.json`, writes one atomic
checkpoint per question, and rejects resume when the run configuration differs.
The learned selector remains locked unless a powered test frontier reaches the
absolute quality gate and is monotonic in token F1.

## Learned Pair Selector, Conditional Design

Training remains locked unless the oracle reaches the inception gate. If it
does, the first selector predicts a score for each ordered source/target record
pair from frozen features:

```text
query embedding
+ source and target record gists
+ reranker scores/ranks
+ lexical overlap and entity overlap
+ source/target lengths and boundary features
```

It first chooses pairs, then allocates a bounded token-edge budget within each
chosen pair. Query-independent, query-conditioned, fixed-boundary, and oracle
conditions remain separate baselines. The loss combines answer NLL, packed
teacher KL, oracle selector supervision, and a small interaction-cost penalty;
selection and task metrics are reported independently. Persistent record K/V
is immutable and every request-local plan carries the source selection receipt
and teacher graph digest.
