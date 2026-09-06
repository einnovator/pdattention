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
  --output docs/papers/shared/results/paper3_3_sparse_crossdoc/oracle_natural_seed11
```

Each output contains immutable candidate and selection receipts, canonical
`ContextRecord` identities, compressed teacher graphs, graph/plan digests,
host-path parity diagnostics, rows, summaries, localization, and the
prespecified gate decision.

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
