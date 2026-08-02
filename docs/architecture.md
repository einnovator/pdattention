# Architecture: URI-Addressed Progressive Retrieval Attention

## Motivation

Current long-context and RAG systems usually expose retrieved context as flat prompt text. Coding agents use a more efficient pattern: progressive disclosure. They first expose tool/skill descriptions, then load full bodies only when selected.

PRA brings that pattern into model/runtime inference.

## Reference handles

A reference handle is an explicit latent context link represented by a lightweight token in the prompt and a runtime table entry:

```text
<REF_1>
```

Examples:

```text
<REF_1> -> mem://doc42#summary
<REF_2> -> mem://repo/file.py#class.Agent.run
<REF_3> -> search://workspace?q=kv-cache+attention#top3.summary
```

## URI + anchor hierarchy

A URI can point to a document, fragment, search result, or anchor.

Suggested anchor syntax:

```text
scheme://host/path#level1.level2.level3
```

Recursive expansion example:

```text
mem://manual#summary
  -> mem://manual#installation
     -> mem://manual#installation.cuda
        -> mem://manual#installation.cuda.vllm
```

## Runtime objects

- `ReferenceHandle`: table entry containing id, token, URI, summary, and metadata.
- `ReferenceTable`: runtime mapping from `<REF_n>` tokens to handles.
- `Resolver`: maps URI to text, summary, metadata, children.
- `PRAMemoryCache`: stores resolved references.
- `PRACacheEntry`: stores summary vector and per-layer K/V.
- `LayerKV`: K/V tensors for a specific transformer layer.

## Inference flow

1. Prompt contains references.
2. Runtime parses `<REF_n>` tokens.
3. Runtime resolves tokens through `ReferenceTable`, then fetches summaries or fragments.
4. Referenced fragment is encoded separately.
5. PRA-enabled layers cache K/V for each layer.
6. During main inference, PRA layers retrieve relevant reference K/V.
7. The layer attends/cross-attends to selected memory.
8. If only summary was retrieved and confidence remains low, expand child anchors.

## Why layer-specific K/V?

A reference memory must not reuse one embedding representation for all layers. Each layer has its own K/V projections and semantic level.

Correct:

```text
cache[uri][layer_id] = (K_layer, V_layer)
```

Incorrect except as toy shortcut:

```text
cache[uri] = token_embeddings
```

## Cross-attention vs K/V concatenation

For standalone training, both can be tested.

For pretrained HF models, start with cross-attention adapter:

```text
x = x + self_attn(x)
x = x + alpha * ref_cross_attn(x, ref_memory)
x = x + mlp(x)
```

Direct K/V concatenation into pretrained attention is harder because of RoPE, GQA/MQA, cache layout, and position semantics.

## Evaluation plan

Compare:

1. No retrieval.
2. Full context.
3. Standard RAG.
4. Summary-first RAG.
5. PRA single-level expansion.
6. PRA recursive expansion.

Metrics:

- exact answer accuracy
- token cost
- latency
- number of resolved refs
- cache hit rate
- distractor robustness
