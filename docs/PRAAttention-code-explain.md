The current PRAttention implementation is now substantially richer than the earlier simple summary_vector → top-k reference → concatenate K/V prototype. The central architecture is:

training example
   │
   ├── prompt tokens ────────────────► normal causal Transformer
   │
   └── reference URIs
          │
          ▼
      resolve text
          │
          ▼
      split into chunks
          │
          ▼
      encode each chunk at every PRA layer
          │
          ├── routing gist   [D]
          └── detailed K/V   [1,H,M,Dh]
                    │
                    ▼
            PRAMemoryCache
                    │
                    ▼
last prompt query ─► route references/chunks
                    │
                    ▼
           materialize selected K/V
                    │
                    ▼
      bucket variable memory lengths
                    │
                    ▼
          memory cross-attention
                    │
                    ▼
 local attention + α · memory attention

That separation—cheap routing representation first, expensive detailed K/V only after selection—is now the key idea in the implementation.

1. attention.py: where PRA actually happens

PRAttention remains fundamentally a normal causal self-attention layer plus a second memory-attention branch.

The input is:

x: [B, T, D]

with:

B  batch size
T  local sequence length
D  d_model
H  number of heads
Dh = D/H

The projections produce:

q,k,v: [B,H,T,Dh]

and the first part of forward() is completely conventional causal attention:

scores = q @ k.transpose(-2, -1) / sqrt(Dh)
scores = causal_mask(scores)
weights = softmax(scores)
local_out = O(weights @ v)

So PRA does not replace self-attention. It augments it.

The important fork comes afterward.

Routing query

The code takes the query representation of the last input token:

routing_query = q[:, :, -1, :].contiguous().view(b, self.d_model)

so:

[B,H,Dh] → [B,D]

This vector answers:

Given what the model is currently processing, what external reference memory is relevant?

It then calls:

selected_by_batch =
    self.pra_cache.search(
        routing_query,
        self.layer_id,
        self.config
    )

The routing is therefore layer-specific. Layer 0 searches layer-0 representations; layer 3 searches layer-3 representations, etc.

That is an important design property of PRAttention: references do not have one universal embedding. They acquire representations in the semantic space of each Transformer layer.

2. Reference memory is stored twice conceptually

One of the most important things to understand in the current implementation is that PRA memory has two very different representations.

For every reference chunk it stores approximately:

ReferenceChunkMemory
    │
    ├── routing_gist
    │      └── [D]
    │
    └── token_kv
           ├── K [1,H,M,Dh]
           └── V [1,H,M,Dh]

The routing gist answers:

Should I retrieve this chunk?

The token K/V answers:

Once retrieved, what detailed information should attention read from it?

This is exactly the distinction we had discussed conceptually between reference/gist selection and materialization.

LayerKV contains full projected K/V:

k: [1, heads, memory_tokens, head_dim]
v: [1, heads, memory_tokens, head_dim]

while ChunkRoutingGist contains a compact d_model-sized key.

So a 1,000-token document does not need to participate with all 1,000 K/V positions just to determine whether it is relevant.

3. memory.py: the logical memory hierarchy

The hierarchy is roughly:

PRAMemoryCache
│
├── URI A → PRACacheEntry
│           │
│           ├── original resolved text
│           ├── child URIs
│           │
│           └── layer_memory
│                │
│                ├── layer 0
│                │    ├── chunk 0
│                │    ├── chunk 1
│                │    └── chunk 2
│                │
│                ├── layer 1
│                │    ├── chunk 0
│                │    ├── chunk 1
│                │    └── chunk 2
│                │
│                └── ...
│
└── URI B → ...

A PRACacheEntry corresponds to a resolved reference URI:

class PRACacheEntry:
    uri
    text
    layer_memory
    child_uris
    metadata

and each layer contains a LayerReferenceMemory consisting of independently routable chunks.

This means there are actually three hierarchy levels already visible in the code:

URI
  ↓
chunk
  ↓
tokens

and PRA can progressively narrow:

many URIs
   ↓
few URIs
   ↓
few chunks
   ↓
selected token K/V

That is much closer to the intended PRAttention design than the original implementation.

4. How routing works in memory.py

Before looking at the specific policies, the basic scoring is simple.

Each chunk has a gist vector:

g_i ∈ R^D

and the current layer has query:

q ∈ R^D

The similarity is cosine:

s
i
	​

=
∥q∥∥g
i
	​

∥
q⋅g
i
	​

	​


The code normalizes both and calculates the dot product. Importantly, it first searches gists, not complete token K/V.

For every chunk the cache keeps enough metadata to eventually create a SelectedChunk:

URI
chunk ID
reference score
chunk score
layer
reference rank
rank within reference
token offsets

This is partly computational, partly experimental infrastructure: you can later determine exactly why some memory was retrieved.

5. Hierarchical routing

One routing mode is _hierarchical().

Suppose we have:

doc A:
    A1 score .91
    A2 score .72
    A3 score .14

doc B:
    B1 score .83
    B2 score .82

doc C:
    C1 score .40

It first aggregates chunk scores to derive a reference-level score:

A = aggregate(.91,.72,.14)
B = aggregate(.83,.82)
C = aggregate(.40)

Then:

top_k_references

selects the winning documents.

Only afterward:

top_k_chunks_per_reference

selects chunks inside them.

So if:

top_k_references = 2
top_k_chunks_per_reference = 2

you might get:

A1 A2 B1 B2

rather than simply the four globally highest chunks.

That distinction matters because it prevents one large reference with many similar chunks from automatically monopolizing retrieval.

6. reference_first is a more explicit two-stage version

There is another interesting mode:

search_strategy = "reference_first"

Here the system explicitly creates a reference-level gist from its chunk gists.

Currently supported:

mean
last

Conceptually:

g
ref
	​

=
N
1
	​

i
∑
	​

g
i
	​


or:

g
ref
	​

=g
N
	​


Then references are ranked using those explicit URI-level representations, and chunks are ranked only inside the winning references.

There is also a placeholder for:

GRU

but the implementation deliberately raises NotImplementedError because a proper learned reference aggregator needs to be registered as a module rather than silently constructed inside the cache.

That is architecturally the right choice.

7. gist_mode versus detail_materialization

These are easy to confuse but represent different decisions.

gist_mode determines:

How do I represent a chunk cheaply for retrieval?

Examples are conceptually:

mean K
last K
GRU/gated representation
possibly multiple gists

detail_materialization determines:

After I select something, how much actual memory do I expose to attention?

The code currently supports three particularly useful experimental modes.

gist_only

One memory position per selected chunk:

selected chunk
     ↓
routing gist
     ↓
[1,H,1,Dh]

No detailed token K/V is inserted.

selected_chunks

The normal mode:

selected chunk
     ↓
all K/V tokens from that chunk
full_reference

Routing still selects a reference using particular chunks, but once the URI wins, all chunks belonging to that reference are materialized.

These three modes form an excellent ablation:

gist_only
     ↓
selected_chunks
     ↓
full_reference

They test whether progressive retrieval is actually doing useful selective work.

8. _materialize() is the bridge between retrieval and attention

This is one of the most conceptually important methods.

Before _materialize() the objects are mostly:

URI
chunk
scores
routing gist
metadata

After _materialize() you have actual tensors:

K: [1,H,M,Dh]
V: [1,H,M,Dh]

that can participate in attention.

It also handles overlapping chunks.

Suppose chunking generated:

chunk 1 = tokens 0–128
chunk 2 = tokens 96–224

Naively concatenating them would duplicate tokens 96–128.

The code tracks:

covered_end_by_uri

and removes overlapping K/V prefixes so the same source tokens aren't materialized twice.

That is a small implementation detail with quite important experimental consequences: otherwise changing chunk overlap would accidentally change the effective amount of memory attention.

9. Memory attention is ordinary cross-attention after retrieval

Once memory has been selected, there is nothing exotic about the mathematical attention operation.

For selected memory:

Kmem: [B,H,M,Dh]
Vmem: [B,H,M,Dh]

and local queries:

Q: [B,H,T,Dh]

the branch computes:

A
m
	​

=softmax(
d
h
	​

	​

QK
m
T
	​

	​

)

and:

Y
m
	​

=A
m
	​

V
m
	​


Then:

memory_out = mem_o_proj(...)

and finally:

Y=Y
local
	​

+αY
memory
	​


where memory_alpha is α.

So PRAttention's novelty is primarily in:

addressing
resolution
representation
routing
progressive selection
materialization

—not in inventing a new dot-product attention primitive.

That is an important distinction when describing the architecture in the paper.

10. Why memory_batching.py exists

Here there is a practical GPU problem.

Suppose a batch contains four examples and routing selects:

example 0 → 400 memory tokens
example 1 → 35
example 2 → 375
example 3 → 20

You cannot directly make:

[B,H,M,Dh]

because each M
i
	​

 is different.

The brute-force solution is:

Mmax = 400

0 → 400
1 → 400   365 padding
2 → 400    25 padding
3 → 400   380 padding

That wastes enormous computation.

memory_batching.py provides a better solution.

11. Memory buckets

The rows are sorted/grouped by memory length.

For example:

20, 35        → bucket 1 width 35
375, 400      → bucket 2 width 400

instead of one huge:

4 × 400

rectangle.

The important guarantee is:

Bucketing is only an efficiency mechanism. Batch examples never attend to each other's memory.

Each row keeps its own mask and its original index is restored afterward.

12. memory_bucket_count

This parameter has a nice interpretation.

0

Effectively:

one bucket per example

Almost no padding, but many small attention operations.

1
one rectangle for the entire batch

Maximum GPU parallelism, potentially lots of padding.

N > 1

A compromise:

at most N rectangles

That gives you an explicit performance tradeoff between:

kernel efficiency ↔ padding waste

The code can choose equal_count grouping or optimal_contiguous.

13. optimal_contiguous is actually solving an optimization problem

After sorting memory lengths:

l1 ≤ l2 ≤ ... ≤ ln

the cost of putting items i..j in one bucket is approximately:

(j−i+1)⋅l
j
	​


because every item must be padded to the longest sequence.

The dynamic-programming planner chooses bucket boundaries minimizing total allocated positions.

That is a good implementation of the batching idea we had discussed.

14. padded_memory_attention()

Once a bucket has compatible examples, each row gets padded to that bucket's maximum:

K: [Bbucket,H,Mmax,Dh]
V: [Bbucket,H,Mmax,Dh]

A Boolean mask says:

row 0 valid: 0...M0
row 1 valid: 0...M1
...

The attention score is:

[B,H,T,Dh]
      ×
[B,H,Dh,M]
      =
[B,H,T,M]

Padding positions receive effectively −∞ before softmax.

It also records:

attention entropy
maximum attention weight
padding fraction
valid memory positions
allocated memory positions

which is excellent for analyzing whether PRA retrieval becomes focused or diffuse.

15. One subtle point: routing is based on the final token, but memory serves all tokens

This part deserves attention.

Routing uses:

q[:, :, -1, :]

meaning the final current token chooses memory.

But after retrieval the selected memory is attended by:

q   # full [B,H,T,Dh]

So all T positions read the same selected memory set.

Conceptually:

last token:
    "Which memory do we need?"

all tokens:
    "Now interact with that memory."

This is computationally cheap and reasonable for the first implementation.

But it is not yet the strongest PRA formulation. Eventually you could route using:

explicit REF token positions
selected query positions
segment queries
multiple routing queries
per-head routing

rather than only the final token.

16. train.py is deliberately ignorant of PRA

This refactor is good.

train.py is now a generic training engine.

Its job is:

model
optimizer
scheduler
mixed precision
gradient accumulation
gradient clipping
validation
checkpointing
logging
early stopping

It knows almost nothing about references or PRA.

The key extension point is:

batch_step(...)

The generic trainer basically does:

with autocast:
    loss, metrics = batch_step(model, batch, device)

loss.backward()
optimizer.step()

PRA injects a specialized batch_step; ordinary LM training can inject another one.

Architecturally this is much cleaner than putting cache construction, reference retrieval, optimizer logic and evaluation in one giant training file.

17. TrainingState

TrainingState packages all mutable runtime machinery:

model
optimizer
scheduler
device
checkpoint manager
logger
grad scaler
early stopping
global step
batch step
best validation loss
...

So functional APIs and higher-level wrappers can share exactly the same training implementation.

The distinction:

batch_step
global_step

is particularly useful because gradient accumulation means:

4 batches consumed
≠
4 optimizer updates
18. pra_train.py plugs PRA into train.py

This file is the glue layer.

The relationship is essentially:

train.py
   generic mechanics
        ▲
        │ injected callbacks
        │
pra_train.py
   PRA semantics

The main specialized function is:

_pra_batch_step()

It:

moves tensors to the device,
builds reference memory for each sample,
executes the model,
gathers selected chunks,
gathers routing diagnostics,
reconstructs the logical batch,
computes ordinary next-token cross-entropy.

So PRA is still trained with normal LM loss here.

There is no special retrieval loss in the basic path.

That means retrieval quality emerges indirectly because good routed memory can help reduce language-model loss.

The explicit retrieval labels are mainly used for evaluation/diagnostics.

19. A notable current limitation in batching

Look carefully at _pra_batch_step():

for index, metadata in enumerate(batch["metadata"]):
    cache = build_cache_from_metadata(... [metadata] ...)

    logits_by_example.append(
        model(input_ids[index:index+1])
    )

So although the generic loader supplies batch size B, PRA currently performs B separate model forwards of batch size 1.

Then:

logits = torch.cat(logits_by_example, dim=0)

reconstructs [B,T,V].

This is done to guarantee:

Sample A's reference table cannot leak into sample B.

Conceptually correct.

Computationally, however, it means the sophisticated memory_batching.py machinery is not yet fully exploited during this path, because the model itself sees one training example at a time.

This is probably the most important implementation issue I would address next.

The desired architecture should eventually be:

logical batch B
      │
      ├── cache/reference set 0
      ├── cache/reference set 1
      ├── ...
      └── cache/reference set B-1
              │
              ▼
    ONE transformer forward B
              │
              ▼
PRA routing keeps memory row-local
              │
              ▼
dynamic_memory_attention buckets M_i

That would finally let memory_batching.py solve the problem it was designed to solve at training scale.

20. Why per-example caches currently exist

The comment explains the motivation correctly:

each prompt has its own runtime reference table

Imagine:

batch row 0:
    !!ref:doc/A!!

batch row 1:
    !!ref:doc/B!!

You absolutely do not want row 1 accidentally retrieving A because the cache is globally shared.

So the invariant must be:

M
i
	​

⊥M
j
	​


for i

=j, except in tensor packing.

The current implementation achieves that by serializing forwards.

A more performant design would represent the cache as:

BatchPRAMemoryCache:
    row 0 → cache A
    row 1 → cache B
    ...

and make search() return one independent set per row.

The attention implementation is already surprisingly close to supporting that because selected_by_batch, variable memory_k, and dynamic_memory_attention() are written batch-wise.

21. pra_train.py measures PRA separately from language modelling

This is one of the strongest parts of the current experiment infrastructure.

It tracks things such as:

reference hit@1
reference hit@k
MRR
NDCG

chunk hit@k
chunk recall
chunk precision

selected references
selected chunks
selected tokens

fraction of available memory selected
routing score
zero-selection fraction

recursive expansion depth
recursive reference budget
recursive token budget

memory padding fraction
attention entropy

This is important because perplexity alone cannot tell whether PRA works.

For example, you could obtain good perplexity while:

retrieving everything

which would defeat the entire purpose.

You want to show something closer to:

LM quality≈full context

while:

available tokens
retrieved tokens
	​

≪1

That is exactly what these metrics allow you to measure.

22. The ablation framework is particularly useful

evaluate_reference_ablation() already implements several causal tests.

Among them:

valid
disabled
empty
shuffled
irrelevant
oracle

oracle_chunks
shuffled_chunks
irrelevant_chunks

gist_only
selected_chunks
full_reference

These let you distinguish very different hypotheses.

For example:

valid >> disabled

means external memory helps.

But:

valid ≈ shuffled

would imply the model isn't actually using the correct memory.

Likewise:

selected_chunks ≈ full_reference

while using far fewer tokens would be strong evidence for progressive materialization.

And:

gist_only << selected_chunks

would demonstrate that detailed K/V carries information that the routing gist alone cannot provide.

This is exactly the sort of causal ablation reviewers will expect.

23. build_cache_from_metadata() is actually another central component

Although you didn't list cache_services.py, conceptually it sits between pra_train.py and the files you've asked about:

metadata / reference URI
          │
          ▼
 build_cache_from_metadata()
          │
          ├── resolver
          ├── recursive expansion
          ├── chunking
          ├── Transformer encoding
          ├── layer K/V projection
          └── gist construction
          │
          ▼
     memory.py structures

pra_train.py explicitly calls it before every sample forward.

So to understand how mean, last, GRU-style gist creation, chunking, recursive references, cache fingerprints, etc. are constructed, cache_services.py is the next file worth tracing.

24. Putting the complete runtime together

For one sample, suppose the prompt is:

Question:
Why did the experiment fail?

<REF_START> doc://experiments/run17 <REF_END>
<REF_START> doc://notes/debugging     <REF_END>

PRA roughly does this.

                 PREPARATION

doc://run17
      │
      ▼
 resolve document
      │
      ▼
 [chunk A1][A2][A3][A4]
      │
      ├──────────── layer 0 ────────────┐
      │                                 │
      │     gist A1      token K/V A1   │
      │     gist A2      token K/V A2   │
      │     ...                         │
      │                                 │
      ├──────────── layer 1 ────────────┤
      │                                 │
      │     gist A1      token K/V A1   │
      │     ...                         │
      │                                 │
      └─────────────────────────────────┘

same for debugging reference

Then at Transformer layer l:

prompt hidden states X_l
        │
        ├──── local Q/K/V
        │       │
        │       ▼
        │   causal attention
        │
        │
        └──── last Q
                │
                ▼
          routing query
                │
                ▼
       compare with chunk gists
                │
                ▼
           select URIs
                │
                ▼
          select chunks
                │
                ▼
     threshold + deduplicate
                │
                ▼
         materialize K/V
                │
                ▼
       cross-attention(Q,Km,Vm)
                │
                ▼
 local_out + α memory_out
                │
                ▼
        Transformer continues

That is the core PRAttention implementation today.

The most important conceptual interpretation

I would describe this implementation as three different memory scales, rather than simply “attention over references”:

Address memory
    URI / reference identity
          ↓

Semantic routing memory
    compact layer-specific chunk gists
          ↓

Detail memory
    layer-specific token K/V

and the computation progressively opens them:

URI→gist→chunk→token K/V.

That is the architectural aspect that differentiates PRA most clearly from simply putting an external KV cache beside the Transformer.

There are also two particularly important implementation observations from reading this version. First, the code now correctly uses layer-specific projected key-space representations for routing and memory, rather than the much weaker old idea of a global embedding summary. Second, the attention/memory-batching code is already designed for true batch-isolated variable retrieval, but _pra_batch_step() currently serializes the batch into B × batch_size=1 model calls; removing that serialization is likely the next major performance/refactoring target.

If you want, I can next go one level deeper and trace cache_services.py + model.py + config.py line by line, because those three complete the missing half of the story: exactly how a raw referenced document becomes chunks → layer hidden states → K/V → mean/last/gated gists → recursive PRA memory.