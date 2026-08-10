# AGENTS.md — PRAttention Long-Prompt Support via Implicit Prompt References

## Mission

Extend PRAttention so that prompts longer than the configured direct-attention budget can reuse the existing reference/chunk/gist/cache/routing machinery instead of being simply truncated.

The key mechanism is an **implicit prompt reference** representing the early prefix of an oversized prompt.

If:

- the prompt has `N` tokens;
- direct prompt access is limited to `D` tokens;
- `N > D`;

then:

- the most recent `D` prompt tokens remain in the normal direct self-attention window;
- the earlier `N-D` tokens become a synthetic PRA reference, logically named `#__head`;
- the synthetic reference is chunked, encoded, summarized into gists, cached, routed, and materialized using the same machinery already used by explicit references.

Example:

```text
prompt length = 32768
max_prompt_direct_tokens = 4096

original:
[P0 ...................................................... P32767]

implicit reference #__head:
[P0 ...................................................... P28671]

direct prompt:
[P28672 .................................................. P32767]
```

The goal is to make long prompts a thin extension of the current PRA reference abstraction, not to build a separate long-context subsystem.

---

# 1. Architectural Principle

Treat displaced prompt history as another PRA memory source.

Conceptually:

```text
very long prompt
    |
    +-- early prefix --> implicit reference #__head
    |                       |
    |                       +-- existing reference chunking
    |                       +-- existing layer-specific K/V encoding
    |                       +-- existing gist computation
    |                       +-- existing PRA cache
    |                       +-- existing routing
    |                       +-- existing K/V materialization
    |
    +-- recent tail ----> ordinary causal self-attention
```

Do not add a separate long-prompt attention implementation.

Do not duplicate routing logic.

Do not special-case implicit prompt memory inside the core `PRAttention` layer unless absolutely necessary.

The attention layer should ideally remain agnostic to whether cached memory originated from:

- an explicit external reference;
- a recursive reference;
- a prompt prefix;
- future persistent/streaming memory.

---

# 2. Required User-Facing Behavior

Add support for a configurable direct prompt limit.

Recommended configuration:

```python
max_prompt_direct_tokens: int | None = None
prompt_overflow_mode: str = "truncate"
max_prompt_gists: int | None = None
```

Semantics:

```text
max_prompt_direct_tokens = None
    -> inherit max_seq_len

prompt_overflow_mode = "truncate"
    -> preserve current behavior

prompt_overflow_mode = "implicit_reference"
    -> convert old prompt prefix to #__head

prompt_overflow_mode = "error"
    -> reject prompts exceeding max_prompt_direct_tokens
```

`max_prompt_gists` applies to the implicit prompt reference only.

Recommended semantics:

```text
max_prompt_gists = None
    -> do not impose the ordinary explicit-reference gist-count cap
       on the implicit prompt head

max_prompt_gists = K
    -> limit the number of chunks/gists associated with #__head to K
       using the same overflow policy semantics as appropriate
```

Backward compatibility is mandatory.

Existing configurations that do not specify the new fields must behave exactly as before.

---

# 3. Synthetic Reference Identity

Use an internal URI that cannot collide with user document references.

Recommended:

```text
pra://implicit/prompt/head
```

Logical/debug/display name:

```text
#__head
```

Do not require the literal text `#__head` or an explicit reference marker to be inserted into the visible prompt.

The implicit reference should be available to the PRA router through the memory cache directly.

Add metadata identifying provenance, for example:

```python
{
    "implicit": True,
    "source": "prompt",
    "kind": "head",
}
```

If the implementation supports per-reference gist/chunk limits through metadata, include the prompt-specific override there.

---

# 4. Prompt Splitting

Implement a small preprocessing function in an appropriate prompt/cache preparation module.

Conceptually:

```python
def prepare_prompt_for_pra(input_ids, config):
    direct_limit = (
        config.max_prompt_direct_tokens
        if config.max_prompt_direct_tokens is not None
        else config.max_seq_len
    )

    if input_ids_length <= direct_limit:
        return unchanged_prompt, no_implicit_reference

    if config.prompt_overflow_mode == "error":
        raise ...

    if config.prompt_overflow_mode == "truncate":
        return last_direct_limit_tokens, no_implicit_reference

    if config.prompt_overflow_mode == "implicit_reference":
        prefix = all_tokens_except_last_direct_limit
        tail = last_direct_limit_tokens
        return tail, implicit_reference(prefix)
```

The split must happen on token IDs, not characters.

The direct tail must preserve exact token order and causal semantics.

The implicit prefix must contain exactly the displaced token sequence.

---

# 5. Avoid Token Decode/Re-Encode if Reasonably Possible

The prompt already exists as token IDs.

Do not unnecessarily perform:

```text
token IDs -> decoded text -> tokenization -> token IDs
```

because this can:

- introduce tokenizer round-trip differences;
- waste compute;
- complicate tests;
- lose exact token identity in some tokenizers.

Prefer introducing or reusing a token-based cache encoder such as:

```python
encode_reference_tokens_to_cache(...)
```

Then refactor the text path so that conceptually:

```python
encode_reference_to_cache(text)
    -> tokenize(text)
    -> encode_reference_tokens_to_cache(tokens)
```

while implicit prompt overflow calls:

```python
encode_reference_tokens_to_cache(prefix_tokens)
```

Keep this refactor small.

Do not rewrite the entire reference materialization pipeline.

---

# 6. Reuse Existing Chunking

The implicit prompt head must use the existing reference chunking machinery.

Initially it should inherit the normal chunking configuration unless there is already a clean mechanism for source-specific configuration.

For example, the existing modes may include:

```text
fixed
markers
semantic
```

For V1, inheritance is sufficient.

Do not add a new prompt-specific chunker unless required by existing interfaces.

The design should leave room for future modes such as:

```text
message-boundary chunking
conversation-turn chunking
semantic dialogue chunking
```

but these are explicitly out of scope for this task.

---

# 7. Critical Issue: Explicit Reference Gist Limits Must Not Destroy Long Prompt History

Inspect the current interaction between:

```text
max_gists_per_reference
partition_reference(...)
gist_overflow_policy
```

The existing explicit-reference default may be small, e.g. 4 gists/chunks per reference.

This is unsuitable for a long synthetic prompt head.

Example:

```text
implicit prompt head = 28K tokens
fixed_chunk_tokens = 1024
expected chunks ~= 28+
max_gists_per_reference = 4
```

If the ordinary limit silently truncates the implicit reference to 4 chunks, most of the prompt history is lost and the feature is incorrect.

Implement the smallest clean override.

Preferred options, in order:

1. allow a per-reference max-gists/chunks override in metadata;
2. pass an explicit override argument to the chunking/materialization path;
3. otherwise add a narrowly scoped prompt-specific config path.

Example conceptual API:

```python
partition_reference(
    ...,
    max_gists_override=config.max_prompt_gists,
)
```

or:

```python
metadata["max_gists"] = config.max_prompt_gists
```

Required semantics:

- explicit references continue using `max_gists_per_reference`;
- implicit prompt history uses `max_prompt_gists`;
- when `max_prompt_gists is None`, do not arbitrarily discard most of the prompt prefix.

Do not confuse:

```text
number of chunks/gists available for routing
```

with:

```text
number of chunks materialized into attention
```

The first may be large.

The second should stay small via existing routing controls such as:

```text
top_k_chunks_per_reference
top_k_references
```

This distinction is central to PRA scalability.

---

# 8. Routing

For V1, the implicit prompt reference should participate in normal PRA routing.

That means its gists should be scored in the same general way as explicit-reference gists.

Do not add a special privileged routing policy yet.

In particular, avoid new knobs such as:

```text
always_include_prompt_head
top_k_prompt_chunks
prompt_reference_priority
```

unless the current architecture absolutely requires one for correctness.

These may be useful later, but first establish the simplest uniform behavior:

> old prompt history becomes one more PRA memory source.

This gives a clean experimental comparison and minimizes code complexity.

---

# 9. Generation Must Stop Blindly Dropping Old Prompt Tokens

Inspect generation code carefully.

The current implementation may do something equivalent to:

```python
input_ids = input_ids[:, -max_seq_len:]
```

before model evaluation.

When `prompt_overflow_mode == "implicit_reference"`, this old-prefix truncation must be replaced by the new prompt preparation logic.

Required generation behavior:

```text
long initial prompt
    -> prefix becomes #__head
    -> tail remains direct

generated continuation grows
    -> preserve correctness of the direct-window logic
```

For the minimal implementation, support long **initial prompts** correctly first.

Do not silently claim full streaming/unbounded generation support if generated tokens are not yet progressively migrated from the direct window into PRA memory.

If generated-history rollover is not implemented in this change:

- document that limitation clearly;
- add tests proving long initial prompts work;
- leave a TODO for streaming prompt-memory rollover.

However, if the existing generation structure makes rollover trivial and low-risk, it may be implemented.

Correctness is more important than expanding scope.

---

# 10. Forward Pass Behavior

The core model currently may assert something like:

```python
T <= max_seq_len
```

That assertion should remain valid for the final direct tensor entering ordinary self-attention.

Do not simply raise `max_seq_len` to allow huge prompts.

Instead:

```text
external long prompt
        |
        v
prepare_prompt_for_pra(...)
        |
        +-- #__head cache
        |
        +-- direct tail length <= configured direct budget
```

Then the transformer still sees a bounded local sequence.

This preserves the intended scalability.

---

# 11. Batch Support

The feature must work for batch size > 1.

Each batch row may have a different prompt length.

Example:

```text
row 0: 2K tokens  -> no implicit reference
row 1: 8K tokens  -> 4K implicit + 4K direct
row 2: 32K tokens -> 28K implicit + 4K direct
```

Do not assume all rows overflow equally.

Reuse the current row-local/batched memory-cache mechanics.

If the batched cache already namespaces memory by row, the same synthetic URI can likely be used for every row:

```text
pra://implicit/prompt/head
```

provided contents remain row-local.

Do not introduce globally unique random URIs merely to solve a namespace problem that the batch cache already solves.

Handle padding/masks correctly.

The implicit prefix must exclude padding tokens.

The retained direct tail must preserve each row's actual prompt ending.

Add tests for mixed-length batches.

---

# 12. Explicit References Must Continue to Work

A long prompt may also contain or be associated with explicit references.

Example:

```text
32K prompt
+ explicit document refs A, B, C
```

Expected memory candidates:

```text
#__head
A
B
C
```

The addition of implicit prompt memory must not disable, overwrite, or corrupt explicit-reference memory.

The same applies to recursive references if currently supported.

Test coexistence explicitly.

---

# 13. Cache Lifecycle

The implicit prompt reference is request/session-specific.

Do not accidentally persist it globally across unrelated prompts.

Ensure cache invalidation or cache identity semantics prevent this scenario:

```text
prompt A creates #__head
prompt B reuses stale #__head from prompt A
```

The simplest correct behavior is to build the implicit reference as part of the current request's memory/cache preparation.

If the existing cache is intentionally persistent, then the implicit prompt reference must be overwritten or namespaced by request/session identity.

Prefer minimal changes consistent with the current cache design.

---

# 14. Gist Computation

Do not invent a special gist mode for implicit prompt references.

Reuse the current gist machinery and all supported gist modes.

The implicit prompt chunks should therefore work with current or future modes such as:

```text
mean
last
GRU/RNN
k-means
SOM
prototype
hybrid
```

where supported by the codebase.

The new long-prompt path must be compatible with multiple gists per chunk/reference if that support exists or is being added.

Avoid assumptions that each chunk or reference always has exactly one gist.

Tensor shapes and loops should follow the current generalized gist representation.

---

# 15. Configuration Validation

Validate new parameters.

Suggested rules:

```text
max_prompt_direct_tokens:
    None or integer > 0

prompt_overflow_mode:
    one of:
        truncate
        implicit_reference
        error

max_prompt_gists:
    None or integer > 0
```

If:

```text
max_prompt_direct_tokens > max_seq_len
```

choose a clear policy.

Preferred initial policy:

```text
effective_direct_limit = min(max_prompt_direct_tokens, max_seq_len)
```

or reject it as invalid.

Follow whichever convention is most consistent with the rest of the config system.

Do not allow the model to receive a direct sequence longer than its supported positional/direct-attention limit.

---

# 16. Diagnostics and Metrics

Expose enough information to verify the mechanism experimentally.

At minimum, where existing metrics/logging infrastructure permits, report:

```text
prompt_total_tokens
prompt_direct_tokens
prompt_implicit_tokens
prompt_implicit_chunks
prompt_implicit_gists
prompt_implicit_chunks_selected
```

If routing diagnostics already exist, mark entries with source metadata so `#__head` can be distinguished from explicit refs.

Example debug output:

```text
source=prompt
ref=#__head
chunks=29
selected=[3, 17]
```

Do not build a large new telemetry framework.

Integrate with existing metrics/debug structures.

---

# 17. Tests

Add focused unit and integration tests.

## 17.1 Short Prompt, Feature Enabled

Input shorter than direct limit.

Expected:

```text
no implicit reference
prompt unchanged
results equivalent to existing path
```

---

## 17.2 Long Prompt, Truncate Mode

Expected behavior identical to old truncation semantics.

This protects backward compatibility.

---

## 17.3 Long Prompt, Error Mode

Expected:

```text
clear exception
```

---

## 17.4 Long Prompt, Implicit Reference Mode

Example:

```text
prompt = 10K tokens
direct limit = 4K
```

Verify:

```text
implicit prefix = first 6K tokens
direct tail = last 4K tokens
no token loss
no duplication at split boundary
```

---

## 17.5 Chunk Count Is Not Accidentally Capped by Explicit Reference Default

Construct a prefix requiring more chunks than `max_gists_per_reference`.

Example:

```text
max_gists_per_reference = 4
prefix requires 10 chunks
max_prompt_gists = None
```

Verify more than 4 prompt chunks/gists remain available.

This is a critical regression test.

---

## 17.6 Explicit Prompt Gist Limit

Example:

```text
prefix requires 10 chunks
max_prompt_gists = 6
```

Verify the configured prompt-specific policy is respected.

---

## 17.7 Routing Can Retrieve an Early Prompt Fact

Create a deterministic synthetic test:

```text
early prefix:
    "The secret code is ZEBRA-731."

many irrelevant tokens

recent tail:
    "What is the secret code?"
```

The direct tail must not contain the answer.

Verify, as far as deterministic test infrastructure allows, that:

- the relevant head chunk exists;
- it is routable;
- the PRA cache can select/materialize it under controlled scoring or mocked query conditions.

Do not rely on stochastic model quality for a unit test.

Mock or construct scores if necessary.

---

## 17.8 Mixed Batch Lengths

Batch with:

```text
short
medium overflow
large overflow
```

Verify each row gets the correct local prefix/tail split and row-local cache.

---

## 17.9 Long Prompt Plus Explicit References

Verify both implicit and explicit references exist and are independently routable.

---

## 17.10 Token Round-Trip

If token-based cache encoding is implemented, verify exact token identity.

If any decode/re-encode path remains, add a tokenizer-sensitive test or document why equality is guaranteed.

---

## 17.11 Generation

Verify a long initial prompt no longer loses its old prefix before PRA memory is constructed.

The generated model input should satisfy the local max-sequence constraint.

---

# 18. Benchmark / Smoke Evaluation

Add one small smoke experiment if the repository has an evaluation framework.

Compare:

```text
A. truncate
B. implicit_reference
```

Use prompts where the answer depends on information located outside the direct window.

Suggested synthetic evaluation:

```text
prefix:
    random key/value facts

filler:
    enough tokens to push facts outside direct window

tail:
    query one earlier key
```

Metrics:

```text
retrieval recall@k
selected correct chunk rate
LM answer accuracy if practical
materialized KV tokens
number of routable prompt chunks
```

The primary V1 correctness metric is retrieval of the correct displaced chunk, not end-to-end language-model accuracy.

---

# 19. Performance Expectations

The feature must preserve the PRA scalability objective.

Do not materialize the entire prefix into attention.

For a long prompt:

```text
32K total
4K direct
28K implicit
~28 prompt chunks
```

the intended flow is:

```text
~28 cheap gists searchable
        |
        v
top-k chunk routing
        |
        v
only a small number of K/V chunks materialized
```

Memory/compute should scale primarily with:

```text
direct window
+ gist index
+ selected chunk K/V
```

not:

```text
full prompt K/V in every layer
```

Do not implement a fallback that defeats this property.

---

# 20. Code Organization

Keep the implementation modular.

Preferred responsibilities:

```text
config.py
    configuration + validation

prompt preparation / model helper
    split prompt
    create implicit reference metadata
    dispatch cache construction

chunking.py
    only small override support if needed

model/cache encoding
    token-based reference encoder if added

memory.py
    ideally no semantic special-casing
    only generic metadata/limits if needed

generation
    invoke prompt-overflow preparation
```

Do not scatter `if source == "prompt"` checks throughout the attention stack.

One or two provenance checks at preparation/config boundaries are acceptable.

---

# 21. Documentation

Update README/config docs with a concise section:

## Long prompts

Example:

```yaml
max_seq_len: 4096
max_prompt_direct_tokens: 4096
prompt_overflow_mode: implicit_reference
max_prompt_gists: null
```

Explain:

```text
For prompts longer than 4096 tokens, the latest 4096 tokens remain in
direct causal self-attention. Earlier tokens are converted to an implicit
PRA reference (#__head), chunked and routed like other references.
```

Also document the current limitation if generated-history streaming rollover is not included.

---

# 22. Non-Goals

Do not add the following in this task unless required for correctness:

```text
hierarchical prompt memory
multi-level #__head trees
automatic recursive compression
special prompt-head routing priority
message-aware chunking
continuous streaming memory migration
new gist algorithms
new attention kernels
global ANN index
disk-backed prompt memory
changes to explicit reference syntax
```

These are possible follow-up extensions.

This task should establish the smallest robust architectural bridge from long prompts to existing PRA memory.

---

# 23. Future-Compatible Design

Although V1 has one implicit reference:

```text
#__head
```

avoid assumptions that there can only ever be one implicit prompt-memory segment.

Future designs may evolve toward:

```text
#__head_0
#__head_1
#__head_2
```

or:

```text
recent direct context
    |
older PRA memory
    |
compressed older PRA memory
    |
persistent/archive memory
```

Likewise, eventual streaming generation may repeatedly migrate expired direct-window tokens into PRA memory.

The V1 abstractions should not make those extensions unnecessarily difficult.

Do not implement them now.

---

# 24. Acceptance Criteria

The task is complete when all of the following are true:

1. A prompt larger than the direct model context can be accepted in `implicit_reference` mode.

2. Its old prefix is preserved as synthetic PRA memory rather than discarded.

3. Its recent tail remains normal direct causal context.

4. The synthetic prefix reuses existing:
   - reference chunking;
   - gist computation;
   - layer-specific cache encoding;
   - routing;
   - K/V materialization.

5. No changes are required to the fundamental PRA attention algorithm.

6. Large implicit prompt references are not accidentally truncated by the ordinary small explicit-reference gist limit.

7. Batch rows with different prompt lengths work correctly.

8. Explicit references continue working alongside `#__head`.

9. Long initial generation prompts no longer blindly lose their old prefix.

10. Existing behavior remains unchanged when the new feature is disabled.

11. Tests cover split correctness, cache correctness, routing availability, batching, explicit-ref coexistence, and the gist-limit regression.

12. Documentation explains the feature and its current limitations.

---

# 25. Implementation Philosophy

Prefer:

```text
small refactor
+ synthetic reference
+ reuse existing machinery
```

over:

```text
new long-context subsystem
```

The conceptual result should be:

> PRAttention has a bounded high-resolution direct context window. When a prompt exceeds that window, displaced history becomes progressively retrievable PRA memory.

This is both the minimal implementation and the desired architectural direction.
