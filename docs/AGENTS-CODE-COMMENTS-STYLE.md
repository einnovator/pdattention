# AGENTS-CODE-COMMENTS-STYLE.md
# Code Comments and Documentation Style Guide

Version: 1.0

## Purpose

This document defines how code comments and docstrings should be written across the repository.

The goal is to make the code understandable to engineers and researchers who did not participate in its design. Comments should help readers understand:

- what a component represents,
- why it exists,
- how it fits into the larger system,
- how data moves through it,
- which operating modes it supports,
- what assumptions it makes,
- what shapes important tensors have.

Comments should be concise, accurate, pedagogical, and easy to maintain.

The objective is not to comment every line. The objective is to explain the parts of the code that are difficult to infer correctly from syntax alone.

---

# Core Principle

Comment intent, structure, and constraints rather than restating syntax.

Bad:

```python
# Increment the index.
index += 1
```

Better:

```python
# Advance to the next retrieval stage after the current reference is resolved.
index += 1
```

The reader can already see what the code does mechanically. The comment should explain why the operation matters.

---

# What Must Be Documented

Document all non-trivial:

- modules,
- classes,
- configuration objects,
- fields and attributes,
- public functions,
- important private methods,
- long-function blocks,
- tensor transformations,
- execution modes,
- assumptions and invariants,
- side effects,
- performance-sensitive choices,
- numerical-stability logic.

Simple and self-explanatory code does not need commentary.

---

# Module Documentation

Important modules should begin with a short docstring explaining:

- the module's responsibility,
- its main abstractions,
- how it connects to neighboring modules,
- the main modes it supports.

Example:

```python
"""
Reference materialization for PRAttention.

This module converts resolved document references into layer-specific key/value
representations. The resolver obtains raw content; this module determines how
that content becomes attention memory.

Supported materialization modes include mean pooling, last-state selection,
and recurrent summarization.
"""
```

The module docstring should orient the reader before they inspect individual classes.

---

# Class Documentation

Every non-trivial class should have a docstring explaining:

- what the class represents,
- its role in the overall architecture,
- the state it owns,
- its relationship to other components,
- its important operating modes,
- lifecycle assumptions when relevant.

Example:

```python
class ReferenceMaterializer(nn.Module):
    """
    Converts resolved reference content into layer-specific attention memory.

    The materializer sits between the external resolver and the attention layer.
    It receives token representations for a referenced chunk and reduces them to
    one or more key/value gist vectors.

    The reduction strategy is selected by `mode`:

    - `mean`: average valid token states;
    - `last`: use the final valid token state;
    - `gru`: summarize the sequence with a recurrent encoder.

    The resulting vectors are later merged with local attention memory by
    `PRAttentionLayer`.
    """
```

Avoid empty descriptions.

Bad:

```python
class CacheManager:
    """Manages the cache."""
```

Better:

```python
class CacheManager:
    """
    Stores layer-specific key/value tensors for resolved references.

    Entries are indexed by reference identity and layer number. Reusing them
    avoids recomputing the same reference representation across attention calls.
    """
```

---

# Field and Attribute Documentation

Document fields when their meaning is not obvious from the name.

This is especially important for:

- configuration objects,
- dimensions,
- masks,
- caches,
- counters,
- thresholds,
- mode selectors,
- routing state,
- persistent tensors,
- units and ranges.

Explain what the field represents and how it affects behavior.

Bad:

```python
self.config = config
```

Better:

```python
# Experiment configuration controlling retrieval, materialization, batching,
# and attention integration.
self.config = config
```

Bad:

```python
self.memory = {}
```

Better:

```python
# Layer-local cache mapping `(reference_id, layer_index)` to materialized
# key/value tensors.
self.memory = {}
```

For configuration classes, document the behavioral meaning of each important field.

Example:

```python
@dataclass
class MaterializationConfig:
    """Controls how referenced content becomes attention memory."""

    mode: str
    """Reduction strategy: `mean`, `last`, or `gru`."""

    max_gists: int
    """Maximum number of key/value gist vectors produced per reference."""

    use_summary: bool
    """Use a metadata-provided summary instead of raw content when available."""
```

Comments should clarify valid values, defaults, units, and consequences.

---

# Function and Method Documentation

Every public function and every non-trivial method should explain:

- what it does,
- why it exists,
- its role in the larger workflow,
- its inputs,
- its output,
- side effects,
- relevant tensor shapes,
- mode-dependent behavior,
- important constraints.

Example:

```python
def materialize_reference(
    hidden_states: Tensor,
    attention_mask: Tensor,
    mode: str,
) -> Tensor:
    """
    Reduce token-level reference states to one attention-memory vector.

    This function runs after the referenced text has been encoded for the
    current Transformer layer. It converts a variable-length sequence into a
    fixed-size representation that can be appended to key/value memory.

    Args:
        hidden_states:
            Encoded reference tokens with shape `[batch, ref_len, hidden_dim]`.
        attention_mask:
            Boolean mask with shape `[batch, ref_len]`. `True` marks valid tokens.
        mode:
            Reduction strategy. Supported values are `mean` and `last`.

    Returns:
        One vector per batch item with shape `[batch, hidden_dim]`.
    """
```

Private helpers may use shorter docstrings, but their role should still be clear when it is not obvious.

---

# Explain Architectural Role

Important methods should describe how they relate to the rest of the system.

Example:

```python
def resolve_and_materialize(...):
    """
    Resolve external references and convert them into attention memory.

    The resolver locates and loads referenced content. This method encodes that
    content and delegates representation reduction to `ReferenceMaterializer`.
    The resulting tensors are consumed by the attention layer during key/value
    assembly.
    """
```

Do not document coordinating methods as isolated utilities.

---

# Different Operating Modes

When behavior changes according to configuration, explain the alternatives explicitly.

Example:

```python
# Materialization modes trade fidelity for cost:
# - `mean`: cheap and order-insensitive;
# - `last`: cheap and sensitive to final-state encoding;
# - `gru`: order-sensitive but adds parameters and sequential computation.
```

For larger mode switches, describe the modes before the branch.

Example:

```python
# Reference selection supports three execution modes:
#
# 1. `all`: materialize every resolved reference;
# 2. `topk_refs`: retain only the highest-scoring references;
# 3. `topk_gists`: materialize all references, then prune gist vectors globally.
#
# Selection happens before padding so rejected references do not consume memory.
```

The reader should understand the design before reading the individual branches.

---

# Tensor Shape Documentation

Tensor shapes must be documented whenever they are not immediately obvious.

Use one consistent notation throughout the repository.

Preferred format:

```python
# queries: [batch, heads, query_len, head_dim]
# keys:    [batch, heads, memory_len, head_dim]
# values:  [batch, heads, memory_len, head_dim]
```

Define symbolic dimensions when first introduced.

Example:

```python
# B = batch size
# H = number of attention heads
# T = local token count
# R = retrieved gist count
# D = per-head dimension
#
# q: [B, H, T, D]
# k: [B, H, T + R, D]
# v: [B, H, T + R, D]
```

Document shape changes near the operation that causes them.

Example:

```python
# Split the model dimension into heads:
# [B, T, hidden_dim] -> [B, H, T, head_dim].
queries = self.q_proj(hidden_states).view(B, T, H, D).transpose(1, 2)
```

For concatenation, explain the semantic meaning of each region.

Example:

```python
# Append retrieved memory after local token memory.
# local_k: [B, H, T, D]
# ref_k:   [B, H, R, D]
# result:  [B, H, T + R, D]
keys = torch.cat([local_k, ref_k], dim=2)
```

For masks, document both shape and convention.

Example:

```python
# attention_mask: [B, 1, T, T + R]
# `True` means the attention edge is allowed.
```

Never assume that `0`, `1`, `True`, or `-inf` mask conventions are self-evident.

---

# Long Functions

Long functions should be divided into logical phases using short block comments.

Example:

```python
def forward(...):
    # 1. Encode the current token sequence.
    ...

    # 2. Resolve references visible at this layer.
    ...

    # 3. Materialize selected references into key/value gist tensors.
    ...

    # 4. Merge local and retrieved memory.
    ...

    # 5. Apply attention and restore the original tensor layout.
    ...
```

Block comments should describe conceptual steps, not individual lines.

If a function needs many block comments, consider splitting it into smaller methods. Comments do not replace good decomposition.

---

# Non-Obvious Control Flow

Explain branches, loops, recursion, and early returns when their purpose is not obvious.

Example:

```python
# Preserve the standard Transformer path when the batch contains no references.
# Returning early also avoids allocating empty padded tensors.
if not references:
    return local_keys, local_values
```

Example:

```python
# Resolve nested references depth-first so child content is available before
# the parent chunk is summarized.
for child in reference.children:
    resolve(child)
```

---

# Invariants and Assumptions

Document conditions that must remain true.

Example:

```python
# Invariant: `reference_ids`, `reference_scores`, and `reference_lengths`
# preserve the same ordering throughout selection and batching.
```

Example:

```python
# Assumption: padding appears only after valid tokens. The `last` materializer
# relies on this to find the final valid state efficiently.
```

Example:

```python
# `head_dim` must be even because RoPE rotates coordinate pairs.
```

Invariants are especially important when future refactoring could silently break behavior.

---

# Side Effects

Methods that mutate caches, persistent state, metrics, or external resources should say so.

Example:

```python
def update_cache(...):
    """
    Add or replace a materialized reference entry.

    This method mutates the layer-local cache. Existing entries with the same
    reference and layer identifiers are replaced.
    """
```

Do not hide state mutation inside apparently pure helper methods.

---

# Units, Ranges, and Semantics

Document units and expected ranges where relevant.

Example:

```python
timeout_seconds: float
"""Maximum resolver wait time in seconds."""

dropout_probability: float
"""Dropout probability in the interval `[0, 1)`."""
```

For scores, explain ordering.

Example:

```python
# Retrieval score in `[0, 1]`; larger values indicate a stronger match.
```

---

# Performance-Sensitive Choices

Explain optimizations whose purpose may otherwise look accidental.

Example:

```python
# Group references by similar lengths before padding. This reduces wasted memory
# compared with placing all references in one global rectangle.
```

Example:

```python
# Use `reshape` instead of `view` because the preceding transpose may produce
# a non-contiguous tensor.
```

Avoid vague comments such as `# optimization`. State what cost is reduced and why the implementation is safe.

---

# Numerical Stability

Document numerical safeguards.

Example:

```python
# Subtract the row maximum before exponentiation to prevent overflow while
# preserving the softmax result.
scores = scores - scores.amax(dim=-1, keepdim=True)
```

Example:

```python
# Avoid division by zero for fully masked references.
denominator = valid_counts.clamp_min(1)
```

---

# Research Code

Research mechanisms often require more explanation than standard application code.

Document:

- the hypothesis behind the mechanism,
- how the implementation corresponds to the paper or design,
- current simplifications,
- which parts are provisional,
- which configuration enables the experiment,
- what baseline behavior is preserved when disabled.

Example:

```python
# Experimental progressive-materialization path.
#
# The hypothesis is that later layers can use fewer, more abstract memory
# vectors without degrading prediction quality. When disabled, execution falls
# back to full materialization and should match the baseline path.
```

Do not present hypotheses as established facts.

Prefer language such as:

- "The hypothesis is..."
- "This implementation approximates..."
- "This experiment tests whether..."
- "The current prototype assumes..."

---

# TODO and FIXME Comments

Use TODO only for concrete future work.

Good:

```python
# TODO(js): Replace global top-k selection with per-head selection.
```

Use FIXME for known incorrect or fragile behavior.

Example:

```python
# FIXME: This assumes all batch items contain the same number of references.
```

Avoid vague comments.

Bad:

```python
# TODO: improve this
```

A good TODO states what should change and, when useful, why.

---

# Configuration Documentation

Configuration fields should explain both meaning and behavioral effect.

Example:

```python
bucket_size: int = 1
"""
Reference batching strategy.

- `0`: process each sample independently;
- `1`: pad all references into one batch rectangle;
- `>=2`: group references into length buckets before padding.
"""
```

Defaults should be chosen and documented intentionally.

---

# Public APIs

Public APIs require more complete documentation than private helpers.

A public API should be usable correctly without reading its implementation.

Document:

- accepted types,
- shape contracts,
- required ordering,
- optional behavior,
- side effects,
- exceptions,
- return semantics.

Private helpers may assume local context and can be shorter.

---

# Comment Placement

Keep comments close to the code they explain.

Use:

- module docstrings for module responsibility,
- class docstrings for component role,
- method docstrings for contracts,
- inline comments for local reasoning,
- architecture documents for system-wide design.

Do not place essential local details only in a distant README.

At the same time, avoid repeating the same long architectural explanation in many files.

---

# Accuracy and Maintenance

Incorrect comments are worse than missing comments.

Whenever code behavior changes, update related comments and docstrings in the same change.

Treat stale comments as defects.

Remove comments that describe:

- deleted modes,
- old tensor shapes,
- obsolete assumptions,
- previous implementation behavior.

---

# Concision

Comments should be on point.

A good inline comment usually explains one idea in one to three lines.

Longer explanations belong in:

- docstrings,
- architecture documents,
- design notes,
- papers.

Inline comments should make the implementation easier to scan.

---

# Pedagogical but Clean

Prefer:

```python
# Merge retrieved memory after local memory so original token positions remain
# unchanged and reference positions occupy one contiguous suffix.
```

Avoid:

```python
# Here we are now going to merge the retrieved memory with the local memory in
# such a way that the local memory comes first and the retrieved memory comes
# after it, which is useful because...
```

Capture the principle directly.

---

# Avoid Redundant Comments

Do not comment obvious imports, assignments, trivial accessors, or standard syntax.

Bad:

```python
# Import torch.
import torch
```

Bad:

```python
# Return the result.
return result
```

Better:

```python
# Return unnormalized routing logits; the caller applies temperature scaling.
return logits
```

---

# Avoid Ambiguous Pronouns

Bad:

```python
# This is used later.
```

Better:

```python
# The attention layer reuses this mask after reference keys are appended.
```

Name the relevant object directly.

---

# Consistent Terminology

Use the same term for the same concept throughout the codebase.

Do not alternate casually between:

- summary,
- gist,
- compressed state,
- memory vector,

unless these terms have explicitly different meanings.

Align code comments with the terminology used in configuration, architecture documentation, and papers.

---

# Example: Well-Documented Method

```python
def merge_reference_memory(
    local_keys: Tensor,
    local_values: Tensor,
    reference_keys: Tensor,
    reference_values: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Append materialized reference memory to local attention memory.

    This preserves local token positions and places all retrieved gist vectors
    in a contiguous suffix. The caller must extend the attention mask to cover
    the additional memory positions.

    Args:
        local_keys:
            Local key tensor with shape `[B, H, T, D]`.
        local_values:
            Local value tensor with shape `[B, H, T, D]`.
        reference_keys:
            Retrieved key tensor with shape `[B, H, R, D]`.
        reference_values:
            Retrieved value tensor with shape `[B, H, R, D]`.

    Returns:
        `(keys, values)`, each with shape `[B, H, T + R, D]`.
    """

    # Append rather than interleave reference positions so indexing for the
    # original token sequence remains unchanged.
    keys = torch.cat([local_keys, reference_keys], dim=2)
    values = torch.cat([local_values, reference_values], dim=2)

    return keys, values
```

---

# Review Checklist

Before considering code complete, verify:

## Modules and Architecture

- Does each important module explain its responsibility?
- Is its relationship to neighboring modules clear?
- Are architectural boundaries documented?

## Classes and Fields

- Does each non-trivial class explain what it represents?
- Are important fields documented?
- Do configuration fields explain their behavioral effect?
- Are caches, masks, counters, and persistent states clear?

## Functions and Methods

- Does each public function explain its role?
- Are inputs, outputs, side effects, and constraints documented?
- Do long functions contain meaningful phase comments?
- Are non-obvious branches and early returns explained?

## Tensor Shapes

- Are important tensor shapes stated?
- Are symbolic dimensions defined?
- Are reshapes, transposes, concatenations, and broadcasts explained?
- Is the mask convention explicit?

## Modes and Assumptions

- Are alternative execution modes documented?
- Are invariants and assumptions stated?
- Are training and inference differences clear?
- Are baseline and experimental paths distinguishable?

## Style

- Do comments explain intent rather than syntax?
- Are comments concise and pedagogical?
- Is terminology consistent?
- Are stale and redundant comments removed?
- Could a new engineer understand the code without reverse-engineering it?

---

# Final Standard

The code should remain readable to someone who did not participate in its design.

A reader should be able to understand the architecture, data flow, operating modes, tensor shapes, and important assumptions by reading the code and its comments together.

Comments should not overwhelm the implementation.

They should illuminate it.
