# AGENTS.md — Extend PRAttention Survey with Long-Context Memory and Compression Techniques

## Goal

Extend the PRAttention paper into a comprehensive survey and positioning paper covering the evolution of long-context transformers, memory architectures, KV-cache compression, sparse attention, and hierarchical memory systems.

The objective is **not** simply to enumerate papers, but to explain:

- why each method exists,
- what problem it solves,
- how it works internally,
- computational complexity,
- strengths and weaknesses,
- implementation difficulty,
- biological/cognitive interpretation when appropriate,
- and how it relates to PRAttention.

The resulting document should become one of the strongest background/reference sections for the PRAttention research program.

---

# General Style

Write for researchers.

Assume readers are familiar with machine learning but not necessarily long-context transformers.

Avoid lists without explanation.

Every mechanism should first be motivated.

Then explained mathematically.

Then illustrated intuitively.

Then compared against alternatives.

Use diagrams whenever useful.

Prefer long explanatory paragraphs over shallow summaries.

---

# Main Structure

## 1. Introduction

Explain why context length has become one of the central research problems in modern LLMs.

Discuss:

- quadratic attention
- memory limitations
- retrieval
- compression
- recurrence
- external memory
- hierarchical memory
- KV cache explosion
- latency

Explain why there is no universally accepted solution.

---

# 2. Taxonomy of Long Context Methods

Build a taxonomy.

Example:

Long-context methods

- Sparse attention
- Local attention
- Global tokens
- Memory tokens
- External memory
- KV compression
- Token pruning
- Hierarchical attention
- Recurrent state
- Learned memory
- Retrieval
- Progressive retrieval
- Prototype memories
- Persistent state
- Dynamic routing

Illustrate the taxonomy.

---

# 3. Detailed Survey

Every technique should receive several pages.

Each section should include:

## Motivation

Why was it proposed?

What bottleneck existed?

---

## Core idea

Explain intuitively.

---

## Mathematical formulation

Include equations.

---

## Computational complexity

Training

Inference

KV cache

Memory

Latency

Bandwidth

---

## Advantages

---

## Limitations

---

## Biological interpretation (if meaningful)

---

## Implementation notes

PyTorch pseudocode.

---

## Relationship with PRAttention

Explain:

Compatible?

Complementary?

Competing?

Can they be combined?

---

# Methods to Cover

## Sparse Attention

### Longformer

Explain

- sliding window
- dilated attention
- global tokens
- complexity
- implementation

Compare against vanilla attention.

---

### BigBird

Explain

- random attention
- block sparse attention
- global attention

Include theoretical guarantees.

Discuss graph connectivity.

---

### Token Sparse Attention

Survey modern token sparsity methods.

Include

dynamic sparsity

learned sparsity

routing

importance prediction

hardware implications

---

# Memory Compression

## Compressive Transformer

Explain

memory hierarchy

compressed memory

compression operators

gradient flow

long horizon

---

## Infini-Attention

Explain

compressed memory

associative memory

constant memory growth

streaming inference

---

## Titans

Provide a detailed explanation of the Titans architecture.

Cover:

- neural memory
- persistent memory
- surprise-based updates
- fast vs slow memory
- inference process
- training
- scalability
- comparison with recurrent transformers

Discuss how Titans differs from traditional KV cache.

---

# KV Cache Compression

Provide a dedicated chapter.

Explain why KV cache dominates inference.

Discuss:

memory bandwidth

GPU limitations

latency

multi-user serving

---

## ClusterKV

Explain

clustering

representative KV

hierarchical clustering

complexity

quality

limitations

---

## SnapKV

Explain

head-specific importance

selection

compression

streaming inference

---

## H2O

Explain

heavy hitter tokens

importance

cache eviction

dynamic retention

comparison with LRU

---

## Other Recent KV Compression

Survey recent work including, when relevant:

- PyramidKV
- FastGen
- Scissorhands
- StreamingLLM
- Keyformer
- MiniCache
- LeanKV
- DynamicKV
- AdaKV
- MiKV
- Quest
- RazorAttention
- CAKE
- CacheBlend
- and other significant recent methods.

For each:

motivation

idea

complexity

pros

cons

---

# Landmark Methods

Explain Landmark Attention.

Discuss:

landmark tokens

hierarchical retrieval

memory anchors

comparison with PRAttention references

---

# Retrieval Methods

Discuss

RETRO

REALM

Memorizing Transformer

Atlas

RAG

GraphRAG

Self-RAG

Compare against explicit references.

---

# Recurrent Memory

Discuss

Transformer-XL

Compressive Transformer

Block Recurrent Transformer

TransformerFAM

Titans

Persistent Memory

Explain evolution.

---

# Persistent State Models

Survey

state-space inspired approaches

persistent latent variables

cross-attention memories

learned slots

memory tokens

Discuss similarities with PRAttention persistent memory.

---

# Memory Compression Landscape

Create a comparison table including:

Method

Year

Memory Type

Compression

External Memory

Streaming

Training Changes

Inference Changes

Complexity

Context Length

GPU Memory

Implementation Difficulty

Open Source

---

# Cognitive Interpretation

For every architecture explain possible mapping to cognition.

Examples:

Longformer

→ local visual attention

BigBird

→ random associative recall

ClusterKV

→ concept clustering

SnapKV

→ selective working memory

Infini

→ semantic memory

Titans

→ long-term memory

Compressive Transformer

→ abstraction

Landmark Attention

→ episodic anchors

PRAttention

→ explicit symbolic recall

Discuss where analogies hold and where they fail.

---

# Relationship to PRAttention

The paper should repeatedly position PRAttention.

Questions to answer:

What problems does PRAttention solve that existing methods do not?

What existing methods solve problems PRAttention does not?

Can they be combined?

Example combinations:

PRAttention + ClusterKV

PRAttention + Titans

PRAttention + Infini

PRAttention + Landmark

PRAttention + H2O

PRAttention + Compressive Transformer

Discuss expected advantages.

---

# Unified Design Space

Create a conceptual design map.

Axes might include:

implicit ↔ explicit memory

compressed ↔ exact memory

learned ↔ symbolic

internal ↔ external

dense ↔ sparse

static ↔ adaptive

persistent ↔ transient

Place every surveyed method on these axes.

Include PRAttention.

---

# Future Research

Discuss open questions such as:

- lifelong memory
- continual learning
- catastrophic forgetting
- semantic compression
- prototype memory
- hierarchical memory
- learned indexing
- explicit references
- memory routing
- cognitive architectures

---

# References

Target 150–250 references.

Prioritize:

NeurIPS

ICML

ICLR

ACL

EMNLP

Nature

Science

arXiv

Official repositories

Include links whenever possible.

---

# Deliverables

Produce:

- `docs/prattention_long_context_survey.md`
- `papers/prattention_long_context_survey.tex`
- compiled PDF
- comparison tables
- SVG figures
- Mermaid diagrams
- bibliography (.bib)

---

# Quality Bar

The resulting survey should read like a modern graduate textbook chapter rather than a literature review.

It should be sufficiently complete that a new PhD student could understand the entire long-context transformer landscape before reading the PRAttention proposal.

Whenever new memory architectures appear during development, incorporate them into the taxonomy and comparison tables while preserving the overall structure.