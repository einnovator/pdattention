# AGENTS.md — Paper 1

## Goal
Develop the standalone PyTorch PRA paper.

## Main Contribution
A fully controlled research prototype where a tiny decoder-only transformer uses PRA layers, reference handles, recursive anchors, and layer-specific memory caches.

## Must Cover
- Exact model definition.
- PRA attention equations.
- Reference table and resolver.
- Per-layer KV cache construction.
- Dataset design.
- PyTorch Dataset/DataLoader pipeline.
- Training system.
- Evaluation metrics.
- Ablations:
  - no PRA
  - summary only
  - one-level retrieval
  - recursive retrieval
  - cross-attention vs KV injection
  - different PRA layers

## Experimental Status
Use mock tables only as placeholders until real results exist.

## Target
Implementation-backed architecture paper.
