# Candidate and Budget Controls

## Candidate breadth

L1 and L2 evaluate 5, 10, 20, and 50 candidate documents. Candidate IDs and order are stored once in receipt form and reused by every condition. L1 guarantees gold presence; L2 preserves the real BM25 output even when evidence is missing.

## Physical-token budget

The natural grid uses 2K, 4K, 8K, and 16K tokens. Both selectors pack whole chunks under the same cutoff. PRA may keep all candidate documents logically addressable, but the reported physical count includes only selected/materialized chunks.

## Distractors

Increasing candidate breadth introduces 3, 8, 18, or 48 non-gold documents around the typical two-document query. We report false selected-document fraction alongside evidence coverage. Higher answer availability with higher false selection is not interpreted as cleaner routing.

## Selector and transport controls

The L1/L2 selection grid compares selector outputs. A model-backed native run additionally compares visible selected text with detached native K/V. For representation-only E0/E2 claims, the selected chunk IDs must be frozen once and fed to both transports; independent selection is forbidden.
