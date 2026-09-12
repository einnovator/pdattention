# Notes

## Current focus

Paper 1 owns the bounded standalone mechanism: contextual source encoding, fine routing
views over layer-native K/V, sparse materialization under an explicit operation limit, and
separate accounting of logical size, active K/V, residency, and routing cost.

The causal evidence comes from controlled PyTorch models. The Qwen3 MLX campaign is a
separate pretrained transport/payload calibration with annotated selection; it is not a
learned-router evaluation or an independent replication of Paper 1.5.

## Completed publication checks

- Added the complete principal training configuration and an experiment-level evidence map.
- Corrected the fragmentation provenance to the dedicated sensitivity artifact.
- Added a 49-check frozen-artifact numerical audit and SHA-256 receipt.
- Made the controlled-versus-pretrained evidence boundary explicit.
- Rendered and visually inspected the final PDF.

## Remaining scientific work

- Export per-question target-hit/miss traces and rebuild downstream state after changing the
  answer-bearing source; until then, keep the result framed as answer-code transport.
- Compare selected native K/V with the same selected evidence re-encoded as text under matched
  output length and complete cold/warm cost accounting.
- Replicate the pretrained calibration across model families and more unique 32K questions.
