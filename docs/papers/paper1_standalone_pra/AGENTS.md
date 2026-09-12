# AGENTS.md — Paper 1

## Goal

Develop and maintain the standalone PyTorch PRA paper.

## Main contribution

A controlled implementation of addressable logical memory in which contiguous source
encoding produces layer-native K/V, compact gists route to smaller views, and only selected
full-detail K/V enters an explicitly bounded attention operation.

## Must cover

- Separate contextual encoding blocks from routing chunks.
- Keep logical source size, accessible K/V, selected physical K/V, and native-operation
  length distinct.
- Separate oracle transport, learned routing, bounded execution, device residency, and
  pretrained calibration claims.
- Report the exact independent unit, seed pairing, model/training configuration, and canonical
  artifact for every result family.
- Treat RCB as a controlled answer-token loss ratio, never as general QA correctness.
- Describe pretrained Qwen3 evidence as annotated-selection transport and payload accounting,
  not learned routing or production memory savings.

## Experimental status

The main tables are backed by frozen JSON artifacts. Do not introduce mock results. Run
`python scripts/audit_paper1_headlines.py` after changing any headline number, then rebuild
and visually inspect `paper.pdf`.

## Target

Claim-focused empirical architecture and memory-systems paper. New experiments are not
required for editorial changes; unresolved causal or serving claims must remain explicit
limitations.
