# Paper 4: PRA-aware training

## Central question

Can transformer plasticity make sparse external native K/V useful as a learned
computational primitive, rather than only an inference-time retrofit?

## Claim boundary

Gate 0 fixes oracle memory identities and trains only memory production and
consumption. Do not describe it as learned routing. Do not escalate the paper's
claims beyond the completed tier recorded in `paper4_findings.json`.

## Reproducibility

All reported values must be generated from
`experiments/paper4_training/summarize_tier0.py`. Private checkpoints are not
paper artifacts.
