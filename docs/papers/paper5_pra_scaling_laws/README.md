# Paper 5: PRA Scaling Laws

This directory contains the controlled first scaling study of Progressive
Retrieval Attention (PRA). The paper separates logical reference memory,
selected native-K/V state, routing-index residency, search work, and model
quality. The current result is a five-seed routing/systems pilot; the matched
Gemma model ladder and end-to-end serving measurements remain open gates.

Regenerate the measured artifacts and manuscript with:

```powershell
python -m experiments.paper5_scaling_laws.run_scaling_study
python -m experiments.paper5_scaling_laws.summarize_scaling_study
Set-Location docs/papers/paper5_pra_scaling_laws
latexmk -pdf -interaction=nonstopmode paper.tex
```

The claim boundary is recorded in
`../shared/results/paper5_scaling/scaling_claim_audit.md`.
