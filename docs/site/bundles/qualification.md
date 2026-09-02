# Bundle Qualification Evidence

The model card is generated from `bundle.yaml` and checksummed files under
`qualification/`. Headline claims are accepted only when baseline and PRA use
the same selected evidence and the exact model ID, immutable revision,
quantization, engine/version, profile, and execution mode match.

Four metric classes remain separate:

1. **End-task quality:** QA F1, exact accuracy, or coding-task success.
2. **Semantic equivalence:** exact output, logit, or first-token parity.
3. **Routing diagnostics:** evidence recall, Recall@budget, MRR, and AUC.
4. **Serving economics:** visible tokens, TTFT, ITL, throughput, memory, and transfers.

Routing recall is useful research evidence, but it is never presented as
headline application quality. When paired evidence does not exist, the card
says what remains to be measured rather than emitting a large table of empty
`NOT_MEASURED` cells.

## Release gate

Publication validates one recommended `QUALIFIED` profile, profile consistency,
recognized evidence tiers, baseline/PRA pairing, and exact evidence identity.
Revision or quantization mismatches fail the build. QUALITY and ECONOMY remain
`CALIBRATION_PENDING` where reduced consumer layers did not pass held-out
quality; BALANCED keeps all eligible layers.

Requalify on your own workload:

```bash
pra evaluate BASE_MODEL -e ENGINE -a OWNER/BUNDLE \
  -D qasper --include-native-memory -o .pra/runs/qualification
pra recommend .pra/runs/qualification
pra report .pra/runs/qualification --format html
```

See the [qualification matrix](qualification-matrix.md) for current evidence
tiers and exact artifacts.
