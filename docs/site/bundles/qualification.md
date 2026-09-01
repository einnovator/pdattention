# Bundle Qualification Evidence

A model card is a compact view of evidence stored in `bundle.yaml` and the
bundle's `qualification/` payload. Metrics are generated from qualification
manifests and the Paper 4.5 product matrix rather than maintained independently.

Each row should identify:

- engine and version;
- exact model and tokenizer revision;
- hardware and precision;
- workload, dataset, cohort, and seed count;
- selected profile and execution mode in public product terms;
- quality, visible-input, native-memory, latency, throughput, and reuse fields;
- evidence tier, date, source artifact, and PRA commit.

`NOT_MEASURED` means unknown. It does not mean zero or parity.

## Promotion boundary

Structural compatibility, learned-router quality, Native Memory correctness,
and Native Serving economics are separate claims. A router validated on QASPER
does not imply HotpotQA transfer. A model-family mapping does not qualify every
quantization or engine. A smoke profile remains calibration-pending until a
held-out workload supports promotion.

Requalify locally:

```bash
pra evaluate BASE_MODEL -e ENGINE -a OWNER/BUNDLE \
  -D qasper --include-native-memory -o .pra/runs/qualification
pra recommend .pra/runs/qualification
pra report .pra/runs/qualification --format html
```
