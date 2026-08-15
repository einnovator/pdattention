# Paper 3 Readability Audit

## Narrative

- The introduction asks one question: given selected conceptual memory, what physical native state
  is computationally sufficient?
- Selection, materialization, and learned use are defined before the implementation or experiments.
- The controlled task appears before Qwen so causal interpretation precedes benchmark transfer.
- The paper distinguishes annotation evidence from computationally sufficient memory three times:
  intuition, formal factorization, and experimental interpretation.
- Whole-parent disclosure is described as a control, never as an assumed optimum.

## Results

- Every main figure is introduced with the question it answers.
- Controlled margin, accuracy, K/V, attention, receptive-field, layer, portability, dispersion, and
  budget results are reported in native units.
- Pretrained quality uses paired example bootstrap intervals and states when intervals include zero.
- The preserved eight-example pilot is labeled as provenance and is not pooled with confirmation.
- MuSiQue positives (parent preservation and K/V reduction) are separated from the no-memory
  consumption mismatch.

## Reproducibility

- Implementation references point to logical intervals, budget allocation, cross-shard gather,
  dispatch, compact attention capture, and HF block encoding.
- Tables state seeds, examples, receptive fields, layers, and validation rules.
- Cohort extension and annotation-token mapping are tested and disclosed in Limitations.
- The test matrix and negative-result ledger include the newly discovered long-cohort ownership and
  transport-metric naming boundaries.

## Final Language Checks

- No universal claim that frozen transformers cannot consume PRA memory remains.
- No throughput, cost, production-kernel, or statistical-significance claim is made.
- No learned routing or adaptation result is attributed to Paper 3.
- Paper 4 receives a falsifiable representation-portability hypothesis rather than an assumed fix.
