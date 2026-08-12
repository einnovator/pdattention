# Paper 2 Behavioral-Equivalence Judge Package

This directory contains 304 blind items in 5 optional batches.

Send `behavioral_judge_prompt.txt` followed by either `behavioral_judge_items.json` or one
file from `batches/` to each external judge. Never send `behavioral_judge_truth.json`.

The blind item file intentionally contains only opaque IDs, task type, the common user prompt,
answers A/B, and requested score names. The truth file records condition labels, deterministic
order, generation metadata, source artifacts, and hashes for later unblinding.

Validate judge output against `behavioral_judge_response.schema.json`. The schema constrains
score ranges and fields; the textual prompt additionally limits each reason to 40 words.

Calibration groups include identical-answer and controlled-corruption pairs. Paraphrase and
native-sampling controls are omitted when no independently recorded generations exist. Results
must be aggregated separately by `comparison_group`; do not silently pool native-context,
PRA-fraction, or adaptation conditions.
