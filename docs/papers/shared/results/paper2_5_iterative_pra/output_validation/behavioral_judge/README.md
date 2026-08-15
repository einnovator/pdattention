# Paper 2 Behavioral-Equivalence Judge Package

This directory contains 588 blind items in 15 optional batches.

For a single-file handoff, send `behavioral_judge_llm_package.json` to each external judge. It
contains the instructions, response schema, and all blind items. Alternatively, send
`behavioral_judge_prompt.txt` followed by `behavioral_judge_items.json` or one file from
`batches/`. Never send or commit `behavioral_judge_truth.json` or its split copies; these
gitignored files are private unblinding metadata.

The blind item file intentionally contains only opaque IDs, task type, the common user prompt,
answers A/B, and requested score names. The truth file records condition labels, deterministic
order, generation metadata, source artifacts, and hashes for later unblinding.

Validate judge output against `behavioral_judge_response.schema.json`. The schema constrains
score ranges and fields; the textual prompt additionally limits each reason to 40 words.

Once external judging begins, freeze the generated files. Do not regenerate, reorder, or edit
items between judges; preserve the gitignored `behavioral_judge_truth.json` privately for
unblinding.

Calibration groups include identical-answer and controlled-corruption pairs. Paraphrase and
native-sampling controls are omitted when no independently recorded generations exist. Results
must be aggregated separately by `comparison_group`; do not silently pool native-context,
PRA-fraction, or adaptation conditions.

## External responses

`responses/behavioral_judge_response_gpt56sol.json` contains all 588 presentations (294 pairs).
`responses/behavioral_judge_response_claude_sonnet5_partial.json` contains 304 presentations that
form 152 complete pairs. The latter is intentionally named `partial`; it is not padded or imputed.

`behavioral_judge_external_metrics.json` is the public pair-collapsed aggregate. It was generated
with the scorer's explicit `--allow-partial` mode, which records per-judge coverage and computes
cross-judge diagnostics only on shared pair IDs. Strict complete coverage remains the default.
The GPT response passes the identical/corrupted calibration anchors; the partial Claude response
does not, so the paper retains Claude as an instrument diagnostic rather than efficacy evidence.
