# Persistent-session N=3 reliability and R4 gate

Campaign: `paper85-autonomous-persistent-spine-n3-reliability-v1`

This cohort uses the direct `.8` to `.6` endpoint, frozen
`qwen3-coder:30b` model and tokenizer revisions, temperature zero, clean
SWE-bench workspaces per issue, and one persistent conversation identity for
the ordered Django, Pytest, and Scikit-learn sequence.

## FULL reliability control

All three infrastructure-clean persistent-FULL repeats resolve all three
issues. Each repeat uses 53 calls; cumulative logical input is 522,479,
550,355, and 553,263 tokens. The cohort therefore contains 9/9 official issue
resolutions and 3/3 all-solved sequences.

A separate earlier repeat is retained as a submission-reliability observation:
all three terminal submissions were invalid non-diff payloads, so its primary
outcome is 0/3. Regrading the saved workspace patches resolves all three
issues. It is reported as 0/3 unconditional submission reliability and 3/3
auxiliary workspace capability, never substituted into the primary score.

## R4/M2/V2 gate

R4 shares the first FULL episode by exact artifact identity and selects
completed history only from issue 2 onward.

| Repeat | FULL resolved | R4 resolved | FULL calls | R4 calls | R4 full tokens | R4 selected tokens | Candidate gross save | Paired raw save |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | 2/3 | 53 | 49 | 539,956 | 377,476 | 30.09% | 27.75% |
| 2 | 3/3 | 2/3 | 53 | 45 | 439,877 | 311,333 | 29.22% | 43.43% |
| aggregate | 6/6 | 4/6 | 106 | 94 | 979,833 | 688,809 | 29.70% | 35.80% |

The arm reaches the target saving interval under the paired raw coordinate and
reduces calls, but loses the Scikit-learn success in both repeats. Its
predeclared failure-aware saving is therefore 0%, and it fails the zero-loss
gate. One failure submits an empty patch after a regex-based edit
does not match; the other submits the correct semantic operation at invalid
indentation. These are genuine task failures, not grader or transport errors.

Temperature zero is not exact trajectory replay on this backend. The first
three Pytest selected-message hashes are identical across R4 repeats, while
the corresponding response hashes differ. Repeated official outcomes are
therefore required even when the logical input and generation parameters are
frozen.

The adaptive R6/M2/V2 repair resolves 3/3, but Scikit-learn expands to 27
calls and the sequence to 63 calls versus 53 for paired FULL. It materializes
538,318 of 807,886 tokens on its own trajectory (33.37% candidate-gross
saving), but paired FULL consumes only 522,479 tokens. Paired raw and
failure-aware saving are therefore both -3.03%, and the arm fails the
no-call-increase and saving gates.

R8/M2/V2 has no scientific outcome in this bundle. Its direct-endpoint
attempt stopped after the first Pytest request when `.8` could no longer reach
the otherwise healthy dedicated `.6:11435` listener. The immutable output is
retained as infrastructure evidence and excluded from every denominator.

The committed `campaign_state.json`, `plan_audit.json`, reduction, and plots
are generated only from completed, infrastructure-clean cells. Health,
transport, and duplicate-writer retries remain quarantined in immutable retry
directories and are excluded from scientific denominators.
