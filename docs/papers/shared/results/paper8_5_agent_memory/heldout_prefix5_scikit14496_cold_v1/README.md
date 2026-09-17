# Third cold-state prompt-pinned task

This bundle evaluates `scikit-learn__scikit-learn-14496` as episode six after
the same frozen five-episode boundary-free prefix used by the other held-out
qualifications. Both arms use `qwen3-coder:30b`, temperature zero, top-p one,
seed zero, the pinned tokenizer, a cold model unload, and the same two-token
warm-up.

The pair is identity-valid: its manifests agree on prefix hash, workspace
source hash, container image, model and tokenizer revisions, decoding settings,
harness and grader versions, boundary mode, and pair ID.

| Arm | Official | Calls | Materialized input | Own-trajectory saving | Paired saving vs FULL |
|---|---:|---:|---:|---:|---:|
| FULL | 1/1 | 9 | 244,411 | 0% | 0% |
| Prompt-pinned E2+F1C | 1/1 | 9 | 152,801 | 38.53% | 37.48% |

No reacquisition event occurs. The treatment preserves the official solve and
the FULL call count while removing 95,787 tokens from its own full-history
counterfactual.

## Three-identity aggregate

Combining one task-clustered observation for each of
`django__django-15741`, `django__django-15368`, and
`scikit-learn__scikit-learn-14496` gives:

- official resolution: 3/3 treatment and 3/3 paired FULL;
- materialized input: 539,022 treatment versus 924,686 FULL;
- paired saving: 41.71%;
- own-trajectory saving: 39.45%;
- calls: 33 treatment versus 34 FULL.

This clears the predeclared three-identity 30--50%, zero-loss, and
no-aggregate-call-increase discovery gate. It promotes E2+F1C to replication
and engine-transfer candidate status. Three successes do not establish a
population accuracy rate or a production default; repeat qualification,
additional identities, cross-model transfer, and autonomous post-fix engine
runs remain required.

## Frozen Paper 4.5 handoff

`paper4_5_e2f1c_frozen_plan.jsonl` exports all nine treatment requests without
rerunning the selector. Its adjacent request replay and manifest bind the
source prefix, trajectory, selection trace, generation settings, and 256-token
physical segmentation. The exporter validates compact closure receipts against
the recorded wire-plan replacements; these receipts are materialized selected
records, not hashes incorrectly searched for in the unmodified FULL request.

- frozen-plan SHA-256: `3f1666323bf6d7409a70a7754210f5e79c7b92029bda5ad1f232ff1f66e596c9`;
- request-replay SHA-256: `ec9c8c12a2830d0e27ab9bcb4ef1ee5087fae35abaa7423ae3f0b2fac698eeed`;
- source prefix SHA-256: `c95cbae606a68e30f776849df0dc09c84caa09430479d45fda30b4cdf1442283`.
