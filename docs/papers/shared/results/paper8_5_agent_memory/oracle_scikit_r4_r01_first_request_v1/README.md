# Scikit first-divergence oracle

This diagnostic freezes the R4/M2/V2 repeat-1 completed-episode prefix and the
first Scikit-learn request. It reconstructs the exact candidate selected-message
digest, then compares byte-identical repeated requests and one retired causal
group restored at a time.

| Arm | Valid Bash action | Unique responses | Interpretation |
|---|---:|---:|---|
| FULL on the same frozen prefix | 5/5 | 2 | Valid control, with backend trajectory variance |
| R4/M2/V2 candidate | 0/5 | 1 | Stable format failure under the selected prompt |
| One retired group added back | 24/25 | 25 trials | Most small prompt restorations recover a valid action |

The candidate already preserved all available verification and finalization
records, plus the two latest mutation turns per completed issue. Therefore the
first failure is not explained by dropping the declared mutation,
edit-verification, or submission floors. The 25 add-backs restore 82--2,513
tokens. Twenty-four produce one valid Bash action and only
`episode-01:turn:t0016` (139 tokens) remains invalid. Every valid action is a
semantically equivalent file-search operation, represented by three exact
command hashes.

This establishes a selection-induced prompt-conditioning effect at the first
divergence, while the two FULL response hashes show that temperature zero is
not bitwise deterministic. It does **not** identify one semantically necessary
old fact: many unrelated old turns repair the format. The deployable follow-up
is therefore a content-independent protocol floor (latest clean completed
action--observation exemplar), not an oracle-selected add-back. Autonomous task
resolution is still required before that floor can be promoted.

The frozen P1 rule adds records `episode-01:m44,m45` (138 tokens). Its selected
message digest is exactly
`de80461231285cee0593a89c71199971bf6e934959aac02427283988f00ff4f9`, the
same materialization as diagnostic `addback_005`; the generic rule was defined
from record roles, not from the future Scikit action.

`oracle.json` is the complete digest-bound result. Add-back trials are
diagnostic oracle evidence and are excluded from policy accuracy curves.
