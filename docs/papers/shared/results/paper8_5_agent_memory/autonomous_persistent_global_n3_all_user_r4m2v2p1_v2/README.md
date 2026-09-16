# Autonomous persistent-session evidence

Campaign: `paper85-autonomous-persistent-global-n3-all-user-r4m2v2p1-v2`

| Sequence | Repeat | Strategy | Resolved | Calls | Candidate gross | Paired failure-aware | Call delta | Admissible |
|---|---:|---|---:|---:|---:|---:|---:|---|
| independent-n3-boundary-free-all-user-a | 1 | S01_persistent_full/boundary_free_all_user_v2 | 3/3 | 55 | 0.00% | -- | -- | True |
| independent-n3-boundary-free-all-user-a | 1 | S03_global_progress_spine/r4m2v2p1_boundary_free_all_user_v2 | 2/3 | 66 | 60.83% | 0.00% | 11 | True |

## Interpretation

Both arms use one continuous transcript without model- or selector-visible
episode IDs, completion markers, or workspace scopes. The semantic floor is
the system prompt plus every user-authored instruction. Tool observations are
selectable because their typed provenance is `TOOL_OBSERVATION`, even though
mini-swe-agent transports them with `role=user`.

The treatment materializes 293,091 tokens versus 599,397 for paired FULL, a
raw paired saving of 51.10%. It loses the paired Django success, however, so
the predeclared failure-aware saving is 0%. Calls increase by 11.

| Issue | FULL resolved/calls/tokens | Global R4 resolved/calls/sent | Paired saving |
|---|---:|---:|---:|
| `django__django-15277` | yes / 27 / 153,654 | no / 40 / 121,280 | 21.07% |
| `pytest-dev__pytest-7982` | yes / 14 / 174,747 | yes / 9 / 42,055 | 75.93% |
| `scikit-learn__scikit-learn-13135` | yes / 14 / 270,996 | yes / 17 / 129,756 | 52.12% |

Django first diverges at request 6, immediately after the first causal-group
retirement and before any second issue exists. The failure therefore cannot be
explained by hidden task boundaries or a removed user prompt. This R4 point is
rejected; the conservative adaptive bracket is global R8/M2/V2/P1.
