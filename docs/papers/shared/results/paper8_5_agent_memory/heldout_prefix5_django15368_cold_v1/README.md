# Second cold-state prompt-pinned task

This bundle evaluates `django__django-15368` as a second task identity after
the same frozen five-episode boundary-free prefix used by the first held-out
qualification.  Every scored row uses `qwen3-coder:30b`, temperature zero,
top-p one, seed zero, the pinned tokenizer, a cold model unload, and the same
two-token warm-up.

## Admissible paired result

| Arm | Boundary mode | Official | Calls | Materialized input | Own-trajectory saving | Paired saving vs FULL |
|---|---|---:|---:|---:|---:|---:|
| FULL r02 | boundary-free | 1/1 | 12 | 318,595 | 0% | 0% |
| Prompt-pinned E2+F1C | boundary-free | 1/1 | 16 | 260,578 | 39.52% | 18.21% |
| Prompt-pinned E2+F1 | boundary-free | 1/1 | 17 | 299,854 | 35.05% | 5.88% |
| Prompt-pinned E3+F1C | boundary-free | 1/1 | 24 | 439,404 | 33.27% | -37.92% |

All four boundary-free rows start with the same 25,501-token source history.
E2+F1C preserves official resolution but adds four calls, so its 39.52%
per-request/own-trajectory reduction becomes only 18.21% paired end-to-end
saving.  Retaining a whole finalization is worse than its compact receipt, and
retaining a third complete epoch is dominated because the additional calls
erase the per-request reduction.

## Quarantined control

`FULL_r01_coldreset_retry01` resolves in seven calls and uses 184,013 input
tokens, but its manifest records `boundary_mode=explicit`.  The explicit
boundary changes the model-visible prefix and its first request contains
25,647 tokens rather than 25,501.  It is retained for audit and must never be
paired with the boundary-free candidate arms.

## Two-identity aggregate

Combining one task-clustered observation from this bundle with the earlier
`django__django-15741` cold qualification gives the current E2+F1C discovery
point:

- official resolution: 2/2 candidate and 2/2 FULL;
- materialized input: 386,221 candidate versus 680,275 FULL;
- paired saving: 43.23%;
- own-trajectory saving: 39.81%;
- calls: 24 candidate versus 25 FULL.

This clears the numerical 30--50%/zero-loss/no-aggregate-call-increase gate
over two identities only.  It is not an accuracy estimate or a production
default.  A third cold-matched task identity and repeats remain required.
