# Current SGLang-MLX Task 1 autonomous gate

This is the fresh current-stack mini-swe-agent gate for
`django__django-15277`, Qwen2.5-Coder-14B-Instruct-4bit, and the standard
`swebench_backticks.yaml` scaffold. The SGLang server uses the guarded import
path introduced at `d4d97732`; the requested-retention handoff is corrected at
Paper 4.5 revision `5c6907a8` and Paper 8.5 revision `717389c1`.

| arm | official solve | calls | assistant content vs plain | commands vs plain | selected-history re-encoding | selected-K/V copy | total K/V copy | peak temporary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 1/1 | 6 | reference | reference | n/a | n/a | n/a | n/a |
| PRA-100 | 1/1 | 6 | 6/6 | 6/6 | 0 tokens | 0 B | 0 B | 51.08 MB |
| matched tail-90 | 1/1 | 6 | 3/6 | 6/6 | 0 tokens | 0 B | 37.16 MB suffix graft | 1.61 GB |

All three arms submit the byte-identical passing patch. PRA-100 is an exact
behavioral control. Tail-90 first changes reasoning text at request 4 but keeps
all six shell commands and the patch unchanged. It realizes 4.45% own
materialized-token saving, 4.09% paired input-plus-completion saving against
plain, and 4.71% selected-K/V reduction against PRA-100. This short task does
not provide the 30--50% opportunity expected from persistent multi-task
sessions.

The physical result is deliberately reported separately. Tail-90 never
re-encodes selected history and never copies selected K/V for attachment, but
its three sparse requests copy 37.16 MB for canonical suffix grafts and peak at
1.61 GB of consumer temporary memory. Thus the logical policy passes quality
while the current sparse consumer remains economically unattractive on this
short trajectory.

The diagnostic ladder also found a confound that is now explicit. Adding a
workflow anchor and transactional-edit restrictions changed the agent itself:
v7 entered an exact fixed point, v10 escaped the loop but edited the wrong
class, and v11/v12 still mislocalized a generic `__init__` search. Returning to
the unmodified standard scaffold immediately restored the archived six-call
solve. Those stopped runs are agent-scaffold diagnostics and are not counted as
PRA failures.

The immutable full artifacts remain on the Big Mac under
`/Users/admin.jorge.simao/git/rd/paper45-sglang-task01-d4d97732/`. SHA-256
identities and reduced metrics are in `summary.json`.
