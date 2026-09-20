# Frozen Paper 8.5 policy transfer

These receipts test whether a frozen engine-independent history policy can be
executed by qualified native PRA paths. They do not rank policies: the imported
causal-tail treatment and the existing task-aware PRA-90 arms have different
realized retention.

Paper 8.5 previously treated `paper8.5-r4m2v2p1-v1` as discovery-qualified:
two autonomous executions resolve 6/6 paired issues, save 36.41% and 45.82%
against contemporaneous FULL, and use six and nine fewer calls. A later audit
established that the policy consumes typed issue scope unavailable in an
ordinary continuous session. It is retained as a mechanism upper bound, not a
deployable boundary-free qualification. The frozen historical contract and
source evidence remain in `paper85_r4m2v2p1_qualification/`.

The replacement boundary-free atomic-policy candidates (Paper 8.5 policy
labels, not the E2/E3 engine depths) were tested in a context-qualified
Easy-14 persistent-session campaign. Persistent FULL resolves 10/14 issues.
Atomic E2 is stopped after five issues at 2/5 versus paired FULL 4/5; its 28.22%
raw paired saving is zero under failure-aware accounting. Atomic E3 is stopped
before its first deletion because a full-materialization replay has already
diverged. The exact-prefix Task-5 diagnosis finds E3 nondominated but still
unqualified: 1/2 resolutions, 16.62% paired saving, and one extra call. E2 also
resolves 1/2, but its longer trajectory turns 41.94% gross saving into -106.77%
paired saving relative to FULL.

- `mlx_task01_causal_tail/`: Qwen2.5-Coder-14B MLX, Task 01. The engine honors
  the exact frozen record plan with zero selected-history re-encoding/copy, but
  the task fails after nine calls at 89.57% aggregate logical retention.
- `vllm_task05_causal_tail/`: Qwen2.5-Coder-7B AWQ, Task 05. The engine again
  reports zero selected-history re-encoding/copy. The trajectory diverges at
  action 3, loops, and reaches bounded 8,192-token context exhaustion after 28
  calls at 88.05% aggregate logical retention; the official grader records 0/1.
- `sglang_task01_matched_tail_ad1152b9/`: Qwen2.5-Coder-14B SGLang-MLX,
  Task 01. At 89.57% aggregate logical retention the official grader records
  0/1 after nine calls, with zero selected-history re-encoding, zero selected-
  K/V physical copy, and zero host-to-device transfer. The assistant action
  trajectory and erroneous 94,017-byte submission are byte-identical to the
  direct-MLX causal-tail negative. The first divergence occurs when the only
  omitted causal bundle is the initial `grep` discovery turn.

All three engines had already passed their matching plain/PRA-100 behavioral gates.
The transfer therefore isolates a policy-quality failure from the native K/V
correctness gate. Paper 8.5 owns subsequent matched-budget policy and oracle
work. No native engine experiment may treat P1, E2, or E3 as a generic default.
Agent-Full is the sole qualified correctness profile. Agent-Quality/E3 remains
a named diagnostic candidate, Agent-Balanced/E2 is rejected for default use,
and Agent-Economy/E1 was not advanced. Any future native transfer must use an
exact frozen record plan, preserve its selected-message digest, and treat
text-replay saving as distinct from physical cache saving.

The complete imported Easy-14 decision receipt is in
`paper85_easy14_ctx131k_v2/`; detailed per-task trajectories and curves remain
in the Paper 8.5 appendix to avoid duplicating that policy study here.
