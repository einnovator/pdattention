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

The replacement boundary-free candidate is atomic E2: infer instruction epochs
from genuine user-message provenance, retain the active and two most recent
complete epochs, and atomically retire older terminally closed instructions
with their interaction. Current evidence is 2/2 reset-workspace forks of one
task at 38.17% own-trajectory saving. It remains pending Easy-14 confirmation.

- `mlx_task01_causal_tail/`: Qwen2.5-Coder-14B MLX, Task 01. The engine honors
  the exact frozen record plan with zero selected-history re-encoding/copy, but
  the task fails after nine calls at 89.57% aggregate logical retention.
- `vllm_task05_causal_tail/`: Qwen2.5-Coder-7B AWQ, Task 05. The engine again
  reports zero selected-history re-encoding/copy. The trajectory diverges at
  action 3, loops, and reaches bounded 8,192-token context exhaustion after 28
  calls at 88.05% aggregate logical retention; the official grader records 0/1.

Both engines had already passed their matching plain/PRA-100 behavioral gates.
The transfer therefore isolates a policy-quality failure from the native K/V
correctness gate. Paper 8.5 owns subsequent matched-budget policy and oracle
work; Paper 4.5 retains the task-aware progress-spine policy for its next engine
experiments. No native engine experiment may treat P1 as a generic default.
Atomic E2 transfers only after its frozen Easy-14 gate passes, without
retuning, and its text-replay saving is not physical cache saving.
