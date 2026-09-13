# Frozen Paper 8.5 policy transfer

These receipts test whether a frozen engine-independent history policy can be
executed by qualified native PRA paths. They do not rank policies: the imported
causal-tail treatment and the existing task-aware PRA-90 arms have different
realized retention.

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
experiments until a better policy is qualified there.
