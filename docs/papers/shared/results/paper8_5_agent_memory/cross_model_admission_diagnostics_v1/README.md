# Cross-model autonomous-admission diagnostics

These runs test whether a model/mini-swe-agent pair can establish a valid
FULL control before any PRA selection policy is applied. They are deliberately
excluded from accuracy--saving curves because neither execution reached a
primary autonomous submission.

| Model / task / scaffold | Requests observed | Workspace outcome | Admission |
|---|---:|---|---|
| Qwen2.5-Coder-14B / Task 2 / backticks | 25 | no tracked mutation; 21/24 executed commands returned nonzero, including 19 failed writes | diagnostic failure |
| Qwen2.5-Coder-14B / Task 3 / XML | 16 | correct one-line source patch exists, but no primary submission; one response repeats seven times and 12/16 commands return nonzero | diagnostic failure |

Both runs use MLX revision `29efdbab55a161237ab1e432a3abaf6c7ae2b477`,
its exact tokenizer snapshot, temperature zero, top-p one, seed zero and a
1,024-token completion ceiling. The backtick run shows an edit-interface
failure. Switching only the mini-swe-agent action encoding to XML repairs the
source edit, but exposes a distinct termination/submission failure. The
correct Task-3 patch snapshot is retained to separate workspace capability
from the primary submission metric.

These are stopped diagnostics, not official task failures and not evidence
about a selective policy. A selective arm is correctly withheld until the same
agent/model/scaffold produces a repeat-qualified FULL submission.
