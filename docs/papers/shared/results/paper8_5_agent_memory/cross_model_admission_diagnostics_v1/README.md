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

## Latest-code Task-3 repeat and protocol-adapter audit

A fresh immutable repeat on the latest branch used the same Qwen2.5-Coder-14B
revision, tokenizer, XML scaffold, temperature zero, top-p one, seed zero and
FULL policy. Three preceding endpoint probes returned identical response
hashes. The first 11 assistant-content hashes and executed-command hashes then
reproduced the earlier stopped diagnostic exactly. The correct source edit was
again present after action 3. Actions 10 and 11 repeated the same failing
command and received the same observation while the workspace fingerprint was
unchanged. The repeat was investigator-stopped at that point rather than
spending the remaining 39 actions on a deterministic no-progress loop. It is
therefore another diagnostic, not an official failure or policy point.

This audit exposed an independent harness defect. The selector-side command
decoder and execution-receipt join recognized mini-swe-agent's fenced action
syntax but not its official XML action syntax. The environment still executed
the XML commands, while the policy layer marked their sidecars unparseable and
would have failed closed to FULL. The decoder now accepts either encoding,
requires exactly one action across both, and has regression coverage for exact
XML receipt joins and mixed/multiple-action rejection. This defect did not
cause the FULL no-progress loop because FULL does not remove records; it would
have invalidated any later selective XML run, so no such run is admitted from
the earlier code.
