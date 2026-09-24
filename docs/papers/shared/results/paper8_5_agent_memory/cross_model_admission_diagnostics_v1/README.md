# Cross-model autonomous-admission diagnostics

These runs test whether a model/mini-swe-agent pair can establish a valid
FULL control before any PRA selection policy is applied. They are deliberately
excluded from accuracy--saving curves unless a FULL control first resolves
officially. No selective policy is executed after a failed or stopped FULL.

| Model / task / scaffold | Requests observed | Workspace outcome | Admission |
|---|---:|---|---|
| Qwen2.5-Coder-14B / Task 2 / backticks | 25 | no tracked mutation; 21/24 executed commands returned nonzero, including 19 failed writes | diagnostic failure |
| Qwen2.5-Coder-14B / Task 3 / XML | 16 | correct one-line source patch exists, but no primary submission; one response repeats seven times and 12/16 commands return nonzero | diagnostic failure |
| Qwen3-14B / Task 1 / XML, thinking disabled | 14 | intended `CharField` guard plus an erroneous duplicate `BinaryField` guard; same failing write repeats three times | diagnostic failure |
| Qwen3-14B / Task 4 / XML, thinking disabled | 13 | submits a three-hunk patch that adds an undefined `lazy` reference | official FULL failure |
| Qwen3-14B / Task 3 / XML, thinking disabled | 19 | no tracked mutation; recurrent command groups revisit the same unchanged workspace | stopped diagnostic |

The Qwen2.5 runs use MLX revision
`29efdbab55a161237ab1e432a3abaf6c7ae2b477`; the Qwen3 runs use revision
`a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`. Each uses its exact tokenizer
snapshot, temperature zero, top-p one, seed zero and a 1,024-token completion
ceiling. The Qwen2.5 backtick run shows an edit-interface failure. Switching
only the mini-swe-agent action encoding to XML repairs the source edit, but
exposes a distinct termination/submission failure. The correct Task-3 patch
snapshot is retained to separate workspace capability from the primary
submission metric.

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
XML receipt joins and mixed/multiple-action rejection. Progress-role detection
also removes either action envelope before classifying reasoning text, so the
two serializations now produce the same logical roles. This defect did not
cause the FULL no-progress loop because FULL does not remove records; it would
have invalidated any later selective XML run, so no such run is admitted from
the earlier code.

## Qwen3-14B serving-profile admission

The first Qwen3-14B control makes the serving profile explicit. With the MLX
default thinking template, a 16-token health request spends the complete budget
inside hidden reasoning and emits no visible action. The bounded launcher now
records `enable_thinking=false`; three catalog probes and a subsequent exact
`PRA_HEALTH_OK` completion pass before the autonomous control. The OpenAI model
catalog observes the served model ID but does not expose a weight digest, so
the local snapshot revision is labelled declared rather than endpoint-observed.

The Task-1 FULL control reads source normally and begins the intended
`CharField.max_length is not None` repair. A broad write also inserts a second
guard into the already guarded `BinaryField` block, producing invalid syntax.
After two different failed recovery commands, actions 11--13 repeat the same
write command, return code 2, against the same workspace fingerprint. The
investigator stops the run at 14 observed executions rather than spending the
remaining 26 calls on a no-progress cycle. There is no primary submission and
no selective arm. This is a model/scaffold admission diagnostic, not a task
failure estimate or a policy-quality point.

Two same-profile follow-ups prevent a single-task conclusion. Task 4 completes
and submits after 13 actions and 74,283 cumulative FULL input tokens, but the
official grader rejects its patch. The model adds a `lazy` parameter to
`get_format()`, changes two unrelated conditionals, and leaves an undefined
`lazy` reference in `number_format()`. This is a clean official FULL failure,
not a transport or submission failure. On the simpler Task 3, the model reaches
19 executions without any tracked source mutation. Five command groups recur
against the same workspace state; only `patch.txt` and `reproduce_issue.py`
exist when the investigator stops the run. It is a stopped diagnostic rather
than an official failure.

The three attempted Qwen3-14B identities therefore do not establish one
plain-success pair. Running a selective arm would condition on a favorable
baseline and is prohibited. The next model-transfer execution must use a
plain-success model/task pair, such as the already qualified Qwen3-Coder-30B
cohort, before comparing FULL with a frozen policy.
