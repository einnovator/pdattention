# Multi-issue frozen first-action smoke

This is a one-decision mechanism smoke, not an autonomous accuracy result. It
uses `qwen3-coder:30b`, temperature zero, seed zero, the frozen Qwen3-Coder
tokenizer, and the composed two- and three-issue histories.

Both contemporaneous persistent-FULL calls changed the command stored in the
historical successful trajectory. This is another direct instability control:
temperature zero and a fixed seed did not reproduce the old trajectory.
Treatments are therefore compared with the contemporaneous FULL command, not
with the old historical command.

| Issues | Policy | Materialized/full logical tokens | Retention | Exact command vs contemporaneous FULL |
|---:|---|---:|---:|---:|
| 2 | FULL persistent | 11,247 / 11,247 | 100.00% | control |
| 2 | Completed-episode spine | 3,939 / 11,247 | 35.02% | 0/1 |
| 2 | Active episode only | 1,495 / 11,247 | 13.29% | 0/1 |
| 3 | FULL persistent | 25,226 / 25,226 | 100.00% | control |
| 3 | Completed-episode spine | 7,176 / 25,226 | 28.45% | 1/1 |
| 3 | Active episode only | 3,001 / 25,226 | 11.90% | 1/1 |

The opposite N=2 and N=3 outcomes show why this smoke cannot rank policies.
The next admissible quality evidence is repeated autonomous execution with
official issue resolution, calls-to-solution, and strict fresh/persistent FULL
pairing. These files remain useful as a recordization, selection, endpoint, and
token-accounting audit.

