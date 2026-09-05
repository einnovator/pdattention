# paper4-5-qwen3-coder-30b-terminal-bench-pilot-v3

This is an official-Harbor **No-PRA baseline**. It does not estimate a PRA effect.

- Frozen manifest: `terminal-bench-2.1-portable-pilot-v1` (10 tasks)
- Completed: `30/30` trials
- Admission: `BLOCKED` - No-PRA success 6.7% is below the promotion floor.
- Tasks solved by any harness: `1/10`

| Harness | Runs | Success | Reported input tokens | Token coverage | Model calls | Tool calls | Wall h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aider` | 10 | 0/10 (0.0%) | 0 | 0/10 | 0 | 0 | 1.04 |
| `mini-swe-agent` | 10 | 1/10 (10.0%) | 2,562,781 | 10/10 | 271 | 271 | 1.14 |
| `qwen-coder` | 10 | 1/10 (10.0%) | 4,154,315 | 10/10 | 0 | 143 | 1.84 |

The admission decision requires the complete preregistered matrix. Harness rows are not an agent ranking because prompts, tools, and loop policies differ.
