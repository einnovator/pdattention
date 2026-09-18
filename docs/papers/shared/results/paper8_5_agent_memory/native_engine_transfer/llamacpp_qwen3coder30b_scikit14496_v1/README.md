# Post-fix native llama.cpp transfer gate

This bundle is the first autonomous transfer of the frozen Paper 8.5 agent-memory policy into Paper 4.5's corrected resident-K/V path. It uses `scikit-learn__scikit-learn-14496` as episode 6 of the locked independent persistent session, Qwen3-Coder 30B, temperature zero, top-p one, and seed zero.

Two ordinary FULL controls solve in eight calls and are hash-identical at every request, assistant response, and action. Corrected PRA-100 also solves in eight calls and is exactly identical to both controls. Its first request captures the canonical 26,370-token source once; the remaining seven requests use resident native K/V with zero selected-history re-encoding and zero K/V-copy bytes.

Prompt-pinned E2+F1C also solves in eight calls. It materializes 128,380 versus 213,524 message-content tokens, a 39.88% saving. Across the seven post-bootstrap native requests, weighted K/V omission is 40.23% (113,643 selected versus an estimated 190,141 full K/V tokens), only 0.35 percentage points from the logical estimate. The first two actions match FULL exactly, call 3 begins an alternate trajectory, and same-index actions reconverge on calls 5--8. Completion tokens increase from 748 to 825.

This is a qualified single-task, single-engine transfer point, not a population accuracy or cross-engine result. Earlier HF, MLX, SGLang and vLLM request-level rows predate the final autonomous bootstrap contract and are not used to claim agent accuracy or current savings. Each requires a fresh post-fix autonomous FULL, PRA-100, and reduced-policy rerun.

The raw subdirectories contain the immutable run manifest, selection trace, autonomous metrics, and official SWE-bench result for both FULL controls, PRA-100, and E2+F1C. `summary.json` is the reduced comparison.
