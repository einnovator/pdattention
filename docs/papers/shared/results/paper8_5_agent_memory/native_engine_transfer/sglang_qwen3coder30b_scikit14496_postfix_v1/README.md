# SGLang-MLX consumption-path control

This compact audit records the post-fix autonomous comparison on
`scikit-learn__scikit-learn-14496` with
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, temperature zero, the same
tokenizer, and the official SWE-bench grader.

Ordinary fresh-prefill FULL and native live-K/V PRA-100 both solve 1/1 in seven
calls and submit byte-identical patches. PRA-100 bootstraps 26,370 source tokens
once, then performs six native-reuse calls with zero selected-history
re-encoding, K/V copy, host-to-device transfer, or suffix graft.

This is not exact action-trajectory parity. Actions 1--2 match, then action 3
diverges even though a stopped diagnostic confirms that live source token IDs
and fresh text retokenization are identical through request 3. The residual is
therefore a consumption/numerical-path effect between fresh full prefill and
incremental prefix-cache continuation, not selection or omitted history.

Paper 8.5 does not score this as a reduced-policy result. PRA-90 remains
withheld until Paper 4.5 runs a matched no-selection live-prefix control (or
predeclares a defensible numerical-equivalence criterion). Full traces are in
the Paper 4.5 engine-gate bundle; `comparison.json` preserves the compact
cross-paper facts and gate decision.

