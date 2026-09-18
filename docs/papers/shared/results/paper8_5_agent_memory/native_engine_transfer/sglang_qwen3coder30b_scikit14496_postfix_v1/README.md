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

The matched no-selection live-prefix control subsequently matches PRA-100 on
all seven request inputs, responses and commands, resolves 1/1, and submits
the same patch.  That matched-consumption gate permits a reduced-policy run
without attributing fresh-prefill numerical sensitivity to selection.

The resulting E2+F1C arm also resolves 1/1 and submits the byte-identical
patch.  It saves 35.07% of selected message-content tokens, 39.21% after
compact closure materialization, and 37.21% of source K/V exposure over its
own 15-call trajectory.  It is nevertheless a valid policy-quality negative:
the matched FULL control needs seven calls, divergence begins at request four,
and the 39.11% within-trajectory input-plus-completion saving becomes a 33.10%
paired workload increase once those extra calls are charged.  E2+F1C is not a
default-profile candidate on this evidence.  Its next experiment is a
request-four oracle add-back and stronger current-task progress spine.

Full traces are in the Paper 4.5 engine-gate bundle.  This directory keeps the
compact comparison plus the E2+F1C metrics, official result and prediction.
