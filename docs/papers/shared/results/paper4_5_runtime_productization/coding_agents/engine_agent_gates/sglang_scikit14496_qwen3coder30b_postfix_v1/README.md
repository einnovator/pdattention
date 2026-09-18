# SGLang-MLX live-history gate: Scikit 14496

This bundle is the first post-fix autonomous SGLang-MLX comparison on the
frozen five-episode persistent prefix and
`scikit-learn__scikit-learn-14496`. Both arms use
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, temperature zero, the same
tokenizer revision, mini-swe-agent 2.4.6, and the official SWE-bench grader.

## Result

| Arm | Official score | Calls | Final patch | Agent wall time |
|---|---:|---:|---|---:|
| ordinary fresh-prefill FULL | 1.0 | 7 | reference patch | 563.03 s |
| native live-K/V PRA-100 | 1.0 | 7 | byte-identical | 279.48 s |
| selection-neutral live-prefix FULL | 1.0 | 7 | byte-identical | 134.61 s |

The corrected PRA-100 path reports zero selected-history re-encoding, zero
selected-history K/V copy, zero host-to-device bytes, and zero canonical
suffix-graft bytes. The first request performs the unavoidable 26,370-token
source bootstrap. Calls 2--7 append only 21--542 new history tokens each.

The earlier full-retention implementation allocated a second full request
cache, copied every generated suffix back to the canonical cache, used roughly
3.2--3.6 GB of temporary memory per call, and failed with Metal OOM at call 7.
The in-place canonical continuation reduces steady-state measured temporary
peak to about 41.4--41.6 MB per call. Bootstrap remains distinct at 5.44 GB.

## Correctness boundary

PRA-100 and fresh-prefill FULL have identical logical inputs through request
3 and identical assistant actions for requests 1--2. They first diverge at
request 3, while retaining the same operation class on all seven calls, the
same call count, the same official solve, and a byte-identical final patch.
The three-request diagnostic confirms that live source token IDs are exactly
equal to fresh text retokenization through the divergence. The remaining
difference is therefore consumption/numerical path sensitivity between fresh
full prefill and incremental prefix-cache continuation, not selection,
retokenization, or omitted history.

The matched selection-neutral live-prefix control resolves the ambiguity.  It
uses the same incremental consumer without excluding any record.  All seven
request-input digests, assistant-content digests, and command digests match
PRA-100 exactly; the submitted patch is also byte-identical.  Thus the
fresh-prefill trajectory gate remains negative, but the matched-consumption
PRA-100 gate is positive.  PRA-90/E2+F1C is now allowed as a selection-quality
experiment.  Wall times remain single-run provenance, not a speed claim.

## Files

- `full/`: ordinary FULL manifest, trace, metrics, official result and patch.
- `pra100/`: corrected native PRA-100 artifacts.
- `live_prefix_control/`: repeat-exact, selection-neutral incremental-cache
  control used for the matched-consumption gate.
- `live_replay_diagnostic/`: stopped three-request diagnostic with live-versus-
  replay token identity counters; it is not an official task result.
- `comparison.json`: compact paired reduction.
