# Cross-agent persistent-session matrix v3

This directory is the immutable evidence root for the prospective agent-by-session-length campaign. The common model is `qwen3-coder:30b-ctx131k` (digest `4abd6222c3c1a34f94cc04542bfe08523213f915b59b4a6443059ef741e8d90f`), decoded with temperature 0, seed 0 and top-p 1. The tokenizer revision is `06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca`. N=1 uses SWE-bench Verified task `django__django-15277`; N=3 adds `django__django-15368` and `scikit-learn__scikit-learn-13135` in that order.

The campaign separates four gates:

1. official patch resolution;
2. autonomous semantic completion, rather than transport exit alone;
3. no increase in aggregate model/tool calls;
4. paired saving only when FULL and treatment have the same agent, task count, task order and declared controls.

## Locked N=1 admissions

| Agent | FULL resolved / calls | tail-90 resolved / calls | Autonomous treatment | Own saving | Paired saving | N=3 decision |
|---|---:|---:|---|---:|---:|---|
| OpenCode | 1/1 / 15 | 1/1 / 19 | yes | 12.48% | 59.86% | reject: +4 calls |
| Pi | 1/1 / 49 | 1/1 / 27 | yes | 52.68% | 77.03% | admit |
| Kilo | 1/1 / 24 | 1/1 / 13 | no | 9.14% | 82.74% | reject: external stop |
| PRA Agent | 1/1 / 19 | 1/1 / 16 | yes | 20.53% | 23.95% | admit |

`tail-90` is a hard ceiling over complete causal groups, not a promise of exactly 90% realized retention. Pi's very large early observation cannot fit without unsafe post-hoc rewriting, so the realized retention is much lower and the saving much higher than 10%. This is a valid ceiling-policy outcome but motivates hierarchical child spans created before first model-visible materialization.

## N=3 status at this checkpoint

PRA Agent's boundary-free Recent Frontier M2/P1 arm resolves all 3 tasks and omits 33.19% of its own full-history counterfactual in 39 requests. It is not promoted: an HTTP 500 occurs after the final verified patch, so autonomous completion is false, and its fresh contemporaneous FULL control fails Task 1 after entering a repeated inspection-budget recovery loop. The older 3/3 FULL artifact is retained as `full_prior`, but is not used for paired saving because its prefix trajectory differs before selection can act.

Pi N=3 FULL is in progress on the direct Big Mac endpoint. No N=6 arm is admitted until a contemporaneous N=3 FULL/treatment pair preserves every FULL success, completes autonomously and does not increase aggregate calls.

`reduction/matrix.csv` and `reduction/matrix.md` are reducer outputs. Missing paired values are deliberate fail-closed results, not zeros.
