# Paper 2 claim ledger

Pass-1 ledger for the manuscript at reviewed base commit
`139b4dc1229ad1d7af6f1486e70ddc99704ebd55`. It records what each headline can
support without treating seeds, model executions, or A/B reversals as independent
questions.

| Headline | Evidence tier and exact artifact | Split and independent unit | Replication | Memory / budget / comparator | Supported interpretation |
|---|---|---|---|---|---|
| Exact disabled-path behavior on Qwen3-0.6B | Loaded HF checkpoint; `qa/qwen3_0_6b_hidden_postrope_confirmation.json`, `qwen/qwen3_0_6b_first_night.json` | Frozen model regression inputs, not a QA sample | Deterministic parity | PRA disabled delegates to original attention | Tested logits, hidden states, greedy output, and cache contracts are exact. This does not establish preservation when memory is active. |
| Exact disabled-path behavior on SmolLM2-135M | Loaded HF checkpoint `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; `productization/llama_product_demo.json` plus repository parity tests | Regression inputs and four-example product demo | Deterministic parity; four task examples for systems demo | PRA disabled vs original model; late-eight active path separately | Llama-family adapter compatibility at this checkpoint. It is not an official Meta Llama result or benchmark QA result. |
| Exact disabled path and native hybrid layout on Gemma3-1B-IT | Loaded official checkpoint, revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`; `gemma3_1b/parity_native_kv.json`, `gemma3_1b/gemma3_1b_summary.json` | Regression inputs; eight causal-probe questions | Deterministic parity; five router seeds reuse fixed identities | PRA only at global layers 17 and 23 in routed probe | Tested disabled tensors and cache are exact; active memory executes through native global layers. This does not prove optimal placement. |
| Matched learned-router improvement | Loaded Qwen3-0.6B features; `routing/learned_adapter/learned_router_extended_summary.json` | Held-out test: 32 unique identities (16 HotpotQA, 16 QASPER), disjoint from 48 train and 16 validation | Seeds 11, 23, 37, 53, 71 on the same 32 identities | 32-token parents; exact top-k; learned asymmetric-128 vs frozen last-state cosine on identical identities | Any-evidence R@3 improves from .469 to `.631 +/- .069`; paired delta `.163 +/- .069`. Seeds measure optimization variation, not question sampling. |
| Requested-fraction routing curve | Mixed routing artifacts summarized in `shared/results/recall_sparsity/paper2_hf/`; learned values also in `learned_router_extended_summary.json` | Historical parameter-free cohort: 16 identities; learned cohort: 32 identities | Learned row: five optimization seeds | Requested parent fraction before physical admission | Historical and learned curves must not form one comparative frontier. On the learned cohort, any-evidence recall is .831/.956 at 10/20% requested parents; this is not identity/all-evidence recall or active-K/V fraction. |
| Oracle-depth causal effect and audit | Loaded Qwen3-0.6B; `multilayer_pra/oracle_layer_depth.json`, `oracle_gap_audit/oracle_gap_audit.json` | Four fixed questions per dataset | Deterministic replay | 128 direct prompt tokens; 32-token parents; 512-token memory cap; no-memory comparator | Last-14 HotpotQA replay gains 3.641 nats per gold token; all-layer loses .766. Exact tested identity/K/V invariants narrow, but do not eliminate, the oracle gap. |
| QASPER 32-token polarity diagnostic | Loaded Qwen3-0.6B; `error_analysis/generation_error_analysis.json` | Eight fixed held-out QASPER questions: five yes, three no | Frozen rows deterministic; residual seeds 11, 23, 37, 53, 71 reuse the same questions | Routed frozen, oracle frozen, residual-16 routed; 32 generated tokens; always-yes majority control | Frozen is 2/8, oracle is 5/8, residual is 29/40 over seed-question rows; majority is 5/8. Per-class residual is 24/25 yes and 5/15 no. Identity-level rates are `[1,1,1,0,0,1,1,.8]`; mean .725 with clipped identity-cluster t interval [.346,1.000]. Exploratory, not benchmark accuracy. |
| Blind behavioral preference | `behavioral_judge/behavioral_judge_results.json`; frozen package at commit `f86ebe1`, randomization seed 1234 | QASPER residual-vs-frozen: 40 pairs = eight identities by five seeds; each A/B reversal is not independent | Two judges, same frozen pairs | Eight generated tokens; prompt and answers only; no evidence or gold answer | Each judge: 9/40 residual preferred, 1/40 frozen preferred, 30/40 zero. This measures judged plausibility/equivalence/preference, not factual correctness. QASPER frozen-vs-no-context mean quality is negative for both judges. |
| Larger-model frozen-consumption replication | Separate MLX backend; `mac_scaling/mlx_scaling_summary.json`; pinned 4-bit MLX-community Qwen3 8B/14B/32B/30B-A3B revisions listed in the manuscript | The same 15 unique selected-evidence questions (five each QASPER/HotpotQA/2Wiki) run on four models | The summary records `seed_count=5`; the agreement denominator remains 15 executions per model, and the independent question count is 15 | Identical selected token IDs and original positions; selected text E0 vs concatenated or segmented native E2 | Concatenated E2 preserves 60/60 model-question executions; segmented E2 preserves 57/60. This tests consumption, not router quality, HF-adapter scale, or 60 independent questions. Exact MLX package version is missing from the summary. |
| Extended family compatibility | `family_contracts/extended_family_contracts.json`, `qwen3_moe/qwen3_30b_a3b_contract.json` | Tiny architecture-faithful configurations, not published weights | Deterministic tests | Disabled parity and active native-K/V smoke | Qwen3-MoE, Mistral 3, and GPT-OSS ownership contracts are implementation gates only. Qwen3.8 is specification-only. |

## Existing-evidence statistical notes

- The Wilson 95% intervals computed from the eight QASPER identities are
  `[.071, .591]` for 2/8 and `[.306, .863]` for 5/8.
- The residual-16 identity-cluster interval uses the eight per-question means
  across five seeds and a two-sided Student-t interval with 7 degrees of freedom,
  clipped to `[0,1]`. It is intentionally much wider than the optimization-seed
  variation (`+/- .056`).
- The routing `+/-` values are variation across five optimization seeds on one
  fixed identity cohort unless a table explicitly says otherwise.

## Experimental requirements not satisfied by existing artifacts

1. A new untouched QASPER cohort for the positive task claim, with class balance,
   per-class performance, and question-level uncertainty predeclared.
2. A matched frozen/adapted x disabled/routed/oracle/wrong-memory factorial. The
   current rows change multiple conditions and do not isolate additive router and
   consumer effects.
3. Evidence-grounded human adjudication of all eight existing QASPER identities
   and a small stratified HotpotQA subset, under natural stopping budgets and a
   defined rubric, using new package IDs.
4. Active-path preservation tests with irrelevant memory, including false
   activation, ordinary-task drift, and erroneous gate opening.
5. Parameter-free rankings recomputed on the learned router's frozen identities
   if a broader matched recall-sparsity comparison is desired.
6. Exact MLX/runtime package provenance for the larger-model consumption campaign.
