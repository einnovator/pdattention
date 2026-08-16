# PRA Precision, Recall, and Distractor Caution Audit

## Verdict

**B - Partially supported.** The corpus supports a retrieval precision/recall pressure,
content-specific memory effects, and a controlled case in which an exact evidence core
outperforms broader parent materialization. It does not establish the complete chain
`breadth -> lower precision -> attention dilution -> lower quality` as a universal PRA
mechanism.

Paper 2.5 does not isolate softmax dilution from retained V content, residual perturbation,
consumer placement, and later preservation. Paper 3 supplies the cleanest fixed-routing
causal control; its frozen pretrained result is still primarily an efficiency result.

## Metric Contract

For exact evidence K/V `KV_E` and selected K/V `KV_S`:

```text
P_E^KV   = |KV_E intersect KV_S| / |KV_S|
R_E^KV   = |KV_E intersect KV_S| / |KV_E|
P_E^attn = A_E / (A_E + A_D)
S_E^attn = A_E / (A_E + A_D + A_native)
```

`P_E^KV` is selected-set evidence density, `R_E^KV` is evidence coverage,
`P_E^attn` conditions on external-memory attention, and `S_E^attn` is evidence share in
the complete shared softmax. Paper 2.5 uses `A_J` for memory distractors; this audit uses
`A_D` synonymously. Evidence density should not be called precision unless both the
selected denominator and evidence mask are exact.

## Claim Matrix

| Claim | Evidence | Status |
|---|---|---|
| Broader search can improve recall | Paper 2.5 parameter sweeps show that facets, roots, neighbors, hops, and budget can increase entry/path opportunity, with dataset-dependent saturation and failures. | Partial; no universal monotonic law. |
| Broader search lowers precision | Paper 2.5 ordinary memory is distractor dominated (`A_E=.147`, `A_D=.521`), and its parameter audit records increasing candidate/distractor pressure. Granularity effects are dataset dependent. | Directionally supported, not isolated as one breadth sweep. |
| Distractors reduce evidence attention | Shared-softmax competition is mathematical. Paper 3's fixed-router toy intervention shows evidence mass falling `.604 -> .466 -> .292 -> .149` from exact core through broader radii to whole parent. | Supported in the controlled model; partial in Paper 2.5 and pretrained systems. |
| Lower evidence attention reduces margin | Paper 3's matched broadening lowers both evidence mass and margin; exact core beats whole parent by `+3.127` margin. Paper 2.5 oracle improves mass and margin but jointly changes content. | Supported in the controlled materialization study; not uniquely mediated in Paper 2.5. |
| Exact core beats whole parent | Paper 3: 5.25 versus 73.88 K/V tokens, margin gain `+4.107` versus `+.981`, accuracy `.720` versus `.300`. | Supported in the controlled model. |
| Effect is content specific, not sparsity only | Paper 3's same-size wrong core trails the exact core by `4.905` margin. Paper 2.5's matched 20-state wrong memory lowers accuracy to `.058`, while oracle reaches `.398`. | Supported. |
| Pretrained quality improves with less memory | Paper 3 evidence-only reduces K/V by 20.9% on MuSiQue and 44.4% on 2Wiki while preserving whole-parent likelihood within paired uncertainty; absolute MuSiQue quality remains poor. | Unsupported as a resolved quality improvement; supported as efficiency/compression. |
| PRA-native training benefits from less memory | No PRA-native scaling result establishes this. | Future hypothesis. |

## Paper 2.5 Checks

- The central causal cohort freezes 25 checkpoints and evaluates 16 paired held-out
  examples per checkpoint, or 400 model-example units. Table values average windows
  equally; artifact rows contain five-seed means per window. The attention table does
  not attach a confidence interval to each mass.
- `E0`, selected, oracle, wrong, and shuffled conditions match model, query, layers,
  example, and materialization path. Every nonzero condition has exactly 20 layer-token
  K/V states. Oracle labels are intervention-only and unavailable to executable retrieval.
- The final-query masses share one denominator:
  `A_E + A_D + A_native = 1`. Every aggregate row in
  `memory_activity_diagnostics.csv` records `attention_mass_sum=1.0`.
- Ordinary selected memory averages evidence mass `.147` and distractor mass `.521`.
  Oracle averages `.270` and `.301`; accuracy changes `.140 -> .398`, and margin
  `-3.103 -> -.703`.
- Wrong memory has zero evidence mass, distractor mass `.546`, and accuracy `.058`.
  Shuffled selected identities yield evidence mass `.122`, distractor mass `.433`,
  and accuracy `.193`. These controls reject quantity-only and
  absolute-consumer-incapacity explanations.
- The path-improved subgroup is exactly the 59/400 units with
  `iterative complete-path recovery - one-shot complete-path recovery > 0`. Its margin
  gain is `+2.164` with paired-bootstrap 95% CI `[1.336, 2.971]`. This is conditional
  association after retrieval; improved units also move evidence/distractor mass, so it
  is not an isolated mediation estimate.
- The strongest exact two-sided five-seed sign result is `p=.0625`; direction
  consistency should not be described as conventional significance.

## Cross-Paper Causal Boundary

Paper 3 fixes routing to one oracle parent, then changes only disclosed native K/V. The
exact core retains complete annotation-core coverage while broader radius and whole-parent
conditions add surrounding states. Evidence attention decreases monotonically with breadth,
and output margin falls. The same-size wrong-core control establishes that relevance, not
sparsity alone, produces the gain.

This supports: **a smaller correct evidence core can outperform broader parent memory in a
controlled model.** It does not support: **arbitrary K/V reduction improves quality.**

In frozen Qwen confirmation, evidence-only and whole-parent materialization have nearly
identical held-out likelihood and answer metrics while evidence-only activates fewer K/V
states. The warranted claim is **less K/V can preserve measured quality**, not that
selective materialization has already improved pretrained reasoning quality.

Paper 3.5 correctly formulates minimum-effort routing as cost minimization subject to a
quality target. Its E0/E1/E2 action set is nevertheless monotonic in several controls, so
E2 must not be treated as an intrinsic quality ceiling. Future action spaces should permit
widening discovery while narrowing physical materialization and should report raw recall,
precision, evidence attention, quality, K/V, latency, and search work rather than optimizing
recall alone.

## Artifact Provenance

Audit commits:

- Paper 2.5: `363bc10a4f6bb2f7d242959d2663eaa94c2529e1`
- Paper 3: `94d5446383ce4569a518a8b5bbf9d7c00b4c79ec`
- Paper 3.5: `44c208232c64b6bf576126ee00a090fd46f486e5`

Canonical Paper 2.5 artifacts:

| Path | SHA-256 | Rows/fields used |
|---|---|---|
| `docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/memory_activity_diagnostics.csv` | `F01B185D7BCC216837B68EC18D5321FE1C2E50402C48C2D5EF0703F7153C2922` | `iterative_matched`, all windows, `e0/oracle/selected/shuffle/wrong`; three masses and mass sum |
| `docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/oracle_consumption_ceiling.csv` | `744A39FE1C87CF833D7391F6E4A0C06BA157F433A6D9ADC45ADD50F5EC89F540` | Same conditions; accuracy, margin, recall, path, layer-token K/V |
| `docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/causal_memory_paired_effects.csv` | `59E911445AFBA63C7A1206BA5E7E52334C437F3C26F70610A46F35C6ED10EB15` | Five-seed paired condition effects by window |
| `docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa_v6/path_gain_answer_gain_summary.csv` | `A1E71E86113CC9C8B1BCF8900CD0106342FA78139EB27511B126D5432910E5D9` | Path-improved counts and answer effects by window |
| `docs/papers/shared/results/paper2_5_iterative_pra/final_reviewer_patch/iteration_benefit_feature_summary.csv` | `ED6BA8308A49A1384ECD9D50A0170F85329412285F00C8FE9A42EDFAD93FA1EA` | 59-versus-341 pre-decision and post-treatment diagnostics |

Cross-paper artifacts:

| Paper | Path | SHA-256 | Use |
|---|---|---|---|
| 3 | `docs/papers/shared/results/paper3_kv_materialization/toy_materialization/toy_materialization_causal_diagnosis.md` | `97E84AC305450C4FA942EF0F8CDF506DEDE979FDC76CDDCD8ED235EE59DFFE34` | Fixed-router diagnosis and matched wrong-core control |
| 3 | `docs/papers/shared/results/paper3_kv_materialization/toy_materialization/toy_attention_decomposition.csv` | `5B0DB78262EB2C1EB06ABF2FBA4B9200F0D0122FC8E0FC65811FBEE431B5A29E` | Example/head-level evidence, distractor, surrounding, and native masses |
| 3 | `docs/papers/shared/results/paper3_kv_materialization/pretrained_confirmation/pretrained_confirmation_heldout_aggregate.csv` | `93D8F4677D874CFC30B4D8856D8949690FAA5EAB80A02DECCECA9663F4346603` | Held-out Qwen K/V, likelihood, answer, and attention aggregates |
| 3.5 | `docs/papers/shared/results/paper3_5_adaptive_pra/adaptive_effort_profiles.json` | `BC1392B7E6E26253687AD750CF696B318AF79D6D218A039FAB615729955E61C1` | E0/E1/E2 action definitions and costs |
| 3.5 | `docs/papers/shared/results/paper3_5_adaptive_pra/router_oracle_targets.csv` | `29509D236E2078C33C0507943A8465FFF13CDEEFA8603539FFD567238BEA1CD6` | Measured minimum-profile labels |

## Recommended Claim Language

Use:

> In frozen pretrained transformers, selective materialization currently appears primarily
> as an efficiency mechanism: less K/V can preserve measured quality. In the controlled
> model, a smaller, correct evidence core can improve the target computation by excluding
> competing distractor states. This motivates resource-bounded inference over a sufficiently
> informative working set.

Avoid:

- "Less is more" without the correct-core and wrong-core qualification.
- "Attention dilution is the cause" outside the fixed-routing controlled intervention.
- "The largest configuration is the quality upper bound."
- Claims that PRA reproduces human or neural bounded rationality.

No new experiment is required to support the revised bounded claims. A future end-to-end
mediation study would be needed to assign the Paper 2.5 output gap uniquely to attention
dilution.
