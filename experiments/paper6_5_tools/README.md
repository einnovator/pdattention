# Paper 6.5: Persistent Agent Context

This experiment studies typed tool and skill discovery before native-K/V
materialization. It follows the mechanism-first order in the Paper 6.5 v4
specification.

## Inherited boundary

The branch inherits:

- versioned PRA cache entries and recursive reference resolution;
- bounded long-prompt rollover through the implicit `#__head` reference;
- Qwen, Llama, and Gemma Hugging Face adapters;
- Paper 2.6 token-native and semantic hybrid scoring;
- routing confidence diagnostics and native-K/V materialization budgets.

The branch does not inherit:

- an OpenAI-compatible HTTP serving endpoint;
- a typed agent-resource registry;
- request/namespace/collection discovery-policy hints;
- persistent postings or BM25 catalog indexes;
- confidence-triggered policy escalation and abstention;
- tool/skill catalog, persistent-session, or session-tree benchmarks.

Paper 6.5 adds the missing resource and policy layer without changing the
native-K/V transport. Discovery produces stable resource identities and an
auditable path; selected identities can later resolve to existing PRA cache
entries.

## Completed M0 study

The deterministic study is reproducible with:

```powershell
python experiments/paper6_5_tools/run_m0_policy_study.py
python experiments/paper6_5_tools/summarize_m0_policy_study.py
```

It evaluates 8--8,192 resource catalogs over five seeds and writes raw traces,
seed summaries, index costs, findings, and figures to
`docs/papers/shared/results/paper6_5_tools`. The checked-in run contains 31,680
policy traces. M0 measures typed discovery and selected-definition accounting;
it does not run a language model, materialize native K/V, or execute tools.

## Completed M1 causal toy gate

M1 trains a 506,400-parameter decoder on an opaque tool-use language. The host
discovers a stable resource URI and binds it to a temporary slot; the model must
use the selected definition's hidden schema to construct the exact call and
continue after a deterministic observation. This deliberately separates
selection, model-side call construction, and host-side execution.

Reproduce the checked-in five-seed CUDA run and summaries with:

```powershell
$env:PYTHONPATH='src;.'
python -m experiments.paper6_5_tools.run_m1_toy_model `
  --seeds 11,23,37,53,71 `
  --catalog-sizes 8,32,128 `
  --steps 3000 `
  --examples-per-size 8 `
  --device cuda `
  --output-dir docs/papers/shared/results/paper6_5_tools/m1
python -m experiments.paper6_5_tools.summarize_m1_toy_model
```

Discovered and oracle native-K/V conditions obtain end-to-end success of
1.000, .950, and .925 at 8, 32, and 128 resources. Shuffled and disabled memory
obtain zero at every size. The eager catalog fits the 512-token native window
only at size 8, where it obtains .025. These are causal mechanism results for a
synthetic grammar and idealized semantic discovery, not pretrained-agent or
real-tool evidence.

## Stages

1. `M0`: deterministic catalog generation and policy/index evaluation
   (complete).
2. `M1`: causal toy decoder with opaque tool identities (complete).
3. `M2`: frozen Qwen3-0.6B selected-schema bridge (complete).
4. `M3`: host-authorized pure execution and typed observations (complete).
5. `M4`: three-to-five-step reactive discovery (complete).
6. `M5`: graph-based speculative capability disclosure (complete; negative
   task-success gate).
7. `M6`: optional co-located/server-native QK discovery on canonical queries
   (complete; negative comparison against lexical/index discovery).
8. `M6.5`: model-independent dictionary, tags, and compact embeddings on a
   frozen H0--H5 semantic-hard benchmark (complete).
9. `M7`: frozen native-Q/K rerun on the identical semantic-hard test identities
   (complete; tool-specific training gate closed).
10. `M8`: automatic Python tool records, bounded candidate palettes, and
    authoritative record-aware materialization (complete).
11. Hierarchical skills, persistent sessions, mutation, and inheritance.
12. OpenAI-compatible serving, cross-model replication, and maintained harness
   integration.

Each stage is gated. A later model or execution stage is not evidence for an
earlier discovery claim, and no side-effecting tool executes solely from a
retrieval confidence score.

## Completed M2--M4 pretrained bridge

```powershell
$env:PYTHONPATH='src;.'
python -m experiments.paper6_5_tools.run_m2_m4_pretrained `
  --seeds 11,23,37,53,71 --device cuda
```

The frozen Qwen3-0.6B run contains 120 M2 call rows, 20 M3 execution rows, and
155 M4 step/summary rows. Selected and oracle schemas produce 20/20 exact
calls; shuffled, irrelevant, and empty controls produce none. Eager disclosure
produces 17/20. All selected M3 calls execute in the pure in-memory harness and
become typed observations, with 17/20 fully grounded continuations. Reactive
JIT completes 10/15 workflows, static required-tool disclosure 3/15, and no
refresh 0/15. The five seeds vary deterministic prompt presentation and tool
order; model weights remain identical.

## Completed M5 disclosure gate

```powershell
python -m experiments.paper6_5_tools.run_m5_disclosure --device cuda
```

The P0--P9 set study separates discovery from disclosure and records every
graph edge used. The combined graph recovers .933 of required tools on average
without destructive-tool exposure, but static combined and matched speculative
disclosure both obtain 0/15 model task success. The stop gate is negative;
reactive JIT remains the supported policy.

## Completed M6 native-discovery gate

```powershell
python -m experiments.paper6_5_tools.run_m6_native_discovery --device cuda
python -m experiments.paper6_5_tools.summarize_m2_m6
```

M6 compares external token/index controls, signed hashing, input embeddings,
native mean/full QK, zero-shot Paper-2.8 rank-16 and rank-8/eight-centroid
indexes, and lexical/native fusion over 18 structured tool definitions. Indexed
lexical routing obtains 1.000 single-step Top-1; token routing obtains .889
multi-step required-tool recall. Native and transferred low-rank modes are
worse. Only the co-located mode is executed; shared-memory projected-query and
identity-only model-server interfaces are implemented as architectural
boundaries. Raw Q/K is not persisted.

## Completed M6.5 semantic-hard external discovery

```powershell
python experiments/paper6_5_tools/run_m6_5_external_semantics.py --device cuda
```

The frozen authored benchmark contains 306 rows over 18 tools: 18 audit, 144
validation, and 144 test queries spanning canonical, synonymous, colloquial,
implicit-goal, contextual, and Portuguese/Spanish/French requests. Compact
encoder and representation selection, fusion weights, calibration, and staged
thresholds use validation only. On test, BM25 obtains .347 Top-1, typed tags
.743, and the dictionary/embedding hybrid .729. The staged resolver selects on
.597 of requests at .907 selective accuracy and asks on the remainder. This is
a controlled concept-inventory result, not a natural multilingual benchmark.

## Completed M7 frozen native rerun

```powershell
python experiments/paper6_5_tools/run_m7_semantic_native.py --device cuda
python experiments/paper6_5_tools/summarize_semantic_hard.py
```

M7 reuses exactly the 144 frozen test identities. Native mean K, full token QK,
and zero-shot Paper-2.8 rank-16 and rank-8/eight-centroid projections obtain
.049--.063 Top-1, around the 1/18 catalog chance rate, versus .743 for tags.
The required native diagnostic is absent, so tool-specific native-Q/K training
does not open. Generated paired effects, calibration tables, and figures live
under `docs/papers/shared/results/paper6_5_tools/semantic_summary`.

## M8 automatic records and bounded candidate palettes

M8 converts 18 ordinary annotated Python callables into provider-independent
tool records without executing them and without manual PRA tags. It then
evaluates automatic metadata and four candidate-set strategies on the same
frozen semantic-hard identities. Candidate selection becomes an authoritative
typed catalog slice: PRA materializes each selected tool definition fully or
returns an explicit budget outcome; it never silently reranks schema fragments.
E5 includes a content-only internal-chunk rerouter as an explicit negative
control for that default.

Reproduce E1--E3 and prepare the compact-encoder candidate sets with:

```powershell
$env:PYTHONPATH='src;.'
python -m experiments.paper6_5_tools.run_auto_union_records --device cuda
python -m experiments.paper6_5_tools.run_union_jit_records `
  --device cuda --prepare-only
```

The second command runs separately so the compact encoder is released before
the generation model is loaded. The checked-in E4--E6 matrix used the official
Ollama `qwen3:0.6b` Q4_K_M artifact on CPU, with deterministic host validation
and model unloading after every call. Run it and regenerate figures/macros with:

```powershell
python -m experiments.paper6_5_tools.run_union_jit_ollama `
  --model qwen3:0.6b --seeds 11,23,37,53,71
python -m experiments.paper6_5_tools.summarize_auto_union_records
```

The runner checkpoints every completed condition and resumes automatically;
pass `--fresh` to discard an incomplete checkpoint. Q4 E4--E6 results are kept
separate from the exact-FP16 M2--M7 evidence. The low-memory backend was required
because the GTX 950M lacked forward-pass workspace and the host could not sustain
the exact-FP16 CPU allocation. Model weights remain frozen and third-party files
are not committed.

## M8 automatic-discovery source ablations

The follow-up A0--A7 experiment freezes one callable-derived `ToolRecord` per
tool and isolates raw text, weighted keywords, a global synonym dictionary,
inferred concepts, automatic tags, compact embeddings, fusion, and bounded
channel union. It also measures source removal, type/schema contribution,
docstring and function-name quality, hardness strata, gain retention, and
per-channel complementarity. Reproduce the CUDA run and all generated paper
inputs with:

```powershell
$env:PYTHONPATH='src;.'
python -m experiments.paper6_5_tools.run_auto_discovery_ablation --device cuda
python -m experiments.paper6_5_tools.summarize_auto_discovery_ablation
```

Outputs live under
`docs/papers/shared/results/paper6_5_tools/auto_discovery_ablation`. The
validation-frozen automatic hybrid reaches 0.812 Top-1 and 0.938 Recall@3 on
the 144 frozen test requests, versus 0.806 and 0.910 for the manual-rich
control. This near-parity is catalog-specific. The automatic-union JIT gate is
closed because union does not improve matched-budget validation recall without
greater unsafe exposure; `union_jit_ablation.csv` records the stopped run.
