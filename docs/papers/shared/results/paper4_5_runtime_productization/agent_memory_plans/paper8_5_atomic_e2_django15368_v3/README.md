# Held-out Atomic-E2 native-K/V handoff

This bundle transfers the held-out `django__django-15368` Atomic-E2 cluster
from Paper 8.5 into Paper 4.5.  The logical cohort was frozen at Paper 8.5
commit `f498e87e`: both FULL controls and both Atomic-E2 treatments resolve,
calls fall from 18 to 16, and the policy omits 58.77% of its candidate-own
history.  Combined with the two earlier task identities, the promoted logical
cohort is 6/6 versus 6/6 resolutions, 59 versus 45 calls, and 50.42%
candidate-own logical saving.  Three task identities are not a population
accuracy estimate.

The files here test the two held-out final requests as immutable native-engine
fixtures.  They do not claim autonomous execution through each engine.  Each
manifest binds the request replay, selected record plan, model parameters, and
source trajectory.  Every engine consumes the same logical record ledger;
vLLM-Metal alone rounds the ledger to complete 16-token pages.

| Engine | Repeat | Source K/V | Selected K/V | Suffix | Visible saving | Re-encoded | Initial K/V copy | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llama.cpp | r01 | 27,796 | 11,934 | 139 | 56.78% | 0 | no | pass |
| llama.cpp | r02 | 27,393 | 11,531 | 138 | 57.62% | 0 | no | pass |
| MLX | r01 | 27,796 | 11,934 | 139 | 56.78% | 0 | no | pass |
| MLX | r02 | 27,393 | 11,531 | 138 | 57.62% | 0 | no | pass |
| SGLang-MLX | r01 | 27,796 | 11,934 | 139 | 56.78% | 0 | no | pass |
| SGLang-MLX | r02 | 27,393 | 11,531 | 138 | 57.62% | 0 | no | pass |
| vLLM-Metal | r01 | 27,792 | 11,952 | 143 | 56.70% | 0 | no | pass |
| vLLM-Metal | r02 | 27,392 | 11,552 | 139 | 57.54% | 0 | no | pass |
| HF/CUDA | r01 | 27,796 | 11,934 | 139 | 56.78% | 0 | no | pass |
| HF/CUDA | r02 | 27,393 | 11,531 | 138 | 57.62% | 0 | no | pass |

All eight Qwen3 cells reproduce the frozen continuation token IDs
`[151667, 198]`; both HF/CUDA cells exactly match their independently
materialized same-subset references. All ten preserve original positions and clear their declared
ownership, concurrent-borrow, cancellation/error, stale-fork, offload/restore,
termination, and cleanup gates.  MLX and SGLang-MLX also match same-subset
reference logits exactly.  Their fused disjoint consumers peak only
156,299--172,049 bytes above baseline, or 0.320--0.364% of one selected
layer's K/V extent.  vLLM-Metal's initial page attachment has zero active-memory
growth; its 3.14--3.19 GB offload/restore materialization is reported
separately and is not described as zero-copy. HF/CUDA uses segmented streaming
attention with no selected-K/V copy or interval pack; its 143.47--144.51 MB
attention temporary is reported separately.

The first vLLM-Metal launch used `gpu-memory-utilization=0.10` and was rejected
before inference because its 24,032-token page capacity was below the declared
32,768-token maximum.  It produced no evidence row.  Both counted runs use the
predeclared admitted configuration (`gpu-memory-utilization=0.20`, 1,800
reserved blocks) and the exact pinned vLLM/vLLM-Metal 0.29.0 package revision.

Claim boundary: this bundle qualifies request-level sparse resident-K/V
mechanics on five engine paths. Autonomous selected-policy execution across
engines remains open.
