# RoPE Distance Study: Preregistered Expectations

These expectations were recorded before examining the new task-level runs.

| Hypothesis | Expected signature |
|---|---|
| Tiny-fragment contextualization | Larger encoding blocks reduce or erase the positive exact-minus-local loss gap. |
| Larger retrieved unit | Larger materialized evidence chunks improve task loss under fixed encoding context. |
| Distance mismatch | Exact-distance regression persists after larger context, while one or more fixed/clipped distances improve loss. |
| Interaction | Larger context and a nearer effective distance both help. |
| No simple explanation | Policy effects vary in direction across domains, tiers, or seeds without a stable regime. |

The fixed-distance sweep uses oracle reference sets first. Model weights, routing
queries, raw K, V, local chunk spacing, and answer targets remain fixed within a
comparison. Distance uses the nearest retrieved-token convention.

## Frozen Result

The five-seed CUDA sweep covers tiny RoPE checkpoints on HotpotQA- and
QASPER-derived probes. Exact nearest-token distances average 87.3 and 90.4.
The pooled near-range candidate, `D=64`, remains close to exact loss, but remote
placements are unstable: `D=1024` raises mean loss from 0.0655 to 1.1855 on
HotpotQA and from 0.0377 to 0.8192 on QASPER. Top-attended-token agreement with
exact falls to 0.090 and 0.067.

This establishes that retrieval-time RoPE placement matters, but it does not
identify a stable beneficial fixed distance. The response is non-monotonic, as
expected from multi-frequency rotary phase rather than scalar distance decay.

`rope_d_sweep_publication.csv` is the seed-balanced manuscript reduction. The
raw CSV/JSON remain the per-example audit trail.
