# MLX 14B short-context consumer control

This bounded control uses the same Qwen3-14B 4-bit checkpoint as the frozen
request-9 gate, but only 154 source tokens. It separates interval addressing
from the attention consumer's numerical reduction.

| consumer | same-consumer packed-reference delta | native dense-oracle delta | generated-token match |
|---|---:|---:|---:|
| fused disjoint Metal | 0.0 | 1.375 | yes, 4/4 |
| eager multi-segment | 0.0 | 1.1611328125 | yes, 4/4 |

Both disjoint consumers exactly match a physically packed copy consumed by
the same implementation. Both differ numerically from MLX native dense
attention, while retaining exact generated tokens. The fused 16-token
diagnostic also preserves every generated token and reports the same `1.375`
maximum dense-oracle delta.

`fused.json` and `eager.json` retain the predeclared `0.01` numerical gate and
are strict negatives. `fused16_diagnostic.json` deliberately raises the bound
to `10` only to observe a longer token trajectory; it is not qualification
evidence and must not be pooled with the strict runs.
