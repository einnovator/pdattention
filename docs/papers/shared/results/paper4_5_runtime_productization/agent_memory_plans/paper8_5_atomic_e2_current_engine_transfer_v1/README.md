# Current Atomic-E2 engine transfer

This bundle records post-fix native-engine checks for the frozen Paper 8.5
Atomic-E2 Scikit task cluster.  The logical selector is not rerun.  Inputs are
the hash-bound repeat fixtures in sibling bundle
`paper8_5_atomic_e2_scikit14496_v2`.

## vLLM-Metal scheduler-pool restore

The bridge revision `0cb0fc67` removes the detached-cache expansion used by
the older receipt.  Initial selections alias scheduler-owned prefix pages.
After an explicit offload, restoration allocates pages from vLLM's existing
block pool and copies the canonical K/V once into those pages.  It does not
allocate a second full Metal cache and does not re-encode selected text.

| Frozen request | Qualified | Source + suffix | Selected K/V + suffix | Visible omission | Page rounding | Initial re-encode / K/V copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repeat 1, request 8 | yes | 28,496 + 313 | 12,656 + 313 | 54.983% | +22 tokens | 0 / 0 |
| repeat 2, request 9 | yes | 28,320 + 306 | 12,480 + 306 | 55.334% | +22 tokens | 0 / 0 |
| aggregate final requests | 2/2 | 57,435 total | 25,755 total | 55.158% | +44 tokens | 0 / 0 |

Both receipts use the official vLLM 0.29.0 CPU and vLLM-Metal 0.29.0 wheels,
the pinned vLLM-Metal source revision
`7390805822b2d7a208b09d55bd07b7572f727e20`, and Qwen3-0.6B-4bit on the
16-GB M5 host.  Every declared correctness and lifecycle check passes:
same-subset and ordinary full-prefix token equality, original positions,
concurrent borrower isolation, cancellation, injected-error cleanup, stale
generation rejection, offload, exact restore, repeated offload, and session
termination.  Restore is intentionally accounted separately: the two
lossless offload payloads total 6,516,156,724 bytes.  These are mechanism and
residency-boundary measurements, not latency results.

## SGLang-MLX current lifecycle repeat

SGLang-MLX revision `ef20fab38a03490e2cdf1b7377145ca3a3f2bfc5`
repeats the same two frozen final requests under Paper 4.5 revision
`48254442`.  The environment uses MLX 0.32.1, MLX-LM 0.32.0, and the pinned
Transformers 5.12.1 contract on the 16-GB M5 host.

| Frozen request | Qualified | Source + suffix | Selected K/V + suffix | Visible omission | Re-encode / selection pack / initial K/V copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat 1, request 8 | yes | 28,503 + 306 | 12,641 + 306 | 55.059% | 0 / 0 / 0 |
| repeat 2, request 9 | yes | 28,321 + 305 | 12,459 + 305 | 55.411% | 0 / 0 / 0 |
| aggregate final requests | 2/2 | 57,435 total | 25,711 total | 55.235% | 0 / 0 / 0 |

Both receipts pass same-subset logits and tokens, the ordinary full-engine
oracle, original-position execution, two-borrower isolation, cancellation,
injected-error cleanup, stale-generation rejection, owner guards, eviction,
offload, exact restore, and termination.  The measured consumer peaks are
155,659 and 156,715 bytes, only 0.301% and 0.307% of one selected layer's K/V
extent.  The two offload payloads total 6,517,074,228 bytes and remain a
separate residency-boundary cost.  Unlike vLLM-Metal, SGLang's interval path
needs no complete-page rounding, so its 25,711 selected visible tokens are the
exact engine realization of this final-request logical ledger.

## Direct MLX 30B numerical boundary

The Qwen3-Coder-30B-A3B-Instruct-4bit request-1 lifecycle receipt is a strict
negative only for the ordinary dense cross-consumer raw-logit gate.  The
zero-copy sparse path is exact against the packed identical-consumer
reference and after restore (zero logit, probability, and total-variation
delta).  Against mlx-lm's ordinary dense consumer it preserves both generated
tokens, but has maximum raw/centered-logit deltas of 2.7578125/2.59375,
maximum probability delta 0.00368075, and maximum total variation 0.00511605.
The result therefore remains unqualified under the predeclared 0.005 raw-logit
gate; short token agreement is not promoted to exact consumer equivalence.
Selected history is not re-encoded or packed, and the measured first-layer
consumer peak is 2,187,275 bytes, 12.30% of one selected layer's K/V extent.

The later bounded-memory distribution runner at revision `3748ed21` releases
packed reference K/V before constructing the dense oracle.  It is a numerical
consumer gate only and cannot substitute for lifecycle qualification.

The corrected revision `dce44d1c` applies that bounded-memory gate to frozen
repeat-1 request 8, the final request used by the vLLM-Metal comparison.  It
selects 12,641 of 28,503 source tokens plus a 306-token suffix (44.941%
realized retention), re-encodes zero selected-history tokens, and packs zero
bytes on the sparse path.  Same-consumer raw, centered, probability, and
total-variation deltas are all zero.  The ordinary dense consumer produces
the same two token IDs but differs by 4.515625 raw logit, 5.640625 centered
logit, 0.00254065 maximum probability, and 0.00257192 total variation, so the
cross-consumer and combined gates remain negative.  Consumer peak delta is
2,187,275 bytes, 8.45% of one selected layer's 25,888,768-byte K/V extent.
This distribution-only artifact makes no cancellation, restore, or eviction
claim; those remain covered by the separate request-1 lifecycle receipt.

## Fixture identity

- Repeat-1 plan: `308d91521fa82c0aedc5d59e40d45086641f30f92fab3e15c395267f15e262aa`
- Repeat-1 replay: `3b52c81a8fa7e52bfc9d1097fbc62e806d3e2541cf853a9c2de0a2512ac1ed6e`
- Repeat-2 plan: `4f931dfd289bd6950e7671e403af6040fc1476170a123cd34dc5ffefeaabf73f`
- Repeat-2 replay: `c0e1efa494a3a140e46d7525fdd3d674a2c5706400f96de70c25c0004409eb6f`

The final-request engine omission is not the Paper 8.5 campaign aggregate.
Paper 8.5 reports 57.03% own logical omission across all requests in this task
cluster.  The engine-portable sequence ledger and each engine's physical
rounding must be reported separately.
