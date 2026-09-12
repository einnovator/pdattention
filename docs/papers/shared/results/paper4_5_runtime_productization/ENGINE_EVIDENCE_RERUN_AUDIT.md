# Engine Evidence Rerun Audit

Date: 2026-09-12

## Decision

All public engine-dependent measurements produced before the corrected live-K/V engine gates are quarantined and must be rerun. Historical artifacts remain checked in for diagnosis, but they no longer qualify a runtime bundle or support current exactness, latency, throughput, memory, cache-hit, K/V-copy, or byte-savings claims.

Public evidence is current only when its provenance records all of:

- PRA source commit;
- engine-gate commit;
- engine contract `live-kv-original-position-lifecycle-v1`;
- a clean source tree (dirty-tree runs remain prequalification diagnostics);
- exact model revision, precision encoding, engine version, hardware, cohort, and artifact.

The local public catalog and trusted auto-resolution registry now downgrade the
affected identities to `CONTROLLED` and recommend Selected Context until a
current-contract native rerun passes. This prevents stale measurements from
continuing to imply an automatic Native Memory default.

The current inventory contains 17 local bundles. Ten canonical native-runtime records and ten corresponding headline rows are stale. They derive from 24 raw native-runtime artifacts: 21 three-dataset `matched_e0_e2` files and three earlier MLX profile files.

## What remains valid

- Bundle structure, immutable model identity, projection maps, profile declarations, and compatibility manifests.
- Frozen selector output, evidence recall, router comparisons, and selection-only diagnostics when they do not depend on native execution.
- Plain checkpoint-load and bounded generation smoke evidence, provided it is presented only as smoke and its software provenance remains explicit.
- Post-fix engine-gate and agent-history artifacts under `coding_agents/engine_gates` and `coding_agents/engine_agent_gates`; these establish narrow gate outcomes, not replacement workload economics.

## What is quarantined

| Runtime bundle | Engine | Prior source | Required replacement |
| --- | --- | --- | --- |
| `pra-gemma3-1b-mlx-8bit` | MLX-LM | 3 matched-QA files | architecture-qualified same-subset segmented consumer, then matched QA |
| `pra-llama3-2-1b-mlx-8bit` | MLX-LM | 3 matched-QA files | architecture-qualified same-subset segmented consumer, then matched QA |
| `pra-qwen2-5-1-5b-instruct-bnb-8bit` | HF eager/CUDA | 3 matched-QA files | post-fix HF live-K/V runner and matched QA |
| `pra-qwen3-14b-mlx-4bit` | MLX-LM | MLX profile file | post-fix segmented matched QA |
| `pra-qwen3-14b-mlx-8bit` | MLX-LM | 3 matched-QA files | post-fix segmented matched QA |
| `pra-qwen3-32b-mlx-4bit` | MLX-LM | MLX profile file | post-fix segmented matched QA |
| `pra-qwen3-4b-mlx-8bit` | MLX-LM | 3 matched-QA files | post-fix segmented matched QA |
| `pra-qwen3-8b-mlx-4bit` | MLX-LM | MLX profile file | post-fix segmented matched QA |
| `pra-qwen3-8b-mlx-6bit` | MLX-LM | 3 matched-QA files | post-fix segmented matched QA |
| `pra-qwen3-8b-mlx-8bit` | MLX-LM | 3 matched-QA files | post-fix segmented matched QA |

## Cross-paper impact

The shared engine measurements were captured before the corrected engine gates,
so the quarantine also applies wherever those artifacts or equivalent
pre-gate runners are cited. This does not invalidate retrieval, routing, policy,
or record-identity results that do not depend on native execution.

| Surface | Last relevant committed evidence | Disposition |
| --- | --- | --- |
| Paper 6.1 SGLang matched E0/E2 and lifecycle economics | 2026-08-30 | rerun exactness, latency, throughput, copying, temporaries, and lifecycle under the current contract |
| Paper 6.2 MLX matched E0/E2 and lifecycle economics | 2026-08-30 | rerun with live-prefix capture and the architecture-qualified segmented consumer |
| Paper 6 vLLM matched E0/E2 and lifecycle economics | 2026-08-30 | rerun on qualified CUDA pages; the older Metal compatibility implementation is historical only |
| Paper 9 MLX fan-out, 32K residency boundary, and direct lifecycle timings | 2026-09-08 to 2026-09-09 | rerun on the corrected live-K/V path; retain lineage, routing, invalidation, and consumption-factorial results |

Until replacements land, numerical tables may remain in source as explicitly
historical diagnostics, but they must not be propagated to abstracts, bundle
headlines, the static evidence matrix, or current product claims.

## Frozen rerun protocol

For each exact identity and dataset, keep the selected source, question, decoding parameters, and cohort fixed. Compare Selected Context with Native Memory using the identical selected source. The rerun must separately report:

1. output/token-logit parity for the same subset and original positions;
2. selected-history tokens re-encoded per request;
3. selection-materialization K/V-copy bytes;
4. source capture/encode cost;
5. resident resource bytes;
6. total consumer temporary active and peak bytes;
7. cold, warm, repeated-query, and true concurrent serving latency;
8. cancellation, termination, ownership, eviction/offload, restore, and cleanup outcomes.

Logical reuse is not evidence of physical copy avoidance. A field that the engine cannot observe remains `NOT_MEASURED`; it is never inferred from K/V size.

## Execution order

1. Qwen3 MLX segmented one-example smoke and parity on the 48 GB Big Mac. A
   dirty-tree prequalification now passes the same-consumer gate exactly; it
   is not publishable until repeated from a clean commit under controlled host
   load. Artifact: `engine_evidence_rerun/qwen3_8b_mlx_8bit_qasper_n1_prequalification.json`
   (SHA-256 `01e6089dacf50cd211f55c6bf6b9816490a263e2bfb70aabb1697ff9d317a512`).
2. Qwen3 MLX exact identities, smallest to largest, across QASPER, HotpotQA, and 2Wiki.
3. HF bitsandbytes 1.5B on CUDA once the RTX host is reachable; the local small GPU is a fallback only for the bounded smoke.
4. Llama and Gemma MLX only after their architecture-specific segmented consumers pass same-subset and lifecycle gates.
5. Regenerate cards, the static evidence matrix, and Paper 4.5 from current-contract artifacts only.
6. Audit Papers 6.x and 9 for any inherited pre-fix engine economics before retaining those values.

The Medium Mac remains reserved for the active Paper 3.3 and Paper 9 work until those jobs finish; loading this rerun matrix there would create memory pressure and compromise timings.
