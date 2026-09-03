# Metrics & Qualification

PRA changes selection, representation, and serving. Measure each transition
separately so a retrieval improvement is not mistaken for a transport gain.

## Canonical product comparison

Cards, qualification reports, agent benchmarks, and Control Plane evidence use
one public condition grammar:

| Condition | Meaning |
| --- | --- |
| **No PRA** | Original model and engine; no PRA routing, materialization, or bundle |
| **PRA - No Adaptor** | PRA with generic routing and required structural compatibility, without learned model-specific behavior |
| **PRA - Adaptor Bundle** | PRA using the exact immutable Runtime Bundle; a bundle may contain structural metadata without a learned adaptor |

For each metric, both PRA deltas are computed against No PRA. A secondary
incremental adaptor delta may compare the bundle with PRA - No Adaptor. Signs
are never inverted: lower TTFT produces a negative delta, while higher F1
produces a positive delta. Every metric declares `higher_is_better`,
`lower_is_better`, or `neutral`, plus an explicit unit and aggregation.

Missing values use `NOT_MEASURED`, `NOT_APPLICABLE`, or `BLOCKED`; none is
rendered as numeric zero. Run `pra report CANONICAL_EVIDENCE.json --format html`
to render the same normalized schema used by the bundle cards and Control Plane
serialization.

The execution-level comparisons below remain useful mechanism attribution.
They do not replace the three-condition product comparison.

## Context gain

Compare the complete available source with **the same task using Selected
Context**:

```text
Full Context -> Selected Context
```

Report full and selected input tokens, token reduction, quality delta, TTFT,
throughput, peak memory, and CPU/GPU time where available. Token reduction is an
input measurement. It is not automatically a monetary or infrastructure-cost
reduction.

## Native-memory gain

Freeze the selected record IDs and intervals, then compare their two
representations:

```text
Selected Context -> Native Memory
```

Report reference-encoding cost, attach/materialization cost, reuse count,
active native K/V bytes, transfers, TTFT, ITL, completion time, cache/reload,
peak memory, and successful requests/s. Native Memory is not universally
faster; cold one-shot workloads can favor Selected Context.

## Native-serving gain

Keep selection and native representation fixed, then compare application-owned
execution with scheduler-owned lifecycle:

```text
Native Memory -> Native Serving
```

Report prefetch readiness, shared residency, eviction and reload, queue delay,
transfer overlap, batch occupancy, useful throughput, and isolation failures.

## Required workload regimes

Run at least:

| Regime | Question |
| --- | --- |
| Cold one-shot | Does native encoding pay for itself without reuse? |
| Warm repeated resource | Does retained native memory reduce repeated work? |
| Multi-query same resource | Can one immutable resource support many selections? |
| Concurrent shared resource | Is physical detail shared safely and efficiently? |

Evidential selection must be computed once per example and reused by every
representation. Otherwise the result confounds routing quality with transport.

## Quality-adjusted economics

Prefer successful requests/s and successful tasks/hour over raw throughput.
Report cost/successful task only when infrastructure cost is actually known.
Always keep quality, input, native-memory, ingestion, serving, and reuse metrics
in separate columns.

## Evidence labels

| Label | Meaning |
| --- | --- |
| Controlled | Mechanism or synthetic check under bounded conditions |
| Model-backed | A real model executed, but the workload may be diagnostic |
| Natural workload | Natural dataset/task examples with explicit sampling |
| Serving | Online or engine-runtime behavior, including queueing where stated |
| Candidate | Implemented but not product-qualified |
| Not measured | Unknown; preserve as `NOT_MEASURED` |
| Not applicable | The required mechanism or seam does not exist |

`NOT_MEASURED` must never render as zero. A missing tail latency, transfer count,
or quality score cannot be used in a ratio or a product recommendation.

## Qualification checklist

1. Pin model, tokenizer, engine, hardware, and profile revisions.
2. Freeze candidate set and selected evidence.
3. Run quality and causal controls before economics.
4. Separate cold ingestion from warm request cost.
5. Verify tenant/session isolation, cleanup, and exactly-once attachment.
6. Report sample count, seeds, uncertainty, and every missing metric.
7. Promote a profile only on held-out workloads matching its intended use.

The [engine matrix](engines/overview.md) applies these labels to current support.
