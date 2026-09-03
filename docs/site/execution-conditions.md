# PRA Execution Conditions

PRA evidence separates context selection, native representation, serving
lifecycle, and learned adaptation. These are different interventions and must
not share one baseline label.

```text
No PRA
  -> PRA Selected Context
  -> PRA Native Memory
  -> PRA Native Serving
```

A bundle or learned adaptor is an orthogonal choice at each PRA stage.

| Condition ID | Display name | What executes |
| --- | --- | --- |
| `NO_PRA` | No PRA | Ordinary model and engine inference. No PRA routing, references, materialization, native memory, or bundle. |
| `PRA_SELECTED_CONTEXT_NO_ADAPTOR` | Selected Context | PRA selects evidence and renders it as visible context. No learned model-specific adaptor is active. |
| `PRA_NATIVE_MEMORY_NO_ADAPTOR` | Native Memory | The same PRA selection is realized as native K/V. No learned model-specific adaptor is active. |
| `PRA_NATIVE_SERVING_NO_ADAPTOR` | Native Serving | PRA selection and native K/V are owned by the serving scheduler and cache lifecycle. |
| `PRA_SELECTED_CONTEXT_BUNDLE` | Selected Context + Bundle | Selected Context with one exact immutable runtime bundle. |
| `PRA_NATIVE_MEMORY_BUNDLE` | Native Memory + Bundle | Native Memory with one exact immutable runtime bundle. |
| `PRA_NATIVE_SERVING_BUNDLE` | Native Serving + Bundle | Native Serving with one exact immutable runtime bundle. |

## What each delta means

| Delta | Attribution |
| --- | --- |
| `delta_sc_vs_no_pra` | PRA selection and visible-context construction |
| `delta_nm_vs_no_pra` | Combined selection plus native realization; do not call this a native-only effect |
| `delta_ns_vs_no_pra` | Combined selection, native realization, and serving lifecycle |
| `delta_nm_vs_sc` | Incremental effect of realizing the frozen selection as native K/V |
| `delta_ns_vs_nm` | Incremental scheduler and lifecycle effect |
| `delta_*_bundle_vs_*` | Incremental bundle/adaptor effect with execution mode held fixed |

The source and target conditions are part of every delta record. A report only
renders a delta when both conditions contain the metric. Missing conditions stay
explicit as `NEEDS_RUN`, `NO_QUALIFIED_ADAPTER`, `CALIBRATION_PENDING`,
`NOT_MEASURED`, `NOT_APPLICABLE`, or `BLOCKED`.

!!! warning "Selected Context is PRA"
    Selected Context already uses PRA routing. It is not a No-PRA baseline.
    A matched-selection experiment comparing Native Memory with Selected
    Context isolates native realization; it does not measure PRA versus No PRA.

## Evidence identity

A comparable evidence row is keyed by task or dataset, hardware and engine,
exact model revision, precision encoding, profile, and condition. Bundle rows
also require the exact bundle ID and immutable revision. Never compare two
precisions or selectors and describe the result as a representation-only delta.

Use `pra report CANONICAL_EVIDENCE.json --format html` to render canonical
records and run the repository audit before publishing generated evidence:

```powershell
python -m experiments.paper4_5_runtime.audit_evidence_conditions --strict
```
