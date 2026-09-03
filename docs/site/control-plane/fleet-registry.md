# Fleet and Registry Operations

The fleet overview combines configured, manually added, and Registry-
discovered engines. Registry discovery reads live `ManagedInstance` records,
not management URLs embedded in desired deployment selectors. For each engine it reads the open Management API and shows
health, runtime versions, loaded model, bundle, profile, execution mode, key
reuse metrics, and alerts. Fleet search and engine, model, and desired-state
filters live in the central Fleet tab; every column is sortable.

Selecting an engine opens one reusable detail tab, or focuses that tab when it
already exists. It provides Summary, Capabilities, Config, Models, Sessions,
Resources, Storage, Observability, and Audit views. Structured service values
are rendered as labeled fields instead of raw JSON, and contextual information
icons explain settings and metrics. The backend proxies the
Management API with a server-side bearer token. It never sends engine
credentials to JavaScript.

Engine reachability and Registry comparison are separate columns. A reachable
engine can have `UNKNOWN` desired state when no deployment applies; that does
not imply an unhealthy runtime. Desired state comes only from Registry
deployment records and has four outcomes:

| State | Meaning |
| --- | --- |
| `IN_SYNC` | Model, bundle, profile, and mode match Registry intent |
| `DRIFT` | At least one governed field differs |
| `UNKNOWN` | No applicable desired deployment is resolved |
| `OFFLINE` | The engine Management API cannot be inspected |

Operators can prefetch, promote, demote, evict, enter maintenance, and apply
safe mutable configuration. Immutable drift is presented as a restart-required
plan rather than silently invoking an external orchestrator.

Registry views cover models, bundles, profiles, qualifications, compatibility,
deployments, managed instances, policies, approvals, and audit. Approver/Admin transitions always
require a reason and produce a Control Plane audit event in addition to the
Registry's own provenance record.
