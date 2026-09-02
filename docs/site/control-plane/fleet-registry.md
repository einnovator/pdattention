# Fleet and Registry Operations

The fleet overview combines configured, manually registered, and Registry-
discovered engines. For each engine it reads the open Management API and shows
health, runtime versions, loaded model, bundle, profile, execution mode, key
reuse metrics, and alerts.

Selecting an engine opens Summary, Capabilities, Config, Models, Sessions,
Resources, Storage, Observability, and Audit views. The backend proxies the
Management API with a server-side bearer token. It never sends engine
credentials to JavaScript.

Desired state comes only from Registry deployment records. The comparison has
four outcomes:

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
deployments, policies, approvals, and audit. Approver/Admin transitions always
require a reason and produce a Control Plane audit event in addition to the
Registry's own provenance record.
