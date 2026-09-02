# Registry and governance

The Registry is the authoritative catalog behind fleet intent. The Control Plane presents records without exposing Registry service credentials to the browser.

## Record families

| Record | Meaning |
| --- | --- |
| Models | Base model identities and immutable revisions |
| PRA bundles | Published adapters, compatibility metadata, and provenance |
| Profiles | Runtime policy and evidence tier |
| Qualifications | Measured engine, model, hardware, and workload evidence |
| Compatibility | Accepted model, bundle, engine, and profile combinations |
| Deployments | Desired rollout state |
| Instances | Engines currently registered with the fleet |
| Policies | Governance and authorization constraints |
| Approvals | Pending and completed human decisions |

## Mutations

Users with an Approver or Administrator role can create or edit supported records. Submit valid JSON and a concise reason. Approval, promotion, deprecation, and revocation remain explicit transitions rather than implicit edits.

See [Audit and alerts](activity.md) for the resulting governance trail.

