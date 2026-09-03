# Registry Resources

## Core records

| Resource | Mutable surface | Immutable identity/provenance |
| --- | --- | --- |
| Model | Descriptive metadata and lifecycle | Provider, repository, revision, fingerprint |
| PRA bundle | Qualification summary and lifecycle | Bundle revision, base model revision, checksums |
| Artifact source | Mirrors and metadata | Locator revision/digest |
| Profile | Draft policy before approval | Version and immutable revision |
| Engine compatibility | New evidence records may be added | Engine/model/bundle/mode evidence |
| Qualification | Annotations only in future schema versions | Measurements and provenance are append-only |
| Deployment | Desired configuration | Every change increments `desired_revision` |
| Policy | Draft payload before approval | Approved versions are immutable |
| Approval | Never edited | Append-only decision and actor |
| Audit event | Never edited | Append-only before/after summary and trace ID |
| Managed instance | Observed state, heartbeat, URLs, labels | Stable instance ID and authenticated registration subject |
| Router instance | Health, observed revision, features | Router kind, stable ID, management secret reference |
| Route | Enablement and pool membership | Stable public model and explicit LLM/MCP/A2A kind |
| Model pool | Selectors and policy metadata | Model identity and optional immutable revision |
| Backend endpoint | Health, maintenance, weight, labels | Engine/model/bundle identity and inference URL |
| Routing policy | Deterministic constraints and preferences | Stable policy identity |
| Route binding | Enablement and priority | Router-to-route assignment |

Approval states are `DRAFT`, `CANDIDATE`, `APPROVED`, `DEPRECATED`, and
`REVOKED`. Deprecation preserves provenance and existing references.

## Artifact sources

The built-in connectors read Hugging Face public/private metadata and local
bundle manifests. The connector protocol admits S3, OCI, Artifactory, MLflow,
Ollama, and private enterprise stores without making the Registry a blob store.

```json
{
  "source_type": "huggingface",
  "locator": "EInnovator/pra-qwen3-4b-mlx-4bit",
  "immutable_revision": "49c1867...",
  "credential_reference": "vault/pra/huggingface"
}
```

`credential_reference` names an external secret. A URL, bearer token, API key,
or other embedded secret is rejected.

## Managed instances

`ENGINE` and `GATEWAY` instances are operational records, separate from
desired deployments. Their status is computed as `ONLINE`, `DEGRADED`, or
`OFFLINE`. Registration is idempotent for a stable ID; attempts to reuse that
ID for an incompatible type, name, or engine kind return a conflict and are
audited.

Routing resources form control-plane desired state. `GET /v1/routers/{id}/desired`
resolves qualification-aware eligible endpoints and records rejected endpoints
with reasons. This endpoint does not choose a request-time replica.
