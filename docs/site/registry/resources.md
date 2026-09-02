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
