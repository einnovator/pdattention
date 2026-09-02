# Registry Authentication

Unauthenticated mode is development-only and may bind only to loopback. Shared
deployments should use a static token for simple installations, service
credentials for automation, or OIDC/JWT for identity-provider integration.

| Scope | Permission |
| --- | --- |
| `registry:read` | Read resources, evidence, audit, and resolve results |
| `registry:write` | Create and mutate draft resources or desired state |
| `registry:approve` | Approve, deprecate, or revoke governed resources |
| `registry:admin` | All registry operations |

```yaml
registry:
  host: 0.0.0.0
  auth:
    mode: oidc
    oidc_issuer: https://identity.example.com/
    oidc_audience: pra-registry
    oidc_jwks_url: https://identity.example.com/.well-known/jwks.json
```

For static auth, set `static_token_env: PRA_REGISTRY_TOKEN`; do not put the
token in checked-in YAML. Artifact repository credentials likewise remain in a
secret manager and are addressed only by `credential_reference`.
