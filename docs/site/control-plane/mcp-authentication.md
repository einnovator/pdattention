# MCP Authentication

MCP identities use the same roles and permissions as REST and the built-in
agent. A named auth profile maps transport credentials to `CallerContext`.

```yaml
control_plane:
  auth_profiles:
    local-agent:
      type: service_identity
      subject: local-codex
      roles: [Viewer]
    remote-agent:
      type: bearer_token
      subject: automation
      roles: [Viewer]
      token_env: PRA_CONTROL_MCP_TOKEN
```

Stdio can use a local service identity. Remote HTTP should use a bearer/OIDC,
client-credential, or mTLS profile. `token_env`, `client_secret_env`, and
`token_file` are references. The Control Plane never serializes resolved secret
values into REST, MCP discovery, logs, or audit payloads.

OIDC profiles require `issuer`, `audience`, and `jwks_url`; signing keys and
issuer/audience claims are verified before a caller is created. For mTLS behind
a reverse proxy, configure `mtls_subject_header` and require the proxy to remove
all client-supplied copies of that header before writing the verified
certificate subject. Do not expose the application listener directly when
using header-forwarded certificate identity.

Mutating tools must be explicitly enabled and the caller must independently
hold the manager permission. Tool visibility is not authorization.
