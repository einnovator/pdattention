# Authentication

The management listener never reuses an inference API key implicitly.
Authentication is configured separately and each operation checks a management
scope.

## Modes

| Mode | Intended use | Configuration |
| --- | --- | --- |
| `none` | Local development only | Restricted to loopback binds |
| `static_bearer` | Private machine/service automation | Token from a secret environment variable |
| `jwt_oidc` | Organization identity | OIDC issuer, audience, and JWKS URL; install `pra-hf[management-auth]` |
| `mtls` | Service-to-service deployments | TLS client certificate and optional subject allowlist |

Direct mTLS serving requires `tls_certfile`, `tls_keyfile`, and `tls_ca_certs`.
The listener sets certificate verification to `CERT_REQUIRED`; an optional
`mtls_subjects` allowlist is enforced when the ASGI server exposes certificate
subject details.

Static bearer example:

```bash
export PRA_MANAGEMENT_TOKEN='replace-through-your-secret-manager'
pra engine serve --engine mlx --host 127.0.0.1 --port 9101 \
  --auth-mode static_bearer --token-env PRA_MANAGEMENT_TOKEN
```

The CLI reads the same environment variable without persisting its value:

```bash
pra engine connect http://127.0.0.1:9101 \
  --name local-mlx --token-env PRA_MANAGEMENT_TOKEN
pra engine inspect local-mlx
```

## Scopes

| Scope | Access |
| --- | --- |
| `pra:read` | Health, state, profiles, resources, and telemetry links |
| `pra:configure` | Safe runtime configuration and profile reload |
| `pra:storage` | Storage state and lifecycle actions |
| `pra:sessions` | Session summaries |
| `pra:models` | Model and bundle metadata/actions |
| `pra:admin` | Local audit and all scoped operations |

Use least privilege. Bind production listeners to a private management network,
terminate TLS at the process or a trusted local proxy, and rotate credentials
independently from model-serving keys.
