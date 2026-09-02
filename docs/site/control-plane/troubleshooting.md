# Control Plane Troubleshooting

## The service refuses to start

Set `PRA_CONTROL_COOKIE_SECRET`. Local-only auth may bind only to loopback. Any
OAuth/OIDC provider also needs the environment variable named by
`client_secret_env`.

## An engine is offline

Call its `/health` and `/v1/pra/info` endpoints from the Control Plane host or
container. Check DNS, firewalls, Management API bearer token scope, and TLS
trust. The token's value belongs only in the named backend environment variable.

## Desired state is unknown

Confirm the Registry URL and token, then inspect deployment environment and
cluster fields. They must match the engine's Control Plane metadata.

## SSO loops or rejects callback

Compare `public_url`, provider callback registration, reverse-proxy scheme, and
host headers. For OAuth, expired transaction cookies and mismatched state are
rejected by design. For SAML, install the optional dependency and validate the
IdP metadata and signed assertion configuration.

## Agent chat reconnects repeatedly

Confirm that the proxy supports WebSocket upgrade and does not strip cookies.
The browser retains the resume token and last event sequence locally; clearing
site data deliberately starts a new conversation.

## A mutation is forbidden

Check the signed-in role, CSRF header, reason, and confirmation requirement.
Review the Audit panel: failed attempts are recorded with their target and
result.
