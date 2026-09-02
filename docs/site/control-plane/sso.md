# Control Plane SSO

The authentication provider interface supports local users, GitHub OAuth,
Google OAuth/OIDC, generic OIDC, and optional SAML 2.0. Browser sessions use
signed, HTTP-only, SameSite cookies. OAuth starts with state, nonce, and PKCE;
provider client secrets stay in backend environment variables.

For GitHub or Google, set `kind`, `client_id`, and `client_secret_env`. Their
standard authorization, token, and user-info endpoints are supplied by default.
Generic OIDC requires explicit endpoint URLs so an operator can review the
trust boundary.

Local accounts name password environment variables rather than embedding
passwords in YAML:

```yaml
auth:
  local_users:
    - username: operator
      password_env: PRA_OPERATOR_PASSWORD
      role: Operator
```

SAML is intentionally dependency-gated:

```bash
pip install -e ".[control-plane,control-plane-saml]"
```

Configure an enabled `kind: saml` provider and its trusted metadata URL. The
adapter runs in strict mode and delegates assertion signature validation to
`python3-saml`. Keep the callback URL and reverse-proxy scheme/host headers
consistent with `public_url`.
