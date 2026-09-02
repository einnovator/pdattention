# Control Plane Configuration

Configuration precedence is explicit CLI options, environment variables, YAML,
then defaults. A `.env` beside the YAML file can provide secret values, but
production secret stores should inject environment variables directly.

```yaml
control_plane:
  host: 0.0.0.0
  port: 9300
  public_url: https://pra-control.example.com
  database_url: postgresql+psycopg://pra:CHANGE_ME@postgres/pra
  auth:
    cookie_secret_env: PRA_CONTROL_COOKIE_SECRET
    cookie_secure: true
    providers:
      - name: company
        kind: oidc
        client_id: pra-control
        client_secret_env: PRA_CONTROL_OIDC_SECRET
        authorization_url: https://id.example.com/oauth2/authorize
        token_url: https://id.example.com/oauth2/token
        userinfo_url: https://id.example.com/oauth2/userinfo
        role_claim: pra_role
        default_role: Viewer
registry:
  url: http://pra-registry:9200
  token_env: PRA_REGISTRY_TOKEN
fleet:
  discovery_mode: combined
  engines:
    - name: vllm-01
      management_url: http://host1:9101
      token_env: VLLM_01_MANAGEMENT_TOKEN
      environment: production
      region: eu-west
      cluster: inference-a
grafana:
  url: https://grafana.example.com
tempo:
  url: https://tempo.example.com
prometheus:
  url: https://prometheus.example.com
```

Supported environment overrides are `PRA_CONTROL_HOST`, `PRA_CONTROL_PORT`,
`PRA_CONTROL_PUBLIC_URL`, `PRA_CONTROL_DATABASE_URL`, and `PRA_REGISTRY_URL`.
When only local authentication is configured, the service refuses a non-loopback
bind address unless `allow_local_auth_non_loopback` is explicitly enabled. Use
that exception only behind a trusted reverse proxy or a host-only published port.

Discovery modes are `static`, `manual`, `registry`, and `combined`. Manual
records belong to the Control Plane; deployment intent remains in the Registry.
