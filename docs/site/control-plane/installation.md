# Control Plane Installation

Install the optional service dependencies:

```bash
pip install -e ".[control-plane]"
export PRA_CONTROL_COOKIE_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
pra control serve --config control-plane.yaml
```

The standalone equivalent is:

```bash
pra-control serve --config control-plane.yaml
```

The `control-plane` extra includes the official Python MCP SDK. MCP remains
disabled until configured. A local stdio server starts with:

```bash
pra control mcp --config control-plane.yaml --transport stdio
# equivalent
pra-control mcp --config control-plane.yaml --transport stdio
```

The two services deliberately have different executable and image boundaries:

| Service | Standalone CLI | Main PRA alias | Reference image |
| --- | --- | --- | --- |
| Open PRA Registry | `pra-registry serve` | `pra registry serve` | `einnovator/pra-registry` |
| Enterprise Control Plane | `pra-control serve` | `pra control serve` | `einnovator/pra-control` |

The aliases make both services discoverable from the open `pra` command tree;
the standalone entry points let packaging, licensing, images, and deployment
lifecycle remain separate.

Open `http://127.0.0.1:9300/index.html`. Use HTTPS and a reverse proxy for any
non-local deployment. The [Compose deployment](deployment.md) starts the
Control Plane, Registry, PostgreSQL, Grafana, Prometheus, Tempo, and the OTel
Collector.

## First operator check

```bash
curl http://127.0.0.1:9300/health
```

Expected response:

```json
{"status":"ok","protocol":"pra-control/1"}
```

Then verify that each configured engine's Management API is reachable from the
Control Plane container or host. Engine bearer tokens are read from named
environment variables by the backend and are never returned to the browser.
