# PRA router deployment

Run the small reference router directly:

```bash
pip install -e ".[router]"
pra router serve --config deploy/router/reference-router.yaml
```

Or run its container:

```bash
docker compose -f deploy/router/compose.yaml up --build
```

The OpenAI-compatible endpoint is `http://127.0.0.1:9400/v1`. The management
surface is under `/v1/router/*`; set `PRA_ROUTER_RELOAD_TOKEN` before exposing
it beyond a local lab. The Kubernetes manifest demonstrates the same reference
router, not a replacement for the GAIE/llm-d adapter at fleet scale.

External router adapters consume Registry desired state through:

```bash
pra router preview ROUTER_ID --registry-url http://registry:9200
pra router reconcile ROUTER_ID --registry-url http://registry:9200 --confirm
pra router controller --registry-url http://registry:9200 --interval 10
```

Normal routing continues from each data plane's last-good configuration when
Registry or the controller is unavailable.
