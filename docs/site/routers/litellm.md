# LiteLLM

LiteLLM remains the OpenAI-compatible request data plane. PRA supplies qualification-aware model-group membership and policy; LiteLLM owns provider compatibility, retries, fallback, rate limits, and load balancing.

The adapter compiles each Registry route into `model_list` deployments with:

- the stable PRA public model as `model_name`;
- each endpoint's OpenAI-compatible base URL;
- deterministic deployment identity;
- weight and non-secret PRA identity metadata;
- LiteLLM routing strategy and fallback configuration.

```yaml
router:
  id: litellm-eu
  kind: litellm
  management_url: http://litellm-admin:4000
  inference_url: http://litellm:4000
  credential_reference: LITELLM_ADMIN_TOKEN
```

```bash
litellm --config litellm_config.yaml --port 4000
pra router preview litellm-eu
pra router reconcile litellm-eu --confirm
```

PRA does not place LiteLLM in a Kubernetes-only deployment and does not fork it. The adapter boundary can target a supported management API or a separately implemented persistent/hot-reload transport. See the [official LiteLLM proxy documentation](https://docs.litellm.ai/) for current server options.

If Registry is unavailable, LiteLLM continues from its last applied model list. The Control Plane marks stale desired state but does not stop inference.
