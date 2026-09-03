# Bifrost

Bifrost remains the request-time OpenAI-compatible gateway. PRA dynamically supplies eligible custom providers and routing rules from Registry state.

The adapter maps:

| PRA | Bifrost |
|---|---|
| Route | Logical model routing rule |
| ModelPool | Provider target set |
| BackendEndpoint | OpenAI-compatible custom provider |
| RoutingPolicy | Weights, CEL rule, and fallback chain |

Stable provider names combine router ID, engine instance, runtime model identity, and a digest. Mutable IP addresses are never the identity.

```yaml
router:
  id: bifrost-eu
  kind: bifrost
  management_url: https://bifrost-admin.internal
  inference_url: https://bifrost.internal
  credential_reference: BIFROST_ADMIN_TOKEN
  region: eu
```

```bash
pra router preview bifrost-eu
pra router reconcile bifrost-eu --confirm
```

Registry and Controller state preserve model/bundle revision, PRA mode/profile, engine, region, and qualification without embedding credentials in Bifrost's strict provider document. Bifrost applies adaptive request-time provider/key selection, retries, and fallbacks. See the official [provider routing](https://docs.getbifrost.ai/providers/provider-routing) and [routing rules](https://docs.getbifrost.ai/providers/routing-rules) documentation.

Registry or controller loss does not alter the last-good Bifrost configuration. A router outage marks only that RouterInstance offline; it does not mutate route intent.
