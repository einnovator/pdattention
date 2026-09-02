# Operations

The management API accepts only bounded operations. Unsupported engine actions
return `501 ACTION_NOT_SUPPORTED`; they are never successful no-ops.

## Safe configuration patch

`PATCH /v1/pra/config` accepts these live-mutable fields:

- `profile`
- `selection_budget`
- `storage_quota`
- `retention_policy`
- `observability`
- `prefetch_policy`

Engine, model, device, topology, listener host, and listener port are
restart-required. They return `409 RESTART_REQUIRED` with the exact field list.

```bash
cat > profile-patch.yaml <<'YAML'
profile: ECONOMY
selection_budget:
  max_selected_tokens: 1024
YAML
pra engine patch-config local-vllm --patch profile-patch.yaml
```

## Action endpoints

| Action | Meaning | Required support |
| --- | --- | --- |
| `prefetch` | Make an authorized resource ready before demand | Storage manager |
| `promote` | Move an authorized resource into HOT native residency | Storage manager |
| `demote` | Move unpinned HOT detail to a cheaper tier | Storage manager |
| `evict` | Remove eligible physical residency | Explicit engine handler |
| `reload-profile` | Activate a safe profile revision | Explicit engine handler |
| `reload-bundle` | Activate a compatible bundle revision | Explicit engine handler |
| `maintenance` | Run one retention/quota maintenance pass | Storage manager or handler |

Actions accept an `idempotency_key`. Repeating the same action/key returns the
original outcome with `idempotent_replay=true` and does not repeat the physical
operation.

```bash
pra engine action promote local-hf \
  --resource-id RESOURCE_ID --tenant-id tenant-a \
  --idempotency-key promote-42
```

Resource authorization scopes come from the authenticated actor. They cannot be
self-asserted in an action body.

Every successful mutation records a bounded local audit event containing time,
actor, request ID, event type, result, and redacted changes. Fleet-wide,
immutable audit remains a control-plane responsibility.
