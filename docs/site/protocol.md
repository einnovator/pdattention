# Agent, Gateway, and Engine Protocol

PRA extends OpenAI-compatible chat requests without replacing their message
protocol. The conversation remains an ordinary sequence of system, user,
assistant, and provider-required tool messages. Large or reusable context is a
separate stream of typed logical resources.

```text
AgentTurnContext
  messages -------------------------- OpenAI conversation spine
  records/tools/skills/tasks -------- detached PRA resources
                         |
                         v
                  capability negotiation
                 /                      \
          ordinary endpoint        PRA-aware endpoint
          selected text          typed full/delta records
```

## Capability handshake

At launch or reconnect, a client requests:

```http
GET /v1/pra/capabilities
```

The response separates gateway features, engine features, and effective
end-to-end features. Clients check capabilities such as `logical_refs`,
`typed_records`, `task_metadata`, `resource_delta`, `session_state`, and
`incremental_messages`; they do not branch on engine names.

A reachable endpoint returning an unsupported-route status can be treated as an
ordinary OpenAI endpoint. An unreachable endpoint is an error, not permission to
silently downgrade.

## Transport policy

Agent profiles choose `auto`, `pra`, or `text` transport:

```yaml
context:
  transport: auto
  allow_text_fallback: true
  require:
    - logical_refs
    - typed_records
```

Resolved behavior is one of:

| Public behavior | Messages | Detached resources |
| --- | --- | --- |
| Pass through | Full OpenAI messages | None |
| Selected Context | Full OpenAI messages | Deterministically rendered selected records |
| Typed full transport | Full OpenAI messages | Full logical resource inventory |
| Typed delta transport | Incremental messages | Add, update, remove, or unchanged operations |

Native K/V is not a transport mode. Raw tensors and physical page identifiers
never cross this wire.

## Request envelope

PRA-aware requests retain ordinary top-level chat fields:

```json
{
  "model": "model-id",
  "messages": [{"role": "user", "content": "Continue the analysis"}],
  "stream": false,
  "pra": {
    "protocol_version": "1",
    "tenant_id": "tenant-a",
    "session_id": "session-a",
    "task_id": "task-1",
    "resources": [],
    "resource_ops": [],
    "budget": {"max_resources": 8, "max_selected_tokens": 2048},
    "pra_policy": {},
    "metadata": {
      "retention_policy": {
        "recent_completed_turns": 3,
        "recent_records_per_turn": 2,
        "recent_source_turns": 1,
        "recent_mutation_turns": 1,
        "recent_verification_turns": 1,
        "large_record_chunk_tokens": 2048,
        "max_records_per_turn_before_chunking": 8,
        "causal_bundle_round_up": true,
        "preserve_action_observation_pairs": true
      }
    }
  }
}
```

Ordinary endpoints receive no `pra` field. Credentials stay in HTTP headers or
local provider configuration and are prohibited from resource, provenance, and
trace metadata.

## Agent-history retention policy

An agent may place request-scoped history constraints under
`pra.metadata.retention_policy`. These are semantic minimums, not instructions
to flatten the transcript into fixed token windows:

| Field | Meaning |
| --- | --- |
| `recent_completed_turns` | Keep at least the newest `N` completed turns in full. A completed turn includes the assistant action and its resulting tool or user observation. |
| `recent_records_per_turn` | Within an earlier or oversized turn, preserve at least the newest `M` logical records. Action/observation pairing can increase the physical record count. |
| `recent_source_turns` | Keep the newest `N` completed turns that established source-localization evidence, even after they leave the recency window. |
| `recent_mutation_turns` | Keep the newest `N` completed code-mutation turns. |
| `recent_verification_turns` | Keep the newest `N` completed verification or submission turns. |
| `large_record_chunk_tokens` | Subdivide an individual oversized record near this token size. Split tool output at natural boundaries such as files, test cases, stack frames, diff hunks, or lines before falling back to token boundaries. |
| `max_records_per_turn_before_chunking` | Permit record-aligned subdivision inside a turn once its record count exceeds this threshold, including inside the otherwise protected `M`-record tail. The logical records and causal group remain identifiable. |
| `causal_bundle_round_up` | Treat the requested token retention as a floor and retain the whole causal bundle that crosses it. |
| `preserve_action_observation_pairs` | Never retain an assistant tool action without its corresponding observation, or an observation without its action. |

Chunking a large record does not erase its identity. Every child span carries a
stable parent record ID and causal-group ID, overlap never crosses a record
boundary, and overlap is deduplicated before budget accounting. Short records
remain intact. The current incomplete turn and other mandatory task state remain
outside ordinary retrieval competition.

The full-turn guarantee applies directly while a turn contains no oversized
record and does not exceed `max_records_per_turn_before_chunking`. An oversized
record is retained as record-local child spans. In a dense turn, the newest
`recent_records_per_turn` records—and any action needed to make their
observations causal—remain protected; only records earlier than that protected
tail enter ordinary selection.

Because full turns and causal pairs are indivisible, realized retention can be
higher than the requested fraction. A trace for a retention-aware request
reports at least:

```json
{
  "target_retention_fraction": 0.90,
  "realized_retention_fraction": 0.914,
  "retention_rounded_up": true
}
```

`retention_rounded_up` means the runtime retained additional tokens to satisfy
the declared causal constraints. It is not selection failure. Evaluation must
use `realized_retention_fraction`, selected unique tokens, and the selected
record identities rather than assuming the target fraction was materialized
exactly.

## Record projection

`context_record_to_wire_resource()` preserves:

- record ID, type, URI, version, and source fingerprint;
- provenance and authorization scope;
- task ID and status;
- available, initial, and selected views;
- tenant/session binding and explicit shareability.

The representations remain distinct:

| Representation | Owner | Purpose |
| --- | --- | --- |
| `ContextRecord` | Agent/runtime | Semantic typed record and named views |
| `PRAWireResource` | Transport | Portable logical network projection |
| `BackingRecord` | Storage | Reconstructible authoritative detail |
| `PRAStorageEntry` | Engine/storage | Model-specific derived native residency |

## Sessions, deltas, and resynchronization

The transport caches acknowledged message and resource inventories per session.
Endpoint changes, reconnects, engine restarts, protocol mismatch, or explicit
refresh invalidate that state. The next request sends full messages and a full
resource inventory before deltas resume.

Resource identity changes when version, source fingerprint, authorization, task
ownership, or shareability changes. Storage deduplication never grants access
across tenants or sessions.

## Responses and streaming

Non-streaming responses remain OpenAI-compatible and may add a `pra` object with
selected resource IDs, materialized token counts, native-memory status, and a
trace ID. Streaming uses ordinary `chat.completion.chunk` content events.
Additive PRA trace events carry no content choices, so conventional clients can
continue reading text deltas.

## Fallback rules

- Fallback must be explicitly permitted by the request or profile.
- Selected Context fallback must preserve identity labels and authorization.
- A request requiring Native Memory fails if the engine cannot provide it.
- Capability loss invalidates cached native receipts.
- The trace records the selected public behavior and any fallback reason.

The legacy research codes that map to these public behaviors are documented
only in [Research / Evidence](research/index.md).
