# Addendum for AGENTS-pra-routing-gists-recursive

## Optional Summary Support (Authoritative)

### Design principle

Summaries are OPTIONAL.

The architecture must not require summaries, but it must preserve support for
them because they enable important future use cases.

The default architecture remains:

Reference
→ Chunks
→ Chunk routing gist (derived from contextual projected keys)
→ Chunk token K/V

Summaries are an optional metadata source that can participate in routing.

### Configuration

```yaml
reference_routing:
  use_summary: false
  summary_mode: replace
```

`use_summary = false` (default)

- Ignore summaries completely.
- Compute routing gists from chunk content.
- Missing summaries are irrelevant.

`use_summary = true`

- If summary metadata exists, use it according to `summary_mode`.
- If summary metadata is absent, automatically fall back to content-derived routing.
- Never fail because a summary is missing.

### Metadata

Keep summaries only as optional metadata, e.g.

```python
metadata = {
    "summary": "...",
    "summary_model": "...",
    "summary_version": "...",
}
```

Do NOT make `summary` a required cache field again.

### Summary modes

Supported modes:

- replace
    summary-derived routing replaces content-derived routing

- hybrid
    both summary-derived and content-derived routing gists are retained

- augment
    summary contributes an additional routing signal but does not replace the
    content-derived routing gist

### Motivation

This preserves support for:

- teacher-model generated summaries
- neural distillation
- memory compression
- enterprise curated summaries
- offline preprocessing
- experiments comparing learned routing gists against externally generated summaries

### Evaluation

Whenever summaries are enabled compare:

- content routing only
- summary routing
- hybrid routing

under identical retrieval budgets.

Report:

- retrieval quality
- selected memory fraction
- latency
- summary storage overhead
- cache size
