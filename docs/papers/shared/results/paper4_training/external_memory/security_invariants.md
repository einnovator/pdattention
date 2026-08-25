# Security invariants

- Credentials are opaque runtime handles and are absent from model/artifact state.
- Unauthorized private cross-user reuse blocked in the mechanism run: `true`.
- Cache reuse revalidates resolver authorization.
- Source versions participate in native and hot cache keys.
- Global cache entries require explicitly public metadata.
- Session teardown removes ephemeral state.

Executable coverage: `tests/test_external_memory_lifecycle.py`.
