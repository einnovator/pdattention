# RepoBench-P

RepoBench-P is the inexpensive repository-level retrieval-plus-completion tier
for debugging context selection before autonomous agent loops. The canonical
implementation and ICLR 2024 artifacts are maintained at
https://github.com/Leolty/repobench; historical paper reproduction must use the
repository's `archive/v0` branch rather than silently mixing it with the live
benchmark.

| Field | Value |
| --- | --- |
| Upstream | `Leolty/repobench` |
| Task | RepoBench-P retrieval followed by code completion |
| Languages | Python and Java |
| Canonical metric | retrieval and completion metrics defined by the pinned upstream release |
| Role here | context-selection microbenchmark, not autonomous repair evidence |

Before execution, a campaign must pin the upstream Git revision, dataset
revision, language split, cross-file setting, task IDs, prompt construction,
retrieval candidate pool, and metric implementation. No aggregate from this
tier may be presented as SWE-bench-style task resolution. Its purpose is to
debug matched truncation versus PRA selection cheaply and to measure physical
tokens, retrieval recall, completion quality, and latency over many examples.
