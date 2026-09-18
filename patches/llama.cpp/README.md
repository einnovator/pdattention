# llama.cpp live agent-history K/V patches

These patches apply in order to upstream llama.cpp revision
`458681e1d5d4a29a1463c4732e03226cf384b997`:

1. `0001-server-pra-live-prefix-kv.patch` adds stable multi-record ranges over an
   already evaluated live source sequence, zero-copy request membership,
   source-position preservation, commit-back, provenance telemetry, and strict
   capability negotiation.
2. `0002-server-pra-support-live-prefix-rollback.patch` adds exact common-prefix
   rollback for parser-rejected agent actions before a replacement suffix is
   attached.
3. `0003-server-pra-pin-restored-agent-slots.patch` restores source ownership when
   a saved live agent slot is reloaded.
4. `0004-server-encode-PRA-receipts-at-original-positions.patch` evaluates compact
   materialized-history receipts once at their original causal positions while
   retaining selected source K/V by zero-copy sequence membership.
5. `0005-server-interleave-PRA-receipts-with-resident-ranges.patch` attaches
   disjoint resident ranges and evaluates old-position receipts in causal order,
   rather than attaching the final resident tail before an earlier receipt.

Apply and build:

```bash
git checkout 458681e1d5d4a29a1463c4732e03226cf384b997
git am /path/to/patches/llama.cpp/0001-*.patch
git am /path/to/patches/llama.cpp/0002-*.patch
git am /path/to/patches/llama.cpp/0003-*.patch
git am /path/to/patches/llama.cpp/0004-*.patch
git am /path/to/patches/llama.cpp/0005-*.patch
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server -j
```

The resulting server must run with `--kv-unified`. It remains intentionally
unqualified for production agent history until the additional lifecycle,
concurrency, cancellation, eviction, and cross-model gates pass.
