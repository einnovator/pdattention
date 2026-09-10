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

Apply and build:

```bash
git checkout 458681e1d5d4a29a1463c4732e03226cf384b997
git am /path/to/patches/llama.cpp/0001-*.patch
git am /path/to/patches/llama.cpp/0002-*.patch
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server -j
```

The resulting server must run with `--kv-unified`. It remains intentionally
unqualified for production agent history until the additional lifecycle,
concurrency, cancellation, eviction, and cross-model gates pass.
