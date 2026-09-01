# llama.cpp native selected-K/V probe

This narrow integration targets llama.cpp commit
`458681e1d5d4a29a1463c4732e03226cf384b997`. It uses the public memory API,
not slot serialization:

1. Encode a selected immutable resource under a dedicated sequence ID.
2. Attach the resource's existing K/V cells to a request sequence with
   `llama_memory_seq_cp`.
3. Decode only the query tokens at prefix-equivalent logical positions.
4. Compare the next-token logits with an ordinary full-prefix decode.

Copy `examples/pra-native` into an upstream checkout and add
`add_subdirectory(pra-native)` to `examples/CMakeLists.txt`. Build the
`llama-pra-native` target and run:

```bash
./build/bin/llama-pra-native -m /path/to/model.gguf -ngl 99
```

The probe explicitly enables `kv_unified`. In that mode sequence copy changes
cache membership metadata and does not duplicate the K/V payload. With
separate cache streams, upstream schedules a physical buffer copy instead.
The unified mode therefore establishes a minimal native selected-K/V seam with
ordinary prefix positional geometry. Resource lifecycle, server request
plumbing, tenant isolation, and multi-resource budgeting remain adapter
responsibilities.
