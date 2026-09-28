# SGLang-MLX admission for Pi N=6 request 194 on M5 16 GB

The exact frozen request was admitted by the tokenizer and model-position
checks, but failed closed during the full lifecycle execution on the 16 GB M5
host.  After approximately 34 minutes Metal terminated a command buffer with
`kIOGPUCommandBufferCallbackErrorOutOfMemory`.  The host had accumulated about
5.6 GiB of swap during the run.  No lifecycle receipt was emitted.

This is a capacity boundary, not a numerical or lifecycle qualification.  The
raw `execution.log` is preserved with SHA-256
`938e578987c23755c417947ed7c14f1dcbf9704d4bbd021a37c08c72c4317e26`.
`admission_summary.json` records the frozen geometry and the exact claim
boundary.  No N=6 SGLang-MLX parity, zero-copy, lifecycle, autonomous-agent, or
latency claim is made from this attempt.
