# vLLM/CUDA admission for Pi N=6 request 194

The exact frozen request fails closed before inference on the RTX 5060 8 GB
host.  Its 77,955-token prompt exceeds Qwen3-0.6B's declared 40,960-position
limit.  Independently, the 77,756-token FP16 source K/V requires
8,917,680,128 bytes (8.305 GiB), already exceeding the GPU's 7.96 GiB before
weights or activation workspace.  The initialized vLLM 0.28 engine reports a
50,400-token / 5.38-GiB K/V capacity.

`admission.log` is the raw fail-closed engine receipt.  The page-rounded N=6
geometry in `admission_summary.json` is a deterministic plan calculation, not
an executed engine result.  No N=6 inference, numerical parity, zero-copy,
lifecycle, or latency claim is made.

