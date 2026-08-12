# Gemma 3 1B validation status

The integration targets only the official `google/gemma-3-1b-it` repository at model and
tokenizer revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.

The thin adapter is implemented and tested offline with Transformers' real
`Gemma3ForCausalLM` classes. Those tests cover exact disabled logits, hidden states, greedy
generation and hybrid-cache tensors; four-query/one-KV-head MQA; Gemma Q/K normalization and
native RoPE; unchanged local sliding layers; global-only PRA placement; CPU post-RoPE detail
K/V; explicit references; the public API; and bounded `#__head` rollover.

The official checkpoint preflight currently returns HTTP 403 even though the local Hugging Face
client is authenticated. The account must separately accept Google's Gemma usage terms and be
granted repository access. `access_status.json` records that external block. No model substitute
was downloaded, and therefore no Gemma routing, systems, memory-use, or quality metric is reported
yet.

After access is granted, run:

```powershell
python -m experiments.paper2_hf.gemma.run_gemma3_1b
```
