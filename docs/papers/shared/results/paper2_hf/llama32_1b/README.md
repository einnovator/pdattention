# Meta Llama 3.2-1B validation status

Target checkpoint: `meta-llama/Llama-3.2-1B` at
`4e20de362430cd3b72f300e6b0f18e50e7166e08`.

The 2026-08-12 preflight resolved that exact public revision and the `llama3.2`
license tag, then received HTTP 401 while requesting `config.json`. The local
Hugging Face client was not authenticated and the repository requires manually
approved access. No model weights were downloaded, no Meta metrics were measured,
and SmolLM2 results were not reused as Meta results.

After the account has accepted the Meta license and received access:

```powershell
hf auth login
python -m experiments.paper2_hf.llama.run_llama32_1b
```

The command first runs exact disabled-path and native-K/V gates, then continues
through matched feature extraction, five router seeds, the public API systems
demo, and the causal-use probe. See `access_status.json` for the machine-readable
attempt record.
