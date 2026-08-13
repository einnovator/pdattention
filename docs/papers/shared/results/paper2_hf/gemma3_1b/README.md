# Gemma 3 1B validation results

The integration targets only the official `google/gemma-3-1b-it` repository at model and
tokenizer revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.

The thin adapter and official checkpoint pass exact disabled logits, hidden states, greedy
generation, and hybrid-cache tensor parity. Gemma's 22 local sliding layers remain unchanged;
layers 5, 11, 17, and 23 are architecturally global-capable, while the measured routed
configuration enables and actually consumes PRA memory only at layers 17 and 23. Captured detail
K/V remains native one-head MQA on CPU, and the bounded `#__head` test records zero native-limit
violations.

The frozen 294,912-parameter router uses 128-dimensional asymmetric projections trained on QASPER
over five seeds. QASPER evidence-identity R@5/10/20/30% is 0.251/0.326/0.432/0.514; HotpotQA
transfer is 0.073/0.133/0.253/0.368. Both domains require exhaustive selection for 0.80 recall.
In the four-example API smoke, the compact index is 0.783% of resident two-layer detail K/V and
6.48% of source K/V tokens are materialized. These are mechanism and systems checks, not benchmark
accuracy estimates.

On eight causal-probe examples, routed memory changes mean gold-token log-probability by +0.135
nats with unchanged aggregate F1. Forced final-global-layer memory changes it by -0.016; direct
evidence text changes it by +3.765. Gemma therefore consumes external native K/V, but this frozen
path is not calibrated like direct context.

Reproduce the complete suite with an authenticated account that has accepted the Gemma terms:

```powershell
python -m experiments.paper2_hf.gemma.run_gemma3_1b
python -m experiments.paper2_hf.gemma.summarize_gemma3_1b
```

To verify the packaged surface in an isolated environment from the repository root:

```powershell
python -m pip wheel . --no-deps --wheel-dir dist
python -m venv .venv-gemma-clean
.\.venv-gemma-clean\Scripts\python -m pip install dist\pra_hf-0.2.0rc1-py3-none-any.whl
.\.venv-gemma-clean\Scripts\pra-hf --help
```

The base checkpoint remains subject to Google's Gemma license; router weights do not redistribute
base-model parameters.
