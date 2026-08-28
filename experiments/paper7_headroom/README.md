# Paper 7 Headroom cross-evaluation

This add-on distinguishes the historical in-house `CCR_STYLE` baseline from
the released `headroomlabs-ai/headroom` implementation. It does not change the
frozen PRA router, controller, typed-record runtime, or size gate.

The release evaluated here is `headroom-ai==0.37.0`, repository commit
`32d7ca4577d599b8a5f811ada74cf31504302c9d`. The official package runs in an
isolated Python 3.10 environment. No external provider credential is required:
the matched controller is the existing local `qwen3:0.6b` Ollama model.

```powershell
./experiments/paper7_headroom/install_headroom.ps1
$env:PYTHONPATH = "src;."
python experiments/paper7_headroom/run_headroom_cross_eval.py
python experiments/paper7_headroom/run_pra_on_headroom.py --device cuda --refresh
python experiments/paper7_headroom/summarize_cross_eval.py
```

`run_headroom_cross_eval.py` invokes the official worker through the isolated
interpreter. The worker uses Headroom's released `ContentRouter`,
`SmartCrusher`, CCR store, marker format, and retrieval-tool schema. The default
profile leaves the official component defaults unchanged. Candidate tuning
changes only `smart_crusher_max_items_after_crush` and is selected on Paper 7's
validation partition before held-out scoring.

`run_pra_on_headroom.py` reuses Headroom's official built-in tool-output and CCR
needle cases plus its HotpotQA and MS MARCO loaders. PRA maps each context to one
retrieval-only native reference, ranks the frozen 32-token/8-overlap chunks,
and exposes the top-four chunks under the existing 256-token ceiling.

Headroom's eager MS MARCO loader raised `DatasetGenerationError` with the
released dependency set. The exporter records that failure and reconstructs
the released adapter over the same Hugging Face validation split in streaming
mode. MS MARCO routing uses the adapter's selected relevant passage as the
evidence target; the answer remains metadata. HotpotQA results include only
cases whose answer occurs verbatim in backing text. The artifacts record these
eligibility decisions instead of counting non-extractive answers as retrieval
failures.

The primary endpoint is exact evidence visibility. This is the same
mechanism-level endpoint as the frozen Paper 7 quality study, not a claim about
free-form answer equivalence. `HEADROOM_OFFICIAL_*` denotes released Headroom
components. `CCR_STYLE` always denotes the older in-house reproduction.
The isolated official environment exercised structural routing, SmartCrusher,
the CCR store, markers, and retrieval schema. Its optional Kompress ONNX model
was unavailable, so plain-text HotpotQA/MS MARCO rows are preservation controls,
not evidence of learned compression. The worker resets the process-local CCR
store before every case.
