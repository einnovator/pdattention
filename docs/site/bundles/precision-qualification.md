# Precision Qualification

PRA qualification is scoped to the exact model revision, weight encoding,
engine, execution mode, and profile. Evidence from one quantization is not
automatically transferred to another.

The runtime records two related fields:

| Field | Meaning | Example |
| --- | --- | --- |
| `precision_family` | Numerical family used for comparison. | `INT4` |
| `precision_encoding` | Concrete storage/runtime representation. | `MLX-4bit` |

`MLX-4bit`, `AWQ-4bit`, `GPTQ-4bit`, and `GGUF-Q4_K_M` share a broad family,
but they are different evidence identities. Serving precision, feature
extraction precision, and adaptor parameter precision are recorded separately.

## Precision levels

### FP32

FP32 is the high-memory reference. Larger configurations pass a staged memory
gate before evaluation: load, short smoke, 2K context, 8K context, and the
target workload. A failed stage is `BLOCKED_MEMORY`, not a zero score.

### FP16 and BF16

FP16 and BF16 are distinct families. BF16 is the principal non-quantized
qualification target on current Apple Silicon and CUDA hosts. Engine and model
support still determine whether either format can be used.

### INT8

INT8 evidence identifies the implementation, such as `MLX-8bit` or
`bitsandbytes-8bit-LLM.int8`. A successful MLX qualification does not qualify a
bitsandbytes conversion of the same source checkpoint.

### INT4

INT4 is the capacity-oriented level with the broadest current catalog coverage.
Its cards retain the exact conversion identity and immutable model revision.
Current INT4 evidence must not be described as proof of FP32/BF16 robustness.

## Qualification conditions

Each complete row compares the same prompt, candidate set, selected evidence,
dataset cohort, model conversion, engine, and generation configuration under:

1. **No PRA**: ordinary inference for the exact model and precision.
2. **PRA - No Adaptor**: generic PRA routing and structural mapping.
3. **PRA - Adaptor Bundle**: an immutable, precision-qualified learned adaptor.

When no learned adaptor has passed the exact qualification, the third condition
is `NO_QUALIFIED_ADAPTER`. Planned runs remain `NEEDS_RUN`; unsupported or
memory-blocked runs retain their explicit cause.

## Run a qualification

```bash
pra model qualify-precision \
  --model Qwen/Qwen3-4B \
  --revision MODEL_COMMIT \
  --precision bf16 \
  --encoding PyTorch-bfloat16 \
  --engine hf \
  --dataset multihop-rag \
  --profile balanced \
  --output .pra/qualification/qwen3-4b-bf16
```

The command resolves the immutable identity and emits a manifest, canonical
evidence record, CSV row, Hugging Face card fragment, and LaTeX table. Supplying
`--evidence` imports measured values only when the model, revision, precision,
engine, mode, profile, and dataset all match.

Use `--memory-gate` with a JSON observation file to retain load and context
limits. Without measured input, the command creates a reproducible run plan;
it does not fabricate benchmark values.

## Current coverage

The [bundle catalog](catalog.md) and [qualification matrix](qualification-matrix.md)
show exact precision identities. The catalog-derived ladder contains inherited
MLX 4-bit and 8-bit evidence and preserves every unmeasured cell. A companion
matched study adds BF16 evidence for Qwen3-4B without treating it as a
published adaptor bundle.

A controlled M5 run also compares Qwen3-4B BF16, MLX 8-bit, and MLX 4-bit deployed
pipelines on the same 10 MultiHop-RAG questions, seed, candidate counts, and
2,048-token budget. Generic PRA changes token F1 by `-0.0176` and `-0.0091`
at 4-bit, `-0.0035` and `+0.0036` at 8-bit, and `-0.0019` and `+0.0072` at
BF16 for 20 and 50 candidates. BF16 and 8-bit preserve task score; the 4-bit
arm loses one or two answers. PRA total latency is `1.058-1.061x` baseline at
BF16, `1.100-1.107x` at 8-bit, and `1.148-1.161x` at 4-bit.
The pinned BF16 load contains 398 BF16 parameter arrays and no quantized
layers.

This is a matched *pipeline* comparison, not a selector-frozen transport
ablation: the baseline uses standard selected-text retrieval, while PRA uses
hybrid retrieval and detached native K/V. It is directional evidence that the
three-rung trend improves monotonically toward BF16 in this cohort. The source
checkpoint is BF16, so casting it to float32 would not create an independent
FP32 reference. A selector-frozen study, peak-memory capture, learned-adaptor
arm, and cross-precision adaptor transfer remain open experiments.

## Encoding caveats

- Quantized model weights and quantized PRA K/V are separate policies.
- Converted artifacts require source revision, converter/version, recipe, and
  checksum provenance.
- Adaptors trained from one feature dtype are portable only after a matched
  transfer experiment.
- Unmatched datasets or conversions may be reported as adjacent evidence, but
  not as a precision effect.
