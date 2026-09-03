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
show exact precision identities. The checked-in precision ladder currently
contains matched MLX 4-bit and 8-bit evidence plus explicit pending FP32/BF16
cells. The complete Qwen3-4B four-precision MultiHop-RAG study and
cross-precision adaptor transfer remain open experiments.

## Encoding caveats

- Quantized model weights and quantized PRA K/V are separate policies.
- Converted artifacts require source revision, converter/version, recipe, and
  checksum provenance.
- Adaptors trained from one feature dtype are portable only after a matched
  transfer experiment.
- Unmatched datasets or conversions may be reported as adjacent evidence, but
  not as a precision effect.
