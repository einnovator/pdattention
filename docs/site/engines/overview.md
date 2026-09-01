# Engine Support

PRA separates context selection from model-native reuse and scheduler
integration. A deeper integration is not automatically a better deployment.
Start with the recommendation in this table, then qualify the next capability
against the same frozen evidence selection.

| Engine | Selected Context | Typed PRA Transport | Native Memory | Native Serving | Recommended today | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| [Hugging Face](hugging-face.md) | ✅ Validated | ✅ Validated | ✅ Validated | ⏳ Not measured | Selected Context for applications; Native Memory for reference and research workloads. | Model-backed |
| [MLX](mlx.md) | ✅ Validated | ✅ Validated | ✅ Validated | 🧪 Candidate | Use Selected Context by default; qualify Native Memory for repeated immutable resources. | Natural workload |
| [vLLM](vllm.md) | ✅ Validated | ✅ Validated | 🧪 Candidate | 🧪 Candidate | Selected Context through the gateway or ordinary vLLM API. | Serving |
| [SGLang](sglang.md) | ✅ Validated | ✅ Validated | ✅ Validated | 🧪 Candidate | Selected Context unless deploying the measured companion integration under controlled scope. | Serving |
| [OpenVINO](openvino.md) | ✅ Validated | ✅ Validated | ⏳ Not qualified | ⛔ Not applicable | Selected Context. | Natural workload |
| [TensorRT-LLM](tensorrt-llm.md) | 🧪 Candidate | ✅ Validated | ⏳ Not measured | ⏳ Not measured | Selected Context after local model and engine validation. | Candidate |
| [AirLLM](airllm.md) | ✅ Validated | ✅ Validated | 🧪 Research only | ⏳ Not measured | Selected Context. | Natural workload |
| [llama.cpp](llama-cpp.md) | ✅ Validated | ✅ Validated | 🧪 Candidate | ⏳ Not measured | Selected Context. | Controlled |
| [Ollama](ollama.md) | ✅ Validated | ✅ Validated | 🧪 Candidate | ⏳ Not measured | Selected Context with keep-alive tuned to the workload. | Natural workload |
| [FreeToken](freetoken.md) | 🧪 Candidate | 🧪 Candidate | ⏳ Not measured | ⏳ Not measured | Use only for controlled logical-transport experiments. | Controlled |

## Key

- ✅ **Validated / measured / recommended:** passed the stated evidence boundary.
- 🧪 **Candidate / research-only:** implemented or under study, but not a default.
- ⏳ **Qualification pending / not measured:** evidence is incomplete; this is not zero.
- ⛔ **Blocked / not applicable:** the required engine seam is unavailable in the stated scope.

## Reading the matrix

**Validated** means the named capability passed the scope described on
  that engine's page. It is not a claim about every model or workload.
**Candidate** or **Research only** means the mechanism exists but has not
  passed the product qualification boundary.
**Not measured** is unknown. It is never interpreted as zero.
**Not applicable** means the required engine seam does not currently exist.

See [Metrics & Qualification](../metrics.md) for comparison contracts and
[Research / Evidence](../research/index.md) for paper terminology and raw
artifact provenance.

_Generated from the engine documentation registry and 265 product-matrix rows; evidence current through 2026-09-01._

## Observability integration

Every engine dashboard combines PRA's normalized selection, realization,
prefix/native reuse, and storage metrics with the engine's own telemetry where
available. vLLM can propagate native OTel and exposes serving metrics; SGLang,
OpenVINO/OVMS, TensorRT-LLM/Triton, and llama.cpp expose useful native metrics;
MLX, HF, AirLLM, Ollama, and FreeToken use wrapper observations where a native
surface is absent. Capability does not imply enablement. See the
[observability overview](../observability.md) and [Grafana dashboards](../observability/grafana.md).
