# Canonical Evidence Matrix

This page preserves the staged attribution chain: ordinary **No PRA**, PRA **Selected Context**, **Native Memory**, and **Native Serving**, with bundle/adaptor use as an orthogonal condition. Values are absolute measurements and pairwise deltas name both source and target. Missing data is never rendered as zero.

## Coverage by model, engine, mode, and profile

| Model | Precision | Engine | Mode | Profile | No PRA | Mode / no adaptor | Same mode / bundle | Evidence tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mlx-community/Qwen3-14B-4bit` | INT4 / MLX-4bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/Qwen3-14B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-14B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | BALANCED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | NEEDS_RERUN |
| `mlx-community/Qwen3-14B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-14B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | NEEDS_RERUN |
| `mlx-community/Qwen3-32B-4bit` | INT4 / MLX-4bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/Qwen3-32B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-32B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | BALANCED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RERUN |
| `mlx-community/Qwen3-32B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-4bit` | INT4 / MLX-4bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/Qwen3-8B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | BALANCED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `Qwen/Qwen2.5-1.5B-Instruct` | BF16 / PyTorch-bfloat16 | hf | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | BF16 / PyTorch-bfloat16 | hf | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | BF16 / PyTorch-bfloat16 | hf | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `Qwen/Qwen2.5-1.5B-Instruct` | BF16 / PyTorch-bfloat16 | hf | Native Memory | QASPER-LEARNED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | CONTROLLED |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | BF16 / PyTorch-bfloat16 | hf | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | BF16 / PyTorch-bfloat16 | hf | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | CONTROLLED |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | BF16 / PyTorch-bfloat16 | hf | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | BF16 / PyTorch-bfloat16 | hf | Native Memory | QASPER-LEARNED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | CONTROLLED |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | INT4 / MLX-4bit | mlx | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | CONTROLLED |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | INT4 / MLX-4bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | CONTROLLED |
| `mlx-community/Qwen3-4B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `mlx-community/Qwen3-4B-4bit` | INT4 / MLX-4bit | mlx | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | CONTROLLED |
| `mlx-community/Qwen3-4B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `mlx-community/Qwen3-4B-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | CONTROLLED |
| `mlx-community/gemma-3-1b-it-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `mlx-community/gemma-3-1b-it-4bit` | INT4 / MLX-4bit | mlx | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | CONTROLLED |
| `mlx-community/gemma-3-1b-it-4bit` | INT4 / MLX-4bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `mlx-community/gemma-3-1b-it-4bit` | INT4 / MLX-4bit | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | CONTROLLED |
| `Qwen/Qwen3-0.6B` | FP16 / PyTorch-float16 | hf | Selected Context | QUALITY | NOT_MEASURED | Selected Context: CALIBRATION_PENDING | Selected Context + Bundle: CALIBRATION_PENDING | RESEARCH |
| `Qwen/Qwen3-0.6B` | FP16 / PyTorch-float16 | hf | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | RESEARCH |
| `Qwen/Qwen3-0.6B` | FP16 / PyTorch-float16 | hf | Selected Context | ECONOMY | NOT_MEASURED | Selected Context: CALIBRATION_PENDING | Selected Context + Bundle: CALIBRATION_PENDING | RESEARCH |
| `Qwen/Qwen3-0.6B` | FP16 / PyTorch-float16 | hf | Selected Context | QASPER-LEARNED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | RESEARCH |
| `mlx-community/Qwen3-4B-8bit` | INT8 / MLX-8bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/Qwen3-4B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-4B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | BALANCED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | NEEDS_RERUN |
| `mlx-community/Qwen3-4B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-4B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | QASPER-LEARNED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NEEDS_RUN | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-8bit` | INT8 / MLX-8bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/Qwen3-8B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | BALANCED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-14B-8bit` | INT8 / MLX-8bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/Qwen3-14B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-14B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | BALANCED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RERUN |
| `mlx-community/Qwen3-14B-8bit` | INT8 / MLX-8bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-6bit` | INT6 / MLX-6bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/Qwen3-8B-6bit` | INT6 / MLX-6bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-6bit` | INT6 / MLX-6bit | mlx | Native Memory | BALANCED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RERUN |
| `mlx-community/Qwen3-8B-6bit` | INT6 / MLX-6bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | INT8 / MLX-8bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | INT8 / MLX-8bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | INT8 / MLX-8bit | mlx | Native Memory | BALANCED | NOT_MEASURED | Native Memory: NEEDS_RUN | Native Memory + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RERUN |
| `mlx-community/Llama-3.2-1B-Instruct-8bit` | INT8 / MLX-8bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/gemma-3-1b-it-8bit` | INT8 / MLX-8bit | mlx | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `mlx-community/gemma-3-1b-it-8bit` | INT8 / MLX-8bit | mlx | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/gemma-3-1b-it-8bit` | INT8 / MLX-8bit | mlx | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RERUN |
| `mlx-community/gemma-3-1b-it-8bit` | INT8 / MLX-8bit | mlx | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `Qwen/Qwen2.5-1.5B-Instruct` | INT8 / bitsandbytes-8bit-LLM.int8 | hf | native runtime | ALL | NEEDS_RUN | QUARANTINED (1 pre-fix record(s)) | NEEDS_RUN | CORRECTION_PENDING |
| `Qwen/Qwen2.5-1.5B-Instruct` | INT8 / bitsandbytes-8bit-LLM.int8 | hf | Native Memory | QUALITY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `Qwen/Qwen2.5-1.5B-Instruct` | INT8 / bitsandbytes-8bit-LLM.int8 | hf | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NO_QUALIFIED_ADAPTER | NEEDS_RERUN |
| `Qwen/Qwen2.5-1.5B-Instruct` | INT8 / bitsandbytes-8bit-LLM.int8 | hf | Native Memory | ECONOMY | NOT_MEASURED | Native Memory: CALIBRATION_PENDING | Native Memory + Bundle: CALIBRATION_PENDING | NEEDS_RERUN |
| `mlx-community/Qwen3.5-27B-4bit` | INT4 / MLX-4bit | mlx | Selected Context | QUALITY | NOT_MEASURED | Selected Context: CALIBRATION_PENDING | Selected Context + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `mlx-community/Qwen3.5-27B-4bit` | INT4 / MLX-4bit | mlx | Selected Context | BALANCED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | CONTROLLED |
| `mlx-community/Qwen3.5-27B-4bit` | INT4 / MLX-4bit | mlx | Selected Context | ECONOMY | NOT_MEASURED | Selected Context: CALIBRATION_PENDING | Selected Context + Bundle: CALIBRATION_PENDING | CONTROLLED |
| `mlx-community/Qwen3.5-27B-4bit` | INT4 / MLX-4bit | mlx | Selected Context | QASPER-LEARNED | NOT_MEASURED | Selected Context: NEEDS_RUN | Selected Context + Bundle: NEEDS_RUN | CONTROLLED |

A `MEASURED (n)` cell reports the number of scalar metrics available for that exact condition. Detailed values follow only for measured records; profile rows without matched evidence remain explicit.

## Measured absolute values and deltas

No exact-identity canonical records are currently packaged.

## Interpretation

Bundle/adaptor use is intentionally distinct from execution depth. A published bundle may contain only structural mapping and profile metadata, or may include an opt-in learned router. A bundle cell becomes measured only when the immutable bundle revision was resolved during the run.

Routing-only recall is reported in each model card's research diagnostics and does not substitute for answer quality, TTFT, ITL, throughput, or memory measurements.
