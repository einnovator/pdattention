"""Measure physical Intel costs for matched OpenVINO E0 representations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import psutil

from experiments.engine_serving.run_openai_natural_e0 import (
    _bounded_text,
    _quality,
)
from experiments.paper6_3_openvino.run_natural_e0 import (
    _decoded_text,
    _history,
    _metric,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _numeric_total(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, Mapping):
        values = [_numeric_total(item) for item in value.values()]
        finite = [item for item in values if item is not None]
        return sum(finite) if finite else None
    return None


@dataclass
class PhysicalSampler:
    """Poll nonprivileged process and OpenVINO plugin counters."""

    core: Any
    device: str
    interval_seconds: float = 0.05
    process: psutil.Process = field(default_factory=psutil.Process)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def supported_gpu_properties(self) -> list[str]:
        try:
            properties = self.core.get_property(self.device, "SUPPORTED_PROPERTIES")
        except Exception:
            return []
        return [
            str(name)
            for name in properties
            if any(term in str(name).upper() for term in ("MEMORY", "POWER", "UTILIZATION"))
        ]

    def _sample(self) -> dict[str, Any]:
        memory = self.process.memory_info()
        io = self.process.io_counters()
        plugin = {}
        for name in self.supported_gpu_properties():
            try:
                plugin[name] = _jsonable(self.core.get_property(self.device, name))
            except Exception as error:
                plugin[name] = {"unavailable": type(error).__name__}
        return {
            "observed_at": time.perf_counter(),
            "rss_bytes": int(memory.rss),
            "vms_bytes": int(memory.vms),
            "system_available_bytes": int(psutil.virtual_memory().available),
            "read_bytes": int(io.read_bytes),
            "write_bytes": int(io.write_bytes),
            "plugin": plugin,
            "plugin_numeric_bytes": {
                name: _numeric_total(value) for name, value in plugin.items()
            },
        }

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples.append(self._sample())

    def start(self) -> None:
        self.samples = [self._sample()]
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.samples.append(self._sample())
        first, last = self.samples[0], self.samples[-1]
        plugin_peaks: dict[str, int] = {}
        for sample in self.samples:
            for name, value in sample["plugin_numeric_bytes"].items():
                if value is not None:
                    plugin_peaks[name] = max(plugin_peaks.get(name, 0), int(value))
        return {
            "sample_count": len(self.samples),
            "rss_peak_bytes": max(sample["rss_bytes"] for sample in self.samples),
            "rss_delta_bytes": last["rss_bytes"] - first["rss_bytes"],
            "vms_peak_bytes": max(sample["vms_bytes"] for sample in self.samples),
            "system_available_min_bytes": min(
                sample["system_available_bytes"] for sample in self.samples
            ),
            "process_read_bytes": last["read_bytes"] - first["read_bytes"],
            "process_write_bytes": last["write_bytes"] - first["write_bytes"],
            "plugin_numeric_peaks": plugin_peaks,
            "plugin_first": first["plugin"],
            "plugin_last": last["plugin"],
        }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def run(args: argparse.Namespace) -> dict[str, Any]:
    import openvino as ov
    import openvino_genai as genai
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    entries = []
    for entry in manifest["entries"]:
        dataset = str(entry["dataset"])
        if args.datasets and dataset not in args.datasets:
            continue
        if counts.get(dataset, 0) >= args.max_examples_per_dataset:
            continue
        counts[dataset] = counts.get(dataset, 0) + 1
        entries.append(entry)

    scheduler = genai.SchedulerConfig()
    scheduler.enable_prefix_caching = True
    scheduler.dynamic_split_fuse = True
    scheduler.max_num_seqs = 8
    scheduler.max_num_batched_tokens = 2048
    scheduler.cache_size = 1
    pipe = genai.LLMPipeline(args.model, args.device, scheduler_config=scheduler)
    config = genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = False
    core = ov.Core()
    sampler = PhysicalSampler(core, args.device, args.sample_interval_ms / 1000.0)
    rows = []
    for entry_index, entry in enumerate(entries):
        selected, selected_tokens = _bounded_text(
            tokenizer, str(entry["selected_source"]), args.max_selected_tokens
        )
        full, full_tokens = _bounded_text(
            tokenizer,
            f"{selected}\n\nDistractor material:\n{entry['distractor_source']}",
            args.max_full_tokens,
        )
        conditions = (
            ("selected_text_e0", selected, selected_tokens),
            ("full_context_e0", full, full_tokens),
        )
        for repeat in range(args.repeats):
            ordered = (
                conditions
                if (entry_index + repeat) % 2 == 0
                else tuple(reversed(conditions))
            )
            for condition, evidence, source_tokens in ordered:
                prompt = (
                    f"Evidence:\n{evidence}\n\nQuestion: {entry['question']}\n"
                    "Answer briefly using only the evidence."
                )
                cpu_before = sampler.process.cpu_times()
                sampler.start()
                started = time.perf_counter()
                result = pipe.generate(
                    _history(genai, [{"role": "user", "content": prompt}]), config
                )
                wall_ms = (time.perf_counter() - started) * 1000.0
                physical = sampler.finish()
                cpu_after = sampler.process.cpu_times()
                output = _decoded_text(result)
                exact, f1, containment = _quality(output, str(entry["answer"]))
                perf = result.perf_metrics
                rows.append(
                    {
                        "dataset": entry["dataset"],
                        "example_id": entry["example_id"],
                        "condition": condition,
                        "repeat": repeat,
                        "cache_state": "COLD" if repeat == 0 else "WARM_REPEAT",
                        "source_tokens": source_tokens,
                        "prompt_tokens": _metric(perf, "get_num_input_tokens"),
                        "completion_tokens": _metric(perf, "get_num_generated_tokens"),
                        "ttft_ms": _metric(perf, "get_ttft"),
                        "mean_itl_ms": _metric(perf, "get_tpot"),
                        "completion_latency_ms": wall_ms,
                        "token_f1": f1,
                        "exact_match": exact,
                        "answer_containment": containment,
                        "process_cpu_seconds": (
                            cpu_after.user
                            + cpu_after.system
                            - cpu_before.user
                            - cpu_before.system
                        ),
                        "physical": physical,
                    }
                )

    aggregates = []
    for condition in ("selected_text_e0", "full_context_e0"):
        selected = [row for row in rows if row["condition"] == condition]
        ttft = [float(row["ttft_ms"]) for row in selected]
        latency = [float(row["completion_latency_ms"]) for row in selected]
        aggregates.append(
            {
                "condition": condition,
                "samples": len(selected),
                "mean_source_tokens": statistics.fmean(
                    float(row["source_tokens"]) for row in selected
                ),
                "mean_token_f1": statistics.fmean(
                    float(row["token_f1"]) for row in selected
                ),
                "ttft_p50_ms": _percentile(ttft, 0.50),
                "ttft_p95_ms": _percentile(ttft, 0.95),
                "completion_p50_ms": _percentile(latency, 0.50),
                "completion_p95_ms": _percentile(latency, 0.95),
                "rss_peak_bytes": max(
                    int(row["physical"]["rss_peak_bytes"]) for row in selected
                ),
                "mean_process_cpu_seconds": statistics.fmean(
                    float(row["process_cpu_seconds"]) for row in selected
                ),
                "mean_process_read_bytes": statistics.fmean(
                    float(row["physical"]["process_read_bytes"])
                    for row in selected
                ),
                "mean_process_write_bytes": statistics.fmean(
                    float(row["physical"]["process_write_bytes"])
                    for row in selected
                ),
                "plugin_numeric_peaks": {
                    name: max(
                        int(row["physical"]["plugin_numeric_peaks"].get(name, 0))
                        for row in selected
                    )
                    for name in sampler.supported_gpu_properties()
                },
            }
        )
    return {
        "schema_version": "paper6.3-openvino-physical-telemetry-v1",
        "evidence_tier": "LIVE_INTEL_PHYSICAL_TELEMETRY",
        "engine_version": importlib.metadata.version("openvino-genai"),
        "model_id": str(args.model),
        "device": args.device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "plugin_physical_properties": sampler.supported_gpu_properties(),
        "energy_status": "NOT_MEASURED_NO_NONPRIVILEGED_COUNTER",
        "gpu_utilization_status": "PLUGIN_PROPERTY_OR_NOT_AVAILABLE",
        "rows": rows,
        "aggregates": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/portable_e0_qa_manifest_expanded.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="*", default=("qasper", "hotpotqa", "2wikimultihopqa")
    )
    parser.add_argument("--max-examples-per-dataset", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-selected-tokens", type=int, default=384)
    parser.add_argument("--max-full-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--sample-interval-ms", type=float, default=50.0)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}))


if __name__ == "__main__":
    main()
