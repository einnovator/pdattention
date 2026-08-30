"""Cross-engine OpenAI-compatible serving benchmark used by Papers 6--6.2."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ServingSample:
    """One streamed request with user-visible and cache telemetry."""

    condition: str
    repeat: int
    ttft_ms: float | None
    completion_latency_ms: float
    mean_itl_ms: float | None
    output_events: int
    output_text: str
    expected_answer_present: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile, or ``None`` for no data."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_messages(
    *,
    distractor_count: int = 12,
    distractor_repeat: int = 28,
) -> dict[str, list[dict[str, str]]]:
    """Build conditions that separate exact-prefix and selected memory.

    The defaults preserve the original smoke workload. Benchmark campaigns can
    scale the retained context without changing the target evidence or prompt
    structure, which keeps comparisons across engines selector-frozen.
    """

    if distractor_count < 0 or distractor_repeat < 0:
        raise ValueError("Distractor count and repeat must be non-negative.")

    stable_prefix = (
        "You are a deterministic evidence reader. Preserve this stable session "
        "prefix across requests. " + "stable-prefix-token " * 96
    )
    target = (
        "Resource target: the requested verification code is PRA_EVIDENCE_4821. "
        "Return only that code when asked."
    )
    distractors = [
        f"Resource distractor {index}: code DECOY_{index:04d}. "
        + "irrelevant-record-token " * distractor_repeat
        for index in range(distractor_count)
    ]
    question = "What is the requested verification code? Return only the code."
    selected = f"Selected PRA text memory:\n{target}"
    full = "Full retained context:\n" + "\n".join([*distractors, target])
    return {
        "no_prefix_no_pra": [{"role": "user", "content": question}],
        "prefix_only": [
            {"role": "system", "content": stable_prefix},
            {"role": "user", "content": question},
        ],
        "pra_only": [
            {"role": "user", "content": f"{selected}\n\n{question}"},
        ],
        "prefix_plus_pra": [
            {"role": "system", "content": stable_prefix},
            {"role": "user", "content": f"{selected}\n\n{question}"},
        ],
        "full_context": [
            {"role": "system", "content": stable_prefix},
            {"role": "user", "content": f"{full}\n\n{question}"},
        ],
    }


def _usage_value(usage: Mapping[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return None if value is None else int(value)


def _cached_tokens(usage: Mapping[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details") or usage.get("prompt_token_details") or {}
    value = details.get("cached_tokens")
    if value is None:
        value = usage.get("cached_tokens")
    return None if value is None else int(value)


def stream_chat_completion(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    cache_salt: str | None,
    max_tokens: int = 16,
) -> dict[str, Any]:
    """Issue one SSE chat completion and return portable timing telemetry."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_content: float | None = None
    content_times: list[float] = []
    pieces: list[str] = []
    usage: Mapping[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or ()
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if not content:
                continue
            observed = time.perf_counter()
            if first_content is None:
                first_content = observed
            content_times.append(observed)
            pieces.append(str(content))
    ended = time.perf_counter()
    intervals = [
        (right - left) * 1000.0
        for left, right in zip(content_times, content_times[1:])
    ]
    return {
        "ttft_ms": None if first_content is None else (first_content - started) * 1000.0,
        "completion_latency_ms": (ended - started) * 1000.0,
        "mean_itl_ms": statistics.fmean(intervals) if intervals else None,
        "output_events": len(content_times),
        "output_text": "".join(pieces),
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "cached_tokens": _cached_tokens(usage),
    }


def run_serving_benchmark(
    base_url: str,
    *,
    model: str,
    engine: str,
    repeats: int,
    timeout_seconds: float = 180.0,
    use_cache_salt: bool = False,
) -> dict[str, Any]:
    """Run fixed conditions and aggregate only statistically defensible fields."""

    if repeats < 2:
        raise ValueError("At least two repeats are required to separate cold and warm requests.")
    cache_salt = None
    if use_cache_salt:
        cache_salt = hashlib.sha256(b"paper6-serving:tenant-a").hexdigest()
    samples: list[ServingSample] = []
    for condition, messages in benchmark_messages().items():
        for repeat in range(repeats):
            values = stream_chat_completion(
                base_url,
                model=model,
                messages=messages,
                timeout_seconds=timeout_seconds,
                cache_salt=cache_salt,
            )
            samples.append(
                ServingSample(
                    condition=condition,
                    repeat=repeat,
                    expected_answer_present="PRA_EVIDENCE_4821" in values["output_text"],
                    **values,
                )
            )
    aggregates = []
    for condition in benchmark_messages():
        rows = [sample for sample in samples if sample.condition == condition]
        ttft = [row.ttft_ms for row in rows if row.ttft_ms is not None]
        latency = [row.completion_latency_ms for row in rows]
        aggregates.append(
            {
                "condition": condition,
                "sample_count": len(rows),
                "quality_success_rate": statistics.fmean(
                    float(row.expected_answer_present) for row in rows
                ),
                "cold_ttft_ms": rows[0].ttft_ms,
                "warm_ttft_ms_mean": (
                    statistics.fmean(
                        row.ttft_ms for row in rows[1:] if row.ttft_ms is not None
                    )
                    if any(row.ttft_ms is not None for row in rows[1:])
                    else None
                ),
                "ttft_ms_p50": percentile(ttft, 0.5),
                "completion_latency_ms_p50": percentile(latency, 0.5),
                "mean_prompt_tokens": (
                    statistics.fmean(
                        row.prompt_tokens for row in rows if row.prompt_tokens is not None
                    )
                    if any(row.prompt_tokens is not None for row in rows)
                    else None
                ),
                "mean_cached_tokens": (
                    statistics.fmean(
                        row.cached_tokens for row in rows if row.cached_tokens is not None
                    )
                    if any(row.cached_tokens is not None for row in rows)
                    else None
                ),
                "tail_latency_status": "NOT_REPORTED_SAMPLE_TOO_SMALL",
            }
        )
    return {
        "schema_version": "1.0",
        "benchmark": "paper6-cross-engine-prefix-pra-smoke",
        "engine": engine,
        "model_id": model,
        "expected_answer": "PRA_EVIDENCE_4821",
        "repeats": repeats,
        "evidence_tier": "SMOKE",
        "measurement_status": "MEASURED",
        "cache_salt_enabled": use_cache_salt,
        "conditions": list(benchmark_messages()),
        "samples": [sample.to_dict() for sample in samples],
        "aggregates": aggregates,
    }
