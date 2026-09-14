"""Generation-level health checks for autonomous agent campaigns."""

from __future__ import annotations

import json
from time import monotonic
from typing import Any, Callable
import urllib.request


def _chat_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    return f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"


def probe_generation_health(
    *, base_url: str, model: str, count: int = 3,
    latency_ceiling_seconds: float = 60.0, timeout_seconds: float = 90.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Require repeated deterministic generations below a frozen latency ceiling."""

    if count < 1:
        raise ValueError("health probe count must be positive")
    if latency_ceiling_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("health latency and timeout must be positive")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "stream": False,
        "max_tokens": 8,
    }).encode("utf-8")
    probes: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        request = urllib.request.Request(
            _chat_url(base_url), data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = monotonic()
        try:
            with opener(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            latency = monotonic() - started
            content = str(body["choices"][0]["message"]["content"]).strip()
            valid = content == "OK"
            within_ceiling = latency <= latency_ceiling_seconds
            probes.append({
                "probe_index": index,
                "status": int(getattr(response, "status", 200)),
                "latency_seconds": latency,
                "response_exact": valid,
                "within_latency_ceiling": within_ceiling,
                "healthy": valid and within_ceiling,
            })
            if not valid or not within_ceiling:
                break
        except Exception as error:  # health audit must record transport failures
            probes.append({
                "probe_index": index,
                "latency_seconds": monotonic() - started,
                "healthy": False,
                "error_type": type(error).__name__,
                "error_detail": str(error),
            })
            break
    healthy = len(probes) == count and all(row["healthy"] for row in probes)
    return {
        "schema_version": 1,
        "endpoint": _chat_url(base_url),
        "model": model,
        "required_probe_count": count,
        "latency_ceiling_seconds": latency_ceiling_seconds,
        "timeout_seconds": timeout_seconds,
        "healthy": healthy,
        "probes": probes,
    }
