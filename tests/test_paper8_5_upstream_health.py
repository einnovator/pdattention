from __future__ import annotations

from experiments.paper8_5_agent_memory.upstream_health import (
    probe_generation_health,
)


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return b'{"choices":[{"message":{"content":"OK"}}]}'


def test_generation_health_requires_all_exact_probes() -> None:
    calls = []

    def opener(request, *, timeout):
        calls.append((request.full_url, timeout))
        return _Response()

    result = probe_generation_health(
        base_url="http://engine.test:11435", model="locked-model", count=3,
        latency_ceiling_seconds=1, timeout_seconds=2, opener=opener,
    )

    assert result["healthy"] is True
    assert len(result["probes"]) == 3
    assert calls == [("http://engine.test:11435/v1/chat/completions", 2)] * 3


def test_generation_health_fails_closed_on_transport_error() -> None:
    def opener(_request, *, timeout):
        raise TimeoutError(f"stalled after {timeout}")

    result = probe_generation_health(
        base_url="http://engine.test/v1", model="locked-model", count=3,
        latency_ceiling_seconds=1, timeout_seconds=2, opener=opener,
    )

    assert result["healthy"] is False
    assert len(result["probes"]) == 1
    assert result["probes"][0]["error_type"] == "TimeoutError"
