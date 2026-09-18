from __future__ import annotations

import json

from experiments.paper8_5_agent_memory.upstream_health import (
    normalize_ollama_cold_start,
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


def test_ollama_cold_start_unloads_then_warms_same_consumer() -> None:
    calls = []

    class Response:
        status = 200

        def __init__(self, body: bytes):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return self.body

    responses = [
        Response(b'{"done":true}'),
        Response(b'{"choices":[{"message":{"content":"OK"}}]}'),
    ]

    def opener(request, *, timeout):
        calls.append((request.full_url, json.loads(request.data), timeout))
        return responses.pop(0)

    result = normalize_ollama_cold_start(
        base_url="http://engine.test:11435/v1",
        model="locked-model",
        timeout_seconds=7,
        opener=opener,
    )

    assert result["healthy"] is True
    assert result["successful_attempt"] == 1
    assert calls[0][0] == "http://engine.test:11435/api/generate"
    assert calls[0][1]["keep_alive"] == 0
    assert calls[1][0] == "http://engine.test:11435/v1/chat/completions"
    assert calls[1][1]["seed"] == 0


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


def test_generation_health_qualifies_and_reuses_one_connection() -> None:
    class Response:
        status = 200

        def __init__(self, body: bytes):
            self.body = body

        def read(self) -> bytes:
            return self.body

    class Connection:
        def __init__(self):
            self.requests = []
            self.responses = [
                Response(b'{"version":"test"}'),
                *[
                    Response(b'{"choices":[{"message":{"content":"OK"}}]}')
                    for _ in range(3)
                ],
            ]
            self.closed = False

        def request(self, method, path, body=None, headers=None):
            self.requests.append((method, path, body, headers))

        def getresponse(self):
            return self.responses.pop(0)

        def close(self):
            self.closed = True

    connection = Connection()
    factories = []

    def factory(host, port, *, timeout):
        factories.append((host, port, timeout))
        return connection

    result = probe_generation_health(
        base_url="http://engine.test:11435", model="locked-model", count=3,
        latency_ceiling_seconds=1, timeout_seconds=2,
        qualification_path="/api/version", connect_attempts=3,
        connection_factory=factory,
    )

    assert result["healthy"] is True
    assert result["connection_qualification"]["healthy"] is True
    assert factories == [("engine.test", 11435, 2)]
    assert [row[:2] for row in connection.requests] == [
        ("GET", "/api/version"),
        *[("POST", "/v1/chat/completions") for _ in range(3)],
    ]
    assert connection.closed is True


def test_generation_health_can_use_direct_curl_transport() -> None:
    calls = []

    class Completed:
        returncode = 0
        stderr = b""
        stdout = b'{"choices":[{"message":{"content":"OK"}}]}\n200'

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    result = probe_generation_health(
        base_url="http://engine.test:11435", model="locked-model", count=3,
        latency_ceiling_seconds=1, timeout_seconds=2,
        curl_executable="/usr/bin/curl", curl_runner=runner,
    )

    assert result["healthy"] is True
    assert len(calls) == 3
    assert all(call[0][-1].endswith("/v1/chat/completions") for call in calls)
    assert all(call[1]["check"] is False for call in calls)


def test_generation_health_requires_active_runtime_context() -> None:
    class Response:
        status = 200

        def __init__(self, body: bytes):
            self.body = body

        def read(self) -> bytes:
            return self.body

    class Connection:
        def __init__(self):
            self.responses = [
                Response(b'{"version":"test"}'),
                Response(
                    b'{"models":[{"name":"locked-model",'
                    b'"context_length":131072}]}'
                ),
                *[
                    Response(b'{"choices":[{"message":{"content":"OK"}}]}')
                    for _ in range(3)
                ],
            ]

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return self.responses.pop(0)

        def close(self):
            return None

    result = probe_generation_health(
        base_url="http://engine.test:11435", model="locked-model", count=3,
        latency_ceiling_seconds=1, timeout_seconds=2,
        qualification_path="/api/version", runtime_state_path="/api/ps",
        minimum_active_context_tokens=131072,
        connection_factory=lambda *_args, **_kwargs: Connection(),
    )

    assert result["healthy"] is True
    assert result["runtime_context_qualification"] == {
        "status": 200,
        "latency_seconds": result["runtime_context_qualification"]["latency_seconds"],
        "model_found": True,
        "active_context_tokens": 131072,
        "minimum_active_context_tokens": 131072,
        "healthy": True,
    }


def test_generation_health_fails_closed_on_undersized_active_context() -> None:
    class Response:
        status = 200

        def __init__(self, body: bytes):
            self.body = body

        def read(self) -> bytes:
            return self.body

    class Connection:
        def __init__(self):
            self.responses = [
                Response(b'{"version":"test"}'),
                Response(
                    b'{"models":[{"name":"locked-model",'
                    b'"context_length":32768}]}'
                ),
            ]

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return self.responses.pop(0)

        def close(self):
            return None

    result = probe_generation_health(
        base_url="http://engine.test:11435", model="locked-model", count=3,
        latency_ceiling_seconds=1, timeout_seconds=2,
        qualification_path="/api/version", runtime_state_path="/api/ps",
        minimum_active_context_tokens=131072,
        connection_factory=lambda *_args, **_kwargs: Connection(),
    )

    assert result["healthy"] is False
    assert result["runtime_context_qualification"]["active_context_tokens"] == 32768
    assert result["probes"][0]["healthy"] is False
