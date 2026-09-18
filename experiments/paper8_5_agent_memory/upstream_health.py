"""Generation-level health checks for autonomous agent campaigns."""

from __future__ import annotations

import http.client
import json
import subprocess
from time import monotonic, sleep
from typing import Any, Callable
from urllib.parse import urlparse
import urllib.request


def _chat_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    return f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"


def normalize_ollama_cold_start(
    *, base_url: str, model: str, timeout_seconds: float = 180.0,
    connect_attempts: int = 1, connect_retry_seconds: float = 1.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    curl_executable: str | None = None,
    curl_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Unload and deterministically warm one Ollama model before a trial.

    Ollama's OpenAI-compatible endpoint does not expose ``keep_alive=0``.
    The native endpoint is therefore used only for the explicit unload; the
    warmup uses the same OpenAI consumer path as the scored trial.  Any failure
    is returned as a fail-closed receipt rather than silently continuing with
    unknown cache state.
    """

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    unload_payload = json.dumps({
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }).encode("utf-8")
    warmup_payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "stream": False,
        "max_tokens": 8,
    }).encode("utf-8")
    if connect_attempts < 1 or connect_retry_seconds < 0:
        raise ValueError("cold-start retry configuration is invalid")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "mode": "ollama_keep_alive_zero_then_openai_warmup",
        "model": model,
        "unload_endpoint": f"{root}/api/generate",
        "warmup_endpoint": _chat_url(base_url),
        "healthy": False,
        "attempts": [],
    }
    for attempt_index in range(1, connect_attempts + 1):
        attempt: dict[str, Any] = {"attempt": attempt_index, "healthy": False}
        try:
            started = monotonic()
            if curl_executable is not None:
                completed = curl_runner(
                    [
                        curl_executable,
                        "--silent", "--show-error",
                        "--max-time", str(timeout_seconds),
                        "--request", "POST",
                        "--header", "Content-Type: application/json",
                        "--data-binary", "@-",
                        "--output", "-", "--write-out", "\n%{http_code}",
                        receipt["unload_endpoint"],
                    ],
                    input=unload_payload,
                    capture_output=True,
                    timeout=timeout_seconds + 10,
                    check=False,
                )
                if completed.returncode:
                    detail = completed.stderr.decode(
                        "utf-8", errors="replace"
                    ).strip()
                    raise OSError(
                        f"curl exited {completed.returncode}: {detail}"
                    )
                _unload_body, status_text = completed.stdout.rsplit(b"\n", 1)
                unload_status = int(status_text)
            else:
                request = urllib.request.Request(
                    receipt["unload_endpoint"], data=unload_payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with opener(request, timeout=timeout_seconds) as response:
                    response.read()
                    unload_status = int(getattr(response, "status", 200))
            attempt["unload"] = {
                "status": unload_status,
                "latency_seconds": monotonic() - started,
            }
            if unload_status >= 400:
                raise OSError(f"Ollama unload returned HTTP {unload_status}")

            started = monotonic()
            if curl_executable is not None:
                completed = curl_runner(
                    [
                        curl_executable,
                        "--silent", "--show-error",
                        "--max-time", str(timeout_seconds),
                        "--request", "POST",
                        "--header", "Content-Type: application/json",
                        "--data-binary", "@-",
                        "--output", "-", "--write-out", "\n%{http_code}",
                        receipt["warmup_endpoint"],
                    ],
                    input=warmup_payload,
                    capture_output=True,
                    timeout=timeout_seconds + 10,
                    check=False,
                )
                if completed.returncode:
                    detail = completed.stderr.decode(
                        "utf-8", errors="replace"
                    ).strip()
                    raise OSError(
                        f"curl exited {completed.returncode}: {detail}"
                    )
                raw_body, status_text = completed.stdout.rsplit(b"\n", 1)
                warmup_status = int(status_text)
                body = json.loads(raw_body.decode("utf-8"))
            else:
                request = urllib.request.Request(
                    receipt["warmup_endpoint"], data=warmup_payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with opener(request, timeout=timeout_seconds) as response:
                    warmup_status = int(getattr(response, "status", 200))
                    body = json.loads(response.read().decode("utf-8"))
            content = str(body["choices"][0]["message"]["content"]).strip()
            attempt["warmup"] = {
                "status": warmup_status,
                "latency_seconds": monotonic() - started,
                "response_exact": content == "OK",
            }
            attempt["healthy"] = warmup_status < 400 and content == "OK"
            receipt["attempts"].append(attempt)
            if attempt["healthy"]:
                receipt["healthy"] = True
                receipt["successful_attempt"] = attempt_index
                break
        except Exception as error:
            attempt["error_type"] = type(error).__name__
            attempt["error_detail"] = str(error)
            receipt["attempts"].append(attempt)
        if attempt_index < connect_attempts:
            sleep(connect_retry_seconds)
    if not receipt["healthy"] and receipt["attempts"]:
        final = receipt["attempts"][-1]
        receipt["error_type"] = final.get("error_type", "NormalizationError")
        receipt["error_detail"] = final.get(
            "error_detail", "cold-start warmup did not return exact OK"
        )
    return receipt


def probe_generation_health(
    *, base_url: str, model: str, count: int = 3,
    latency_ceiling_seconds: float = 60.0, timeout_seconds: float = 90.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    qualification_path: str | None = None,
    runtime_state_path: str | None = None,
    minimum_active_context_tokens: int | None = None,
    connect_attempts: int = 1,
    connect_retry_seconds: float = 1.0,
    connection_factory: Callable[..., Any] | None = None,
    curl_executable: str | None = None,
    curl_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Require repeated deterministic generations below a frozen latency ceiling."""

    if count < 1:
        raise ValueError("health probe count must be positive")
    if latency_ceiling_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("health latency and timeout must be positive")
    if connect_attempts < 1 or connect_retry_seconds < 0:
        raise ValueError("health connect retry configuration is invalid")
    if qualification_path is not None and not qualification_path.startswith("/"):
        raise ValueError("health qualification path must be absolute")
    if runtime_state_path is not None and not runtime_state_path.startswith("/"):
        raise ValueError("runtime state path must be absolute")
    if minimum_active_context_tokens is not None:
        if minimum_active_context_tokens < 1:
            raise ValueError("minimum active context tokens must be positive")
        if runtime_state_path is None:
            raise ValueError(
                "minimum active context tokens requires a runtime state path"
            )
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
    qualification: dict[str, Any] | None = None
    runtime_context: dict[str, Any] | None = None
    connection: Any | None = None
    chat_url = _chat_url(base_url)
    if qualification_path is not None and curl_executable is None:
        parsed = urlparse(chat_url)
        if connection_factory is None:
            connection_type = (
                http.client.HTTPSConnection
                if parsed.scheme == "https" else http.client.HTTPConnection
            )
            connection_factory = connection_type
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        last_error: Exception | None = None
        for attempt in range(1, connect_attempts + 1):
            started = monotonic()
            candidate = connection_factory(
                parsed.hostname, port, timeout=timeout_seconds
            )
            try:
                candidate.request("GET", qualification_path)
                response = candidate.getresponse()
                response.read()
                if response.status >= 500:
                    raise OSError(
                        f"qualification returned HTTP {response.status}"
                    )
                connection = candidate
                qualification = {
                    "attempt": attempt,
                    "status": response.status,
                    "latency_seconds": monotonic() - started,
                    "healthy": True,
                }
                if runtime_state_path is not None:
                    context_started = monotonic()
                    candidate.request("GET", runtime_state_path)
                    context_response = candidate.getresponse()
                    context_body = json.loads(
                        context_response.read().decode("utf-8")
                    )
                    active_models = list(context_body.get("models") or ())
                    active = next((
                        row for row in active_models
                        if str(row.get("name") or row.get("model")) == model
                    ), None)
                    active_context = (
                        int(active.get("context_length"))
                        if active is not None and active.get("context_length") is not None
                        else None
                    )
                    context_healthy = (
                        context_response.status < 500
                        and active_context is not None
                        and (
                            minimum_active_context_tokens is None
                            or active_context >= minimum_active_context_tokens
                        )
                    )
                    runtime_context = {
                        "status": context_response.status,
                        "latency_seconds": monotonic() - context_started,
                        "model_found": active is not None,
                        "active_context_tokens": active_context,
                        "minimum_active_context_tokens": minimum_active_context_tokens,
                        "healthy": context_healthy,
                    }
                    if not context_healthy:
                        raise OSError(
                            "active runtime context is missing or below the "
                            f"required {minimum_active_context_tokens} tokens: "
                            f"observed {active_context}"
                        )
                break
            except (OSError, http.client.HTTPException) as error:
                candidate.close()
                last_error = error
                if attempt < connect_attempts:
                    sleep(connect_retry_seconds)
        if connection is None:
            qualification = {
                "attempt": connect_attempts,
                "healthy": False,
                "error_type": type(last_error).__name__,
                "error_detail": str(last_error),
            }
            probes.append({
                "probe_index": 1,
                "healthy": False,
                "error_type": type(last_error).__name__,
                "error_detail": str(last_error),
            })
    for index in range(1, count + 1):
        if qualification_path is not None and curl_executable is None and connection is None:
            break
        request = urllib.request.Request(
            chat_url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = monotonic()
        try:
            response_status = 200
            if curl_executable is not None:
                completed = curl_runner(
                    [
                        curl_executable,
                        "--silent", "--show-error",
                        "--max-time", str(timeout_seconds),
                        "--request", "POST",
                        "--header", "Content-Type: application/json",
                        "--data-binary", "@-",
                        "--output", "-", "--write-out", "\n%{http_code}",
                        chat_url,
                    ],
                    input=payload,
                    capture_output=True,
                    timeout=timeout_seconds + 10,
                    check=False,
                )
                if completed.returncode:
                    detail = completed.stderr.decode(
                        "utf-8", errors="replace"
                    ).strip()
                    raise OSError(
                        f"curl exited {completed.returncode}: {detail}"
                    )
                raw_body, status_text = completed.stdout.rsplit(b"\n", 1)
                response_status = int(status_text)
                body = json.loads(raw_body.decode("utf-8"))
            elif connection is not None:
                parsed = urlparse(chat_url)
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                connection.request(
                    "POST", target, body=payload,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response_status = response.status
                body = json.loads(response.read().decode("utf-8"))
            else:
                with opener(request, timeout=timeout_seconds) as response:
                    response_status = int(getattr(response, "status", 200))
                    body = json.loads(response.read().decode("utf-8"))
            latency = monotonic() - started
            content = str(body["choices"][0]["message"]["content"]).strip()
            valid = content == "OK"
            within_ceiling = latency <= latency_ceiling_seconds
            probes.append({
                "probe_index": index,
                "status": response_status,
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
    if connection is not None:
        connection.close()
    healthy = len(probes) == count and all(row["healthy"] for row in probes)
    return {
        "schema_version": 1,
        "endpoint": _chat_url(base_url),
        "model": model,
        "required_probe_count": count,
        "latency_ceiling_seconds": latency_ceiling_seconds,
        "timeout_seconds": timeout_seconds,
        "connection_qualification": qualification,
        "runtime_context_qualification": runtime_context,
        "healthy": healthy,
        "probes": probes,
    }
