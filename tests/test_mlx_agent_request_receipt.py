from __future__ import annotations

import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

from pra_hf.deployment import PRAEngineResult
from experiments.paper4_5_agent.serve_mlx_agent_pra import _direct_handler


def test_mlx_endpoint_writes_request_shape_without_message_text(tmp_path) -> None:
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            return [1, 2, 3]

    class Executor:
        tokenizer = Tokenizer()

        @staticmethod
        def generate(request):
            return PRAEngineResult(
                text="OK",
                raw={
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                    "pra": {"native_kv_used": False},
                },
                trace=(),
            )

        @staticmethod
        def close_session(session_id):
            raise AssertionError(f"unexpected ephemeral session {session_id}")

        @staticmethod
        def release_request_temporaries():
            return {
                "active_bytes_before": 20,
                "active_bytes_after": 10,
                "cache_bytes_before": 5,
                "cache_bytes_after": 0,
                "python_objects_collected": 2,
            }

    receipt = tmp_path / "requests.jsonl"
    interaction = tmp_path / "interactions.jsonl"
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _direct_handler(
            Executor(), "model", request_log=receipt,
            interaction_log=interaction,
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = {
            "model": "model",
            "messages": [{"role": "user", "content": "secret prompt"}],
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": 7,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
        assert payload["choices"][0]["message"]["content"] == "OK"
        assert payload["pra"]["post_request_cleanup"] == {
            "active_bytes_before": 20,
            "active_bytes_after": 10,
            "cache_bytes_before": 5,
            "cache_bytes_after": 0,
            "python_objects_collected": 2,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["request_start", "request_end"]
    assert rows[0]["rendered_prompt_tokens"] == 3
    assert rows[0]["messages"][0]["content_chars"] == 13
    assert "secret prompt" not in receipt.read_text()
    assert rows[1]["usage"]["total_tokens"] == 4
    assert rows[1]["post_request_cleanup"]["cache_bytes_after"] == 0
    transcript = [json.loads(line) for line in interaction.read_text().splitlines()]
    assert transcript[0]["messages"] == [{
        "message_index": 0,
        "role": "user",
        "content": "secret prompt",
        "content_chars": 13,
        "elided_chars": 0,
        "content_sha256": rows[0]["messages"][0]["content_sha256"],
    }]


def test_mlx_endpoint_rejects_a_stale_run_lease(tmp_path) -> None:
    executor = type("Executor", (), {"tokenizer": object()})()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _direct_handler(executor, "model", run_token="owned-run"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=json.dumps({"model": "model", "messages": []}).encode(),
            headers={"content-type": "application/json", "X-PRA-Run-Lease": "stale"},
        )
        try:
            urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            assert error.code == 409
            payload = json.loads(error.read())
        else:
            raise AssertionError("stale run lease was accepted")
        assert payload["error"] == "stale_run_lease"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_mlx_endpoint_cleans_up_and_records_rejected_request(tmp_path) -> None:
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            return [1, 2, 3]

    class Executor:
        tokenizer = Tokenizer()
        cleanup_calls = 0

        @staticmethod
        def generate(request):
            raise ValueError("declared context exceeds frozen engine window")

        @classmethod
        def release_request_temporaries(cls):
            cls.cleanup_calls += 1
            return {
                "active_bytes_before": 30,
                "active_bytes_after": 20,
                "cache_bytes_before": 10,
                "cache_bytes_after": 0,
                "python_objects_collected": 1,
            }

    receipt = tmp_path / "requests.jsonl"
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _direct_handler(Executor(), "model", request_log=receipt),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = {
            "model": "model",
            "messages": [{"role": "user", "content": "private oversized prompt"}],
            "max_tokens": 1024,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            assert error.code == 400
            payload = json.loads(error.read())
        else:
            raise AssertionError("invalid request was accepted")
        assert payload == {
            "error": "ValueError",
            "message": "declared context exceeds frozen engine window",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert Executor.cleanup_calls == 1
    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["request_start", "request_error"]
    assert rows[1]["http_status"] == 400
    assert rows[1]["error_type"] == "ValueError"
    assert rows[1]["post_request_cleanup"]["cache_bytes_after"] == 0
    assert "private oversized prompt" not in receipt.read_text()
