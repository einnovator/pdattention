from __future__ import annotations

import json
import threading
import urllib.request
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
