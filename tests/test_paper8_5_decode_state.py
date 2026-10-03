from experiments.paper8_5_agent_memory.reduce_decode_state import reduce_decode_state


def _payload(digest: str, response: str, command: str):
    return {
        "rows": [{
            "selected_messages_sha256": digest,
            "response_content_sha256": response,
            "command": command,
            "choice_logprobs": {"content": [{
                "token": "b",
                "logprob": -2.0,
                "top_logprobs": [
                    {"token": "a", "logprob": -0.1},
                    {"token": "b", "logprob": -2.0},
                ],
            }]},
        }]
    }


def test_decode_state_reducer_preserves_clusters_and_identity():
    result = reduce_decode_state([
        ("cold", _payload("same", "r1", "c1")),
        ("warm", _payload("same", "r2", "c2")),
    ])
    assert result["exact_request_identity"] is True
    assert result["causal_admission"] is True
    assert result["cohorts"]["cold"]["command_clusters"] == {"c1": 1}
    assert result["cohorts"]["warm"]["reported_top1_mismatch_tokens"] == 1


def test_decode_state_reducer_rejects_mixed_request_identity():
    result = reduce_decode_state([
        ("cold", _payload("left", "r1", "c1")),
        ("warm", _payload("right", "r2", "c2")),
    ])
    assert result["exact_request_identity"] is False
    assert result["causal_admission"] is False
