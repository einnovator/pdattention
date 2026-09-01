from types import SimpleNamespace

from experiments.paper6_vllm.run_v1_apc_concurrency import _output_row


def test_output_row_reports_queue_ttft_tpot_and_total_latency():
    output = SimpleNamespace(
        request_id="request-1",
        num_cached_tokens=32,
        num_cache_creation_tokens=16,
        metrics=SimpleNamespace(
            arrival_time=10.0,
            first_scheduled_time=10.01,
            first_token_time=10.05,
            finished_time=10.11,
        ),
        outputs=[SimpleNamespace(text="7391", token_ids=[7, 3, 9, 1])],
    )

    row = _output_row(output)

    assert row["exact_recovery"] is True
    assert abs(row["queue_ms"] - 10.0) < 1e-9
    assert abs(row["ttft_ms"] - 50.0) < 1e-9
    assert abs(row["request_latency_ms"] - 110.0) < 1e-9
    assert abs(row["tpot_ms"] - 20.0) < 1e-9
