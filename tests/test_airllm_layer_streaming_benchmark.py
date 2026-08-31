from experiments.paper6_6_airllm.run_layer_streaming_benchmark import (
    BenchmarkConfig,
    run_condition,
)


def test_layer_streaming_benchmark_decomposes_memory_and_releases_detail() -> None:
    cfg = BenchmarkConfig(layers=4, weight_layer_mib=1, compute_ms=0.1)
    row = run_condition(
        source_tokens=2048,
        profile="balanced",
        residency="layer_streamed",
        prefetch="none",
        tier="warm",
        repeats=1,
        cfg=cfg,
    )
    memory = row["memory_bytes"]
    assert row["consumer_layers"] == 2
    assert row["pra_bytes_read"] > 0
    assert memory["peak"] == (
        memory["weights_hot"]
        + memory["local_kv"]
        + memory["pra_hot"]
        + memory["temporary"]
        + memory["framework"]
    )
    assert memory["pra_hot"] < memory["pra_warm"]
