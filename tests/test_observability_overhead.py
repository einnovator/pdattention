from experiments.paper4_5_runtime.run_observability_overhead import run_mode


def test_disabled_overhead_path_never_evaluates_attributes() -> None:
    off = run_mode("off", 10, 1)
    metrics = run_mode("metrics", 10, 1)
    sampled = run_mode("sampled_otel", 10, 1)

    assert off["attributes_evaluated"] == 0
    assert metrics["attributes_evaluated"] == 0
    assert sampled["attributes_evaluated"] == 10
