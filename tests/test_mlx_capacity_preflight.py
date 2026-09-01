from experiments.paper6_2_mlx.run_capacity_preflight import required_bytes


def test_capacity_preflight_blocks_swap_dominated_model() -> None:
    decision = required_bytes(
        checkpoint_bytes=17 * 2**30,
        physical_memory_bytes=16 * 2**30,
        selected_kv_bytes=128 * 2**20,
        reserve_fraction=0.25,
        minimum_reserve_bytes=4 * 2**30,
    )

    assert decision["status"] == "NOT_RUN_CAPACITY_GATE"
    assert decision["required_bytes"] > decision["physical_memory_bytes"]


def test_capacity_preflight_accepts_model_with_workspace_headroom() -> None:
    decision = required_bytes(
        checkpoint_bytes=8 * 2**30,
        physical_memory_bytes=16 * 2**30,
        selected_kv_bytes=128 * 2**20,
        reserve_fraction=0.25,
        minimum_reserve_bytes=4 * 2**30,
    )

    assert decision["status"] == "ELIGIBLE"
