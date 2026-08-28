from experiments.paper6_2_mlx.run_rotating_archive import EXPECTED, KV_SIZES, SEEDS


def test_rotating_archive_protocol_has_five_seeds_and_bounded_cache_ladder() -> None:
    assert SEEDS == (11, 23, 37, 53, 71)
    assert KV_SIZES == (64, 128, 256, 512)
    assert EXPECTED == "7391"
