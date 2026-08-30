from experiments.paper6_2_mlx.run_selective_kv_quantization import (
    quantization_profiles,
)


def test_selective_quantization_profiles_cover_components_and_layer_bands() -> None:
    profiles = quantization_profiles(16)

    assert profiles["all_keys"]["keys"] is True
    assert profiles["all_keys"]["values"] is False
    assert profiles["all_values"]["keys"] is False
    assert profiles["early_half_kv"]["layers"] == tuple(range(8))
    assert profiles["late_half_kv"]["layers"] == tuple(range(8, 16))
    assert profiles["late_quarter_kv"]["layers"] == tuple(range(12, 16))
