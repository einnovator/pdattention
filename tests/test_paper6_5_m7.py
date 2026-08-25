"""Contracts for the frozen M7 semantic-hard native-Q/K comparison."""

from experiments.paper6_5_tools.run_m7_semantic_native import (
    NATIVE_MODE_MAP,
    _best_validation_mode,
    _external_controls,
)


def _row(split: str, mode: str, correct: bool, mrr: float) -> dict[str, str]:
    return {
        "split": split,
        "mode": mode,
        "top1_correct": str(correct),
        "mrr": str(mrr),
    }


def test_validation_selection_does_not_consult_test_rows() -> None:
    rows = [
        _row("validation", "a", True, 1.0),
        _row("validation", "b", False, 0.5),
        _row("test", "a", False, 0.1),
        _row("test", "b", True, 1.0),
    ]

    assert _best_validation_mode(rows, ("a", "b")) == "a"


def test_external_controls_are_frozen_from_validation() -> None:
    rows = []
    for mode in (
        "P2_dictionary",
        "P3_tags",
        "P5_english_embedding",
        "P6_multilingual_embedding",
    ):
        rows.append(_row("validation", mode, mode in {"P3_tags", "P6_multilingual_embedding"}, 1.0))

    controls = _external_controls(rows)

    assert controls["external_dictionary"] == "P3_tags"
    assert controls["external_compact_embedding"] == "P6_multilingual_embedding"
    assert controls["external_hybrid_p8"] == "P8_lexical_dictionary_embedding"
    assert controls["external_staged_p10"] == "P10_staged_external"


def test_m7_native_modes_keep_mean_full_and_zero_shot_compressors() -> None:
    assert NATIVE_MODE_MAP == {
        "native_mean_k": "native_mean_k",
        "native_token_qk": "native_token_qk",
        "paper2_8_rank16_zero_shot": "paper2_8_rank16_ensemble",
        "paper2_8_rank8_centroids_zero_shot": "paper2_8_rank8_centroids",
    }
