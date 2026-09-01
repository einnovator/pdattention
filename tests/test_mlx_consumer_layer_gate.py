from experiments.paper6_2_mlx.run_consumer_layer_gate_calibration import (
    GateScore,
    choose_gate_candidate,
    layer_groups,
    removable_masks,
)


def test_layer_groups_cover_decoder_once() -> None:
    groups = layer_groups(28, 8)

    assert tuple(layer for group in groups for layer in group) == tuple(range(28))
    assert max(map(len, groups)) - min(map(len, groups)) <= 1


def test_removable_masks_support_noncontiguous_learned_placement() -> None:
    masks = removable_masks(tuple(range(8)), ((0, 1), (2, 3), (4, 5), (6, 7)))

    assert (0, 1, 4, 5, 6, 7) in masks
    assert (0, 1, 2, 3, 6, 7) in masks


def test_gate_candidate_obeys_fidelity_constraints_before_cost() -> None:
    candidates = (
        GateScore((0, 2), 0.02, 1.0, 0.5),
        GateScore((2,), 0.03, 0.8, 0.25),
        GateScore((0, 1, 2), 0.01, 1.0, 0.75),
    )

    chosen = choose_gate_candidate(
        candidates,
        max_abs_logprob_delta=0.05,
        min_first_token_agreement=0.9,
    )

    assert chosen == candidates[0]
