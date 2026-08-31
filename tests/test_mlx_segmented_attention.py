import pytest


def test_segmented_attention_matches_concatenated_reference() -> None:
    mx = pytest.importorskip("mlx.core")
    from pra_mlx.native import segmented_selected_attention

    query = mx.random.normal((1, 2, 1, 8)).astype(mx.float16)
    memory_k = mx.random.normal((1, 2, 5, 8)).astype(mx.float16)
    memory_v = mx.random.normal((1, 2, 5, 8)).astype(mx.float16)
    local_k = mx.random.normal((1, 2, 7, 8)).astype(mx.float16)
    local_v = mx.random.normal((1, 2, 7, 8)).astype(mx.float16)
    scale = 8**-0.5
    keys = mx.concatenate((memory_k, local_k), axis=2)
    values = mx.concatenate((memory_v, local_v), axis=2)
    reference = mx.softmax((query @ mx.swapaxes(keys, -1, -2)) * scale, axis=-1) @ values
    actual = segmented_selected_attention(
        query, memory_k, memory_v, local_k, local_v, scale=scale
    )
    mx.eval(reference, actual)

    assert float(mx.max(mx.abs(reference - actual)).item()) <= 2e-3


def test_segmented_attention_supports_gqa_and_causal_mask() -> None:
    mx = pytest.importorskip("mlx.core")
    from pra_mlx.native import segmented_selected_attention

    query = mx.random.normal((1, 4, 2, 8)).astype(mx.float16)
    memory_k = mx.random.normal((1, 2, 3, 8)).astype(mx.float16)
    memory_v = mx.random.normal((1, 2, 3, 8)).astype(mx.float16)
    local_k = mx.random.normal((1, 2, 2, 8)).astype(mx.float16)
    local_v = mx.random.normal((1, 2, 2, 8)).astype(mx.float16)
    repeated_k = mx.repeat(mx.concatenate((memory_k, local_k), axis=2), 2, axis=1)
    repeated_v = mx.repeat(mx.concatenate((memory_v, local_v), axis=2), 2, axis=1)
    mask = mx.array(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    scores = (query @ mx.swapaxes(repeated_k, -1, -2)) * (8**-0.5)
    reference = mx.softmax(mx.where(mask, scores, -1e9), axis=-1) @ repeated_v
    actual = segmented_selected_attention(
        query,
        memory_k,
        memory_v,
        local_k,
        local_v,
        scale=8**-0.5,
        mask=mask,
    )
    mx.eval(reference, actual)

    assert float(mx.max(mx.abs(reference - actual)).item()) <= 2e-3
