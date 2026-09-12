import pytest


def test_interval_consumer_matches_mlx_two_pass_dispatch_geometry() -> None:
    from pra_mlx.native import _native_mlx_vector_blocks

    # The admitted Qwen2.5-Coder-14B request on M4 Pro has 2,640 total K/V
    # tokens and 15 SIMD planes (GQA factor 5 x query width 3).
    assert _native_mlx_vector_blocks(2640, 15, "s", grouped_query=True) == 128
    assert _native_mlx_vector_blocks(2640, 15, "d", grouped_query=True) == 128
    assert _native_mlx_vector_blocks(1023, 15, "s", grouped_query=True) is None
    assert _native_mlx_vector_blocks(20_000, 15, "s", grouped_query=True) == 256
    assert _native_mlx_vector_blocks(70_000, 15, "d", grouped_query=True) == 1024
    assert _native_mlx_vector_blocks(4096, 4, "x", grouped_query=False) is None


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


def test_disjoint_segmented_attention_matches_one_dense_reference() -> None:
    mx = pytest.importorskip("mlx.core")
    from pra_mlx.native import disjoint_segmented_selected_attention

    query = mx.random.normal((1, 4, 2, 8)).astype(mx.float16)
    source_k = mx.random.normal((1, 2, 8, 8)).astype(mx.float16)
    source_v = mx.random.normal((1, 2, 8, 8)).astype(mx.float16)
    intervals = ((0, 3), (5, 7))
    memory_k = tuple(source_k[:, :, start:end, :] for start, end in intervals)
    memory_v = tuple(source_v[:, :, start:end, :] for start, end in intervals)
    local_k = mx.random.normal((1, 2, 2, 8)).astype(mx.float16)
    local_v = mx.random.normal((1, 2, 2, 8)).astype(mx.float16)
    dense_k = mx.repeat(mx.concatenate((*memory_k, local_k), axis=2), 2, axis=1)
    dense_v = mx.repeat(mx.concatenate((*memory_v, local_v), axis=2), 2, axis=1)
    mask = mx.array(
        [
            [True, True, True, True, True, True, False],
            [True, True, True, True, True, True, True],
        ]
    )
    scores = (query @ mx.swapaxes(dense_k, -1, -2)) * (8**-0.5)
    reference = mx.softmax(mx.where(mask, scores, -1e9), axis=-1) @ dense_v
    actual = disjoint_segmented_selected_attention(
        query,
        memory_k,
        memory_v,
        local_k,
        local_v,
        scale=8**-0.5,
        mask=mask,
        source_keys=source_k,
        source_values=source_v,
        source_intervals=intervals,
    )
    mx.eval(reference, actual)

    assert float(mx.max(mx.abs(reference - actual)).item()) <= 2e-3


def test_compiled_segmented_attention_matches_eager_with_growing_cache() -> None:
    mx = pytest.importorskip("mlx.core")
    from pra_mlx.native import (
        compiled_segmented_selected_attention,
        segmented_selected_attention,
    )

    memory_k = mx.random.normal((1, 2, 3, 8)).astype(mx.float16)
    memory_v = mx.random.normal((1, 2, 3, 8)).astype(mx.float16)
    for local_tokens, query_tokens in ((2, 3), (5, 1)):
        query = mx.random.normal((1, 4, query_tokens, 8)).astype(mx.float16)
        local_k = mx.random.normal((1, 2, local_tokens, 8)).astype(mx.float16)
        local_v = mx.random.normal((1, 2, local_tokens, 8)).astype(mx.float16)
        mask = mx.ones((1, memory_k.shape[2] + local_tokens), dtype=mx.bool_)
        eager = segmented_selected_attention(
            query,
            memory_k,
            memory_v,
            local_k,
            local_v,
            scale=8**-0.5,
            mask=mask,
        )
        compiled = compiled_segmented_selected_attention(
            query,
            memory_k,
            memory_v,
            local_k,
            local_v,
            scale=8**-0.5,
            mask=mask,
        )
        mx.eval(eager, compiled)
        assert float(mx.max(mx.abs(eager - compiled)).item()) <= 1e-5
