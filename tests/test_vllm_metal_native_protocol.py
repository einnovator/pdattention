from __future__ import annotations

from pra_vllm.metal_native import VLLMMetalBlockHandle


def test_block_handle_separates_logical_identity_from_physical_pages() -> None:
    handle = VLLMMetalBlockHandle(
        logical_key="resource-version-layer",
        block_ids=(7, 2),
        token_count=23,
        byte_count=4096,
    )
    assert handle.logical_key == "resource-version-layer"
    assert handle.block_ids == (7, 2)
    assert handle.token_count < len(handle.block_ids) * 16
