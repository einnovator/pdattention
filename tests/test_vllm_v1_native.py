from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from pra_vllm.v1_metadata import VLLMNativeBlockSet
from pra_vllm.v1_native import augment_paged_context, observe_prefill_rows


@dataclass
class _Group:
    block_tables: list[list[int]]
    block_size: int = 16
    slot_mapping: list[int] = field(default_factory=lambda: [32, 33])


@dataclass
class _Context:
    kv_groups: tuple[_Group, ...]
    context_lens: list[int]
    offsets: list[int]
    block_tables: list[list[int]]
    slot_mapping: list[int]
    kernel_metadata_cache: dict = field(default_factory=lambda: {(0, 16): object()})


def _selected(tokens: int = 32) -> VLLMNativeBlockSet:
    return VLLMNativeBlockSet(
        logical_keys=("resource-R",),
        block_ids_by_group=((100, 101),),
        selected_token_count=tokens,
        source_position_base=48,
        consumer_layers=(0, 1),
    )


def test_context_adds_native_pages_without_rewriting_scheduler_slots() -> None:
    group = _Group([[2, 3], [4]])
    context = _Context(
        kv_groups=(group,),
        context_lens=[18, 7],
        offsets=[17, 6],
        block_tables=group.block_tables,
        slot_mapping=group.slot_mapping,
    )
    original_slots = list(context.slot_mapping)

    augment_paged_context(context, ("A", "B"), {"A": _selected()})

    assert group.block_tables == [[100, 101, 2, 3], [4]]
    assert context.context_lens == [50, 7]
    assert context.offsets == [65, 6]
    assert context.slot_mapping == original_slots
    assert context.kernel_metadata_cache == {}


def test_context_rejects_unaligned_selected_tokens() -> None:
    group = _Group([[2]])
    context = _Context((group,), [1], [0], group.block_tables, group.slot_mapping)

    with pytest.raises(ValueError, match="page-aligned"):
        augment_paged_context(context, ("A",), {"A": _selected(23)})


def test_context_requires_one_request_identity_per_attention_row() -> None:
    group = _Group([[2], [3]])
    context = _Context((group,), [1, 1], [0, 0], group.block_tables, [])

    with pytest.raises(RuntimeError, match="request rows"):
        augment_paged_context(context, ("A",), {})


def test_prefill_observation_preserves_scheduler_owned_apc_geometry() -> None:
    rows = observe_prefill_rows(
        (
            SimpleNamespace(
                req_id="selected",
                start_pos=64,
                token_ids=[10, 11, 12],
                prompt_len=67,
            ),
            SimpleNamespace(
                req_id="ordinary",
                start_pos=32,
                token_ids=[13, 14],
                prompt_len=None,
            ),
        ),
        {"selected"},
    )

    assert rows[0].as_dict() == {
        "request_id": "selected",
        "scheduler_cache_start": 64,
        "scheduled_query_tokens": 3,
        "prompt_tokens": 67,
        "selected_registered": True,
    }
    assert rows[1].scheduler_cache_start == 32
    assert rows[1].prompt_tokens is None
    assert not rows[1].selected_registered
