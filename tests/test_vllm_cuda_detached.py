from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from pra_vllm.cuda_detached import (
    DetachedCommand,
    apply_detached_query_geometry,
    expand_worker_kv_cache_config,
    prepend_detached_blocks,
)


@dataclass
class _TensorSpec:
    size: int


@dataclass
class _CacheConfig:
    num_blocks: int
    kv_cache_tensors: list[_TensorSpec]


def test_worker_reserve_extends_physical_tensors_without_mutating_scheduler() -> None:
    scheduler = _CacheConfig(100, [_TensorSpec(1000), _TensorSpec(2000)])

    worker, layout = expand_worker_kv_cache_config(scheduler, 8)

    assert scheduler.num_blocks == 100
    assert [tensor.size for tensor in scheduler.kv_cache_tensors] == [1000, 2000]
    assert worker.num_blocks == 108
    assert [tensor.size for tensor in worker.kv_cache_tensors] == [1080, 2160]
    assert layout.block_ids == tuple(range(100, 108))


def test_query_geometry_offsets_rope_but_not_local_slot_positions() -> None:
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["ordinary", "selected"]),
        positions=torch.tensor([0, 1, 0, 1, 2], dtype=torch.int64),
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        optimistic_seq_lens_cpu=torch.tensor([2, 3], dtype=torch.int32),
    )

    apply_detached_query_geometry(
        runner,
        [2, 3],
        {"selected": DetachedCommand("selected", source_tokens=32)},
    )

    assert runner.positions.tolist() == [0, 1, 32, 33, 34]
    assert runner.seq_lens.tolist() == [2, 35]
    assert runner.optimistic_seq_lens_cpu.tolist() == [2, 35]


def test_detached_blocks_are_prepended_without_changing_query_slot_mapping() -> None:
    metadata = SimpleNamespace(
        block_table=torch.tensor(
            [[4, 5, -1, -1, -1, -1], [7, 8, -1, -1, -1, -1]],
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor([24, 24], dtype=torch.int32),
        slot_mapping=torch.tensor([64, 65, 112, 113], dtype=torch.int64),
    )
    original_slots = metadata.slot_mapping.clone()

    prepend_detached_blocks(
        {"layer.0": metadata, "layer.1": metadata},
        row=0,
        block_ids=[100, 101],
        source_tokens=16,
        block_size=8,
    )

    assert metadata.block_table[0, :3].tolist() == [100, 101, 4]
    assert torch.equal(metadata.slot_mapping, original_slots)
    assert metadata.block_table[1, :2].tolist() == [7, 8]
