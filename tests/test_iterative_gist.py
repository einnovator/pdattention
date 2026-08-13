"""Scientific and integration gates for bounded associative gist closure."""

from __future__ import annotations

import pytest
import torch

from pra_hf.iterative import (
    GistIndex,
    HierarchicalGistIndex,
    HierarchicalLocalGistRouter,
    IterativeGistRouter,
    IterativeRoutingConfig,
)
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    ReferenceChunkMemory,
)


def _entry(
    uri: str,
    vectors: list[list[float]],
    *,
    query_vectors: list[list[float]] | None = None,
    layer: int = 0,
) -> PRACacheEntry:
    chunks = []
    for offset, vector in enumerate(vectors):
        gist = torch.tensor([vector], dtype=torch.float32)
        kv = gist.view(1, 1, 1, -1)
        chunks.append(
            ReferenceChunkMemory(
                chunk_id=f"{uri}:{offset}",
                source_uri=uri,
                token_start=offset,
                token_end=offset + 1,
                token_kv=LayerKV(kv.clone(), kv.clone()),
                routing_gist=ChunkRoutingGist(
                    k=gist,
                    query_k=(
                        torch.tensor([query_vectors[offset]], dtype=torch.float32)
                        if query_vectors is not None
                        else None
                    ),
                ),
            )
        )
    return PRACacheEntry(uri=uri, text=uri, layer_memory={layer: LayerReferenceMemory(chunks)})


def _router(*, device="cpu", layer=0):
    # Root e0 reaches A; A then reaches B through their shared e1 component.
    entries = [
        _entry("A", [[0.8, 0.6, 0.0]], layer=layer),
        _entry("B", [[0.0, 1.0, 0.0]], layer=layer),
        _entry("X", [[0.7, 0.0, 0.7141428]], layer=layer),
        _entry("Y", [[0.69, 0.0, -0.723809]], layer=layer),
    ]
    return IterativeGistRouter(GistIndex.from_entries(entries, layer, device=device))


def test_depth_one_matches_one_shot_top_b():
    router = _router()
    result = router.route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(depth=1, branch_top_k=3, beam_size=3, max_unique_chunks=3),
    )
    expected = sorted(range(4), key=lambda index: (-result.direct_scores[index], index))[:3]
    assert list(result.selected_indices) == expected
    assert [node.hop for node in result.graph.nodes] == [1, 1, 1]


def test_iterative_closure_recovers_indirect_node_at_matched_budget():
    router = _router()
    root = torch.tensor([1.0, 0.0, 0.0])
    one_shot = router.route(
        root,
        IterativeRoutingConfig(depth=1, branch_top_k=2, beam_size=2, max_unique_chunks=2),
    )
    iterative = router.route(
        root,
        IterativeRoutingConfig(
            depth=2,
            branch_top_k=1,
            beam_size=1,
            max_unique_chunks=2,
            root_anchor_alpha=0.0,
        ),
    )
    ids = router.index.chunk_ids
    assert ids.index("B:0") not in one_shot.selected_indices
    assert [ids[index] for index in iterative.selected_indices] == ["A:0", "B:0"]
    assert iterative.graph.stop_reason == "unique_budget"


def test_query_projected_frontier_uses_aligned_companion_gist():
    entries = [
        _entry("A", [[0.8, 0.6, 0.0]], query_vectors=[[0.0, 1.0, 0.0]]),
        _entry("B", [[0.0, 1.0, 0.0]], query_vectors=[[1.0, 0.0, 0.0]]),
        _entry("X", [[0.7, 0.0, 0.7141428]], query_vectors=[[0.0, 0.0, 1.0]]),
    ]
    router = IterativeGistRouter(GistIndex.from_entries(entries, 0))
    result = router.route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(
            depth=2,
            branch_top_k=1,
            beam_size=1,
            max_unique_chunks=2,
            root_anchor_alpha=0.0,
            frontier_projection="query",
        ),
    )
    assert [router.index.chunk_ids[index] for index in result.selected_indices] == [
        "A:0",
        "B:0",
    ]
    assert result.graph.schema_version == "2.0"
    assert result.graph.edges[-1].projection_type == "query_to_memory"


def test_query_projected_frontier_requires_companion_gists():
    with pytest.raises(ValueError, match="query_gists"):
        _router().route(
            torch.tensor([1.0, 0.0, 0.0]),
            IterativeRoutingConfig(frontier_projection="query"),
        )


def test_local_bridge_succeeds_when_parent_mean_hides_association():
    # The root selects parent A through e0. A's whole-parent query points away
    # from B, while A's activated local query points exactly to B's local key.
    index = HierarchicalGistIndex(
        parent_ids=("A", "B", "X"),
        parent_spans=((0, 256), (256, 512), (512, 768)),
        parent_memory_gists=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.7, 0.0, 0.7141428]]
        ),
        parent_query_gists=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        local_spans=((0, 32), (32, 64), (256, 288), (288, 320), (512, 544)),
        local_parent_indices=torch.tensor([0, 0, 1, 1, 2]),
        local_memory_gists=torch.tensor(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [0.7, 0.0, 0.7141428]]
        ),
        local_query_gists=torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    )
    result = HierarchicalLocalGistRouter(index).route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(
            depth=2,
            branch_top_k=1,
            beam_size=1,
            max_unique_chunks=2,
            root_anchor_alpha=0.0,
            frontier_projection="query",
        ),
        evidence_parent_ids={"A", "B"},
    )
    assert result.selected_indices == (0, 1)
    assert result.graph.nodes[-1].parent_chunk_id == "B"
    assert result.graph.nodes[-1].resolution_level == "local"
    assert result.graph.costs["unique_parents_selected"] == 2
    assert result.graph.costs["bridge_locality_score"] > 0
    edge = result.graph.to_dict()["edges"][-1]
    assert edge["source_node"] == edge["source"]
    assert edge["target_node"] == edge["target"]


def test_hierarchical_index_rejects_invalid_parent_mapping():
    with pytest.raises(ValueError, match="valid parent"):
        HierarchicalGistIndex(
            parent_ids=("A",),
            parent_spans=((0, 32),),
            parent_memory_gists=torch.ones(1, 2),
            parent_query_gists=torch.ones(1, 2),
            local_spans=((0, 16),),
            local_parent_indices=torch.tensor([1]),
            local_memory_gists=torch.ones(1, 2),
            local_query_gists=torch.ones(1, 2),
        )


def test_hierarchical_cache_index_deduplicates_parent_payloads():
    entry = _entry(
        "doc",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        query_vectors=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    )
    for chunk in entry.layer_memory[0].chunks:
        chunk.routing_gist.metadata["segment_token_spans"] = [[0, 1]]
    index = HierarchicalGistIndex.from_entries([entry], 0)
    router = HierarchicalLocalGistRouter(index)
    result = router.route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(
            depth=2, branch_top_k=1, beam_size=1, max_unique_chunks=2,
            root_anchor_alpha=0.0, frontier_projection="query",
        ),
    )
    selected = router.selected_chunks(result)
    assert len(selected) == 2
    assert len({hit.chunk.chunk_id for hit in selected}) == 2
    assert all(hit.metadata["selection_policy"] == "local_iterative_closure" for hit in selected)


def test_cycles_duplicates_and_budget_do_not_repeat_nodes():
    router = _router()
    result = router.route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(depth=10, branch_top_k=4, beam_size=4, max_unique_chunks=4),
    )
    assert len(result.selected_indices) == len(set(result.selected_indices)) == 4
    assert result.graph.costs["visited_nodes"] == 4
    assert result.graph.costs["branch_entropy"] >= 0.0
    assert result.graph.stop_reason == "unique_budget"


@pytest.mark.parametrize("frontier_mode", ["direct", "residual", "mean", "weighted_mean"])
@pytest.mark.parametrize("path_mode", ["product", "logsum", "last", "min", "mean", "direct"])
def test_level_and_path_variants_emit_versioned_graph(frontier_mode, path_mode):
    result = _router().route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(
            depth=2,
            branch_top_k=2,
            beam_size=2,
            max_unique_chunks=3,
            frontier_mode=frontier_mode,
            path_score_mode=path_mode,
        ),
        example_id="example-1",
        evidence_chunk_ids={"B:0"},
    )
    graph = result.graph.to_dict()
    assert graph["schema_version"] == "2.0"
    assert graph["example_id"] == "example-1"
    assert graph["nodes"] and graph["edges"]
    assert all("path_score" in node and "parent_ids" in node for node in graph["nodes"])


def test_zero_limits_and_layer_locality():
    router = _router(layer=3)
    result = router.route(
        torch.tensor([1.0, 0.0, 0.0]),
        IterativeRoutingConfig(depth=0, branch_top_k=2, beam_size=2, max_unique_chunks=2),
    )
    assert result.selected_indices == ()
    assert result.graph.layer_id == 3
    assert result.graph.stop_reason == "zero_limit"


def test_batch_rows_are_independent_and_deterministic():
    router = _router()
    config = IterativeRoutingConfig(depth=2, branch_top_k=1, beam_size=1, max_unique_chunks=2)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    first = router.route_batch(queries, config)
    second = router.route_batch(queries, config)
    assert [row.selected_indices for row in first] == [row.selected_indices for row in second]
    assert first[0].selected_indices != first[1].selected_indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_gpu_identity_parity():
    config = IterativeRoutingConfig(depth=3, branch_top_k=2, beam_size=2, max_unique_chunks=4)
    root = torch.tensor([1.0, 0.0, 0.0])
    cpu = _router().route(root, config)
    gpu = _router(device="cuda").route(root.cuda(), config)
    assert gpu.selected_indices == cpu.selected_indices
