"""Execute the full graph-faceted root/successor request/reply surface.

The runner keeps the original Paper 2.6 58/74 validation/test partition.  Each
identity receives five facet construction modes, separate facet-count profiles,
five root methods, and five successor methods at a matched two-plus-two chunk
budget.  Root states are persisted once per initial action so callback models
can be retrained without repeating retrieval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_6_hybrid_pra.run_channel_geometry import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    WEIGHTS,
    _load_cases,
    _pieces,
)
from experiments.paper2_6_hybrid_pra.run_channel_selection import (  # noqa: E402
    ROOT_MODES,
    SUCCESSOR_CHANNELS,
    SUCCESSOR_MODES,
    _candidate_scores,
    _index_memory_bytes,
    _metrics,
    _rank,
    _root_gold,
    _score_gap,
    _successor_gold,
)
from experiments.paper2_6_hybrid_pra.run_study import _records  # noqa: E402
from experiments.paper2_7_query_graph.helpers import resolve_artifact  # noqa: E402
from pra_hf.adaptive_facets import (  # noqa: E402
    GraphFacetConfig,
    build_adaptive_query_facets,
)
from pra_hf.hybrid_discovery import TokenNativeIndex  # noqa: E402
from pra_hf.iterative import IterativeGistRouter  # noqa: E402
from pra_hf.root_callback import ROOT_STATE_FEATURE_NAMES, RootState  # noqa: E402


ROUTER_SEEDS = (1, 7, 21, 42, 87)
ROOT_METHODS = ("semantic", "exact", "bm25", "approx", "hybrid")
ROOT_MODE_NAMES = {
    "semantic": ROOT_MODES["gist"],
    "exact": ROOT_MODES["exact"],
    "bm25": ROOT_MODES["bm25"],
    "approx": ROOT_MODES["approx"],
    "hybrid": ROOT_MODES["hybrid"],
}
FACET_PROFILES = (
    ("global", "global", 1),
    ("structural", "syntactic", 2),
    ("structural", "syntactic", 4),
    ("multiscale", "multiscale", 2),
    ("multiscale", "multiscale", 4),
    ("graph", "graph", 2),
    ("graph", "graph", 4),
    ("structural_graph", "syntactic_graph", 2),
    ("structural_graph", "syntactic_graph", 4),
)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_local(path: Path, *siblings: str) -> Path:
    if path.exists():
        return path
    relative = path.relative_to(ROOT)
    for sibling in siblings:
        candidate = ROOT.parent / sibling / relative
        if candidate.exists():
            return candidate
    return path


def _query_state_map(args: argparse.Namespace) -> dict[tuple[str, str, str], dict]:
    mapping: dict[tuple[str, str, str], dict] = {}
    validation = torch.load(
        args.validation_query_states, map_location="cpu", weights_only=False
    )
    heldout = torch.load(
        args.heldout_query_states, map_location="cpu", weights_only=False
    )
    natural = torch.load(
        args.natural_features, map_location="cpu", weights_only=False, mmap=True
    )
    for row in (*validation, *heldout, *natural):
        partition = str(row.get("partition", "test"))
        key = (partition, str(row["dataset"]), str(row["example_id"]))
        mapping[key] = {
            "query_hidden_states": row["query_hidden_states"],
            "question_span": tuple(map(int, row["question_span"])),
            "prompt_input_ids": row["prompt_input_ids"],
        }
    return mapping


def _read_features(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (str(row["split"]), str(row["dataset"]), str(row["example_id"])): {
            key: float(value)
            for key, value in row.items()
            if key not in {"split", "dataset", "example_id"} and value not in {None, ""}
        }
        for row in rows
    }


def _semantic_scores(facets: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    facets = F.normalize(facets.float(), dim=-1, eps=1e-12)
    memory = F.normalize(memory.float(), dim=-1, eps=1e-12)
    return torch.einsum("fd,cd->fc", facets, memory).max(dim=0).values


def _entropy(scores: Mapping[str, float]) -> float:
    values = torch.tensor(
        [float(value) for value in scores.values() if math.isfinite(float(value))]
    )
    if len(values) <= 1:
        return 0.0
    probability = torch.softmax(values - values.max(), dim=0)
    entropy = float((-(probability * probability.clamp_min(1e-12).log())).sum())
    return entropy / math.log(len(values))


def _root_dispersion(index, selected: Sequence[str]) -> float:
    if len(selected) <= 1:
        return 0.0
    positions = [index.chunk_ids.index(identity) for identity in selected]
    values = F.normalize(index.gists[positions, 0].float(), dim=-1)
    similarities = values @ values.T
    distances = [
        1.0 - float(similarities[left, right])
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    ]
    return max(0.0, fmean(distances))


def _facet_agreement(facets: torch.Tensor, memory: torch.Tensor) -> float:
    if facets.shape[0] <= 1:
        return 1.0
    scores = F.normalize(facets.float(), dim=-1) @ F.normalize(memory.float(), dim=-1).T
    winners = scores.argmax(dim=-1).tolist()
    return max(Counter(winners).values()) / len(winners)


def _new_addresses(tokenizer, token_index, query_ids, selected: Sequence[str]):
    query_terms = _pieces(tokenizer, query_ids)
    records = {row.chunk_id: row for row in token_index.records}
    selected_rows = [records[value] for value in selected]
    root_terms = set().union(*(set(row.normalized_tokens) for row in selected_rows), set())
    cutoff = 0.75 * max(token_index.idf.values(), default=0.0)
    addresses = tuple(
        sorted(
            value
            for value in root_terms - query_terms
            if token_index.idf.get(value, 0.0) >= cutoff
        )
    )
    maximum = max(token_index.idf.values(), default=1.0)
    rarity = (
        fmean(token_index.idf.get(value, 0.0) for value in addresses) / max(maximum, 1e-12)
        if addresses
        else 0.0
    )
    entities = tuple(value for value in addresses if any(character.isdigit() for character in value))
    return addresses, entities, rarity


def _root_state(
    *,
    identity: tuple[str, str, str],
    query_features: Mapping[str, float],
    profile_name: str,
    facet_count: int,
    root_method: str,
    selected: Sequence[str],
    scores: Mapping[str, float],
    channel_rankings: Mapping[str, Sequence[str]],
    facets: torch.Tensor,
    index,
    token_index,
    tokenizer,
    query_ids,
    total_search_budget: int,
    total_kv_budget: int,
) -> RootState:
    ranking = _rank(dict(scores))
    ordered_scores = tuple(float(scores[value]) for value in selected)
    top1 = float(scores[ranking[0]]) if ranking else 0.0
    gap = _score_gap(dict(scores))
    winners = {values[0] for values in channel_rankings.values() if values}
    disagreement = float(max(0, len(winners) - 1))
    agreement = 1.0 - disagreement / max(len(channel_rankings) - 1, 1)
    addresses, entities, rarity = _new_addresses(
        tokenizer, token_index, query_ids, selected
    )
    positions = [index.chunk_ids.index(value) for value in selected]
    embedding = (
        index.gists[positions, 0].float().mean(dim=0)
        if positions
        else torch.zeros(index.gists.shape[-1])
    )
    entropy = max(0.0, min(1.0, _entropy(scores)))
    evidence_proxy = (1.0 - entropy) * float(torch.sigmoid(torch.tensor(gap)))
    return RootState(
        example_id=f"{identity[1]}:{identity[2]}",
        query_features=query_features,
        facet_mode=profile_name,
        facet_count=facet_count,
        root_method=root_method,
        root_ids=tuple(selected),
        root_scores=ordered_scores,
        root_top1_score=top1,
        root_score_gap=gap,
        candidate_entropy=entropy,
        channel_agreement=max(0.0, min(1.0, agreement)),
        channel_disagreement=disagreement,
        root_embedding=tuple(float(value) for value in embedding),
        new_entities=entities,
        new_addresses=addresses,
        address_count=len(addresses),
        address_rarity=max(0.0, min(1.0, rarity)),
        facet_agreement=_facet_agreement(facets.cpu(), index.gists[:, 0].cpu()),
        root_dispersion=_root_dispersion(index, selected),
        evidence_proxy=max(0.0, min(1.0, evidence_proxy)),
        searched_fraction=1.0,
        remaining_search_budget=max(0, total_search_budget - len(selected)),
        total_search_budget=total_search_budget,
        remaining_kv_budget=max(0, total_kv_budget - len(selected)),
        total_kv_budget=total_kv_budget,
    )


def _successor_rows(
    *,
    feature: Mapping[str, Any],
    index,
    token_index,
    tokenizer,
    query_ids: Sequence[int],
    root_method: str,
    root_scores: Mapping[str, float],
    selected_roots: Sequence[str],
    root_gold: set[str],
    all_gold: set[str],
    successor_k: int,
    traversal_cache: dict[tuple[str, ...], dict[str, dict[str, Any]]] | None = None,
    candidate_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    router = IterativeGistRouter(index)
    roots = set(selected_roots)
    successor_gold, mapping = _successor_gold(feature, index, roots & all_gold)
    id_to_row = {identity: row for row, identity in enumerate(index.chunk_ids)}
    frontier_rows = [id_to_row[identity] for identity in selected_roots]
    frontier_semantic, _ = router._scores(index.gists[frontier_rows, 0])
    root_token_ids = [token_index.records[row].token_ids for row in frontier_rows]
    root_ranking = _rank(dict(root_scores))
    cache_key = tuple(selected_roots)
    cache = traversal_cache if traversal_cache is not None else {}
    candidates = candidate_cache if candidate_cache is not None else {}
    if cache_key not in cache:
        cache[cache_key] = {}
        for successor_method in SUCCESSOR_CHANNELS:
            merged = {identity: float("-inf") for identity in index.chunk_ids}
            elapsed = 0.0
            for frontier_index, state_ids in enumerate(root_token_ids):
                frontier_id = selected_roots[frontier_index]
                candidate_key = (frontier_id, successor_method)
                if candidate_key not in candidates:
                    lexical_state = (
                        [*query_ids, *state_ids]
                        if successor_method in {"bm25_state", "hybrid_state"}
                        else list(state_ids)
                    )
                    started = time.perf_counter()
                    scores, _ = _candidate_scores(
                        index,
                        token_index,
                        tokenizer,
                        lexical_state,
                        frontier_semantic[frontier_index],
                        SUCCESSOR_MODES[successor_method],
                        hop=2,
                    )
                    candidates[candidate_key] = {
                        "scores": scores,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                    }
                scores = candidates[candidate_key]["scores"]
                elapsed += float(candidates[candidate_key]["latency_ms"])
                for identity, score in scores.items():
                    merged[identity] = max(merged[identity], score)
            cache[cache_key][successor_method] = {
                "ranking": _rank(merged, roots),
                "latency_ms": elapsed,
            }

    output = []
    for successor_method in SUCCESSOR_CHANNELS:
        successor_ranking = cache[cache_key][successor_method]["ranking"]
        elapsed = cache[cache_key][successor_method]["latency_ms"]
        selected_successors = successor_ranking[:successor_k]
        combined = list(dict.fromkeys([*selected_roots, *selected_successors]))
        combined_ranking = list(dict.fromkeys([*root_ranking, *successor_ranking]))
        overall = _metrics(combined, all_gold, combined_ranking)
        successor = _metrics(selected_successors, successor_gold, successor_ranking)
        output.append(
            {
                "successor_method": successor_method,
                "mapping_semantics": mapping,
                "evidence_recall": overall["recall"],
                "precision": overall["precision"],
                "mrr": overall["mrr"],
                "complete_recovery": overall["complete_recovery"],
                "successor_recall": successor["recall"],
                "successor_precision": successor["precision"],
                "root_recall": len(roots & root_gold) / max(len(root_gold), 1),
                "path_gain": len(set(selected_successors) & all_gold - roots) / max(len(all_gold), 1),
                "selected_root_ids": "|".join(selected_roots),
                "selected_successor_ids": "|".join(selected_successors),
                "selected_chunk_ids": "|".join(combined),
                "positive_chunk_ids": "|".join(sorted(all_gold)),
                "requested_chunks": len(combined),
                "root_comparisons": len(index.chunk_ids),
                "successor_comparisons": len(index.chunk_ids) * len(frontier_rows),
                "successor_index_lookups": sum(len(values) for values in root_token_ids),
                "successor_token_span_operations": (
                    sum(len(values) for values in root_token_ids)
                    * sum(len(row.token_ids) for row in token_index.records)
                    if successor_method != "native_semantic"
                    else 0
                ),
                "successor_latency_ms": elapsed,
                "index_memory_bytes": (
                    _index_memory_bytes(token_index)
                    if successor_method != "native_semantic"
                    else 0
                ),
            }
        )
    return output


def _mode_trees(hidden, token_texts, span, device, graph_config):
    output = {}
    for _, internal, _ in FACET_PROFILES:
        if internal in output:
            continue
        output[internal] = build_adaptive_query_facets(
            hidden.float().to(device),
            token_texts,
            mode=internal,
            support_span=span,
            coarse_partition_mode="clause",
            graph_config=graph_config,
        )
    return output


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only
    )
    query_map = _query_state_map(args)
    feature_map = _read_features(args.selector_features)
    loader_args = argparse.Namespace(
        seed=args.seed,
        cache_dir=args.cache_dir,
        paper2_feature_dir=args.paper2_feature_dir,
        natural_features=args.natural_features,
        musique_dev=args.musique_dev,
        twowiki_dev=args.twowiki_dev,
    )
    cases = _load_cases(loader_args)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    expected = Counter((feature["split"], feature["dataset"]) for feature, _ in cases)
    if args.max_cases is None and sum(expected.values()) != 132:
        raise AssertionError(f"Expected the frozen 132-identity cohort, received {expected}.")
    missing = [
        (feature["split"], feature["dataset"], feature["example_id"])
        for feature, _ in cases
        if (feature["split"], feature["dataset"], feature["example_id"]) not in query_map
    ]
    if missing:
        raise AssertionError(f"Missing contextual query states: {missing[:3]}")

    graph_config = GraphFacetConfig(
        similarity_mode="contextual",
        threshold=args.graph_threshold,
        top_k=args.graph_top_k,
        graph_policy="union",
        cluster_method="connected_components",
    )
    action_rows: list[dict[str, Any]] = []
    root_state_rows: list[dict[str, Any]] = []
    facet_rows: list[dict[str, Any]] = []
    state_cache: dict[str, dict[str, Any]] = {}
    for case_index, (feature, example) in enumerate(cases, start=1):
        identity = (
            str(feature["split"]),
            str(feature["dataset"]),
            str(feature["example_id"]),
        )
        query = query_map[identity]
        hidden = query["query_hidden_states"]
        prompt_ids = query["prompt_input_ids"]
        span = tuple(map(int, query["question_span"]))
        token_texts = [str(value) for value in tokenizer.convert_ids_to_tokens(prompt_ids.tolist())]
        trees = _mode_trees(hidden, token_texts, span, device, graph_config)
        index, all_gold = _records(feature, example)
        token_index = TokenNativeIndex.from_gist_index(index, tokenizer)
        query_ids = tokenizer(example["question"], add_special_tokens=False).input_ids
        root_gold = _root_gold(feature, index)
        memory = index.gists[:, 0].float().to(device)
        global_query = feature["queries"]["question_exp_h2.0"].float().to(device)
        query_features = feature_map[identity]
        successor_cache: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
        successor_candidate_cache: dict[tuple[str, str], dict[str, Any]] = {}

        for profile_name, internal_mode, requested_facet_count in FACET_PROFILES:
            tree = trees[internal_mode]
            local_facets = tree.scoring_facets.hidden[1:requested_facet_count]
            facets = torch.cat((global_query.unsqueeze(0), local_facets), dim=0)
            actual_facet_count = int(facets.shape[0])
            semantic = _semantic_scores(facets, memory).cpu()
            root_outcomes: dict[str, dict[str, Any]] = {}
            for root_method in ROOT_METHODS:
                started = time.perf_counter()
                scores, _ = _candidate_scores(
                    index,
                    token_index,
                    tokenizer,
                    query_ids,
                    semantic,
                    ROOT_MODE_NAMES[root_method],
                )
                root_ms = (time.perf_counter() - started) * 1000.0
                ranking = _rank(scores)
                root_outcomes[root_method] = {
                    "scores": scores,
                    "ranking": ranking,
                    "selected": ranking[: args.root_count],
                    "latency_ms": root_ms,
                }
            channel_rankings = {
                method: values["ranking"] for method, values in root_outcomes.items()
            }
            for root_method, root in root_outcomes.items():
                state = _root_state(
                    identity=identity,
                    query_features=query_features,
                    profile_name=profile_name,
                    facet_count=actual_facet_count,
                    root_method=root_method,
                    selected=root["selected"],
                    scores=root["scores"],
                    channel_rankings=channel_rankings,
                    facets=facets,
                    index=index,
                    token_index=token_index,
                    tokenizer=tokenizer,
                    query_ids=query_ids,
                    total_search_budget=args.root_count + args.successor_k,
                    total_kv_budget=args.kv_budget,
                )
                initial_id = f"{profile_name}.f{requested_facet_count}.{root_method}"
                cache_key = "|".join((*identity, initial_id))
                state_cache[cache_key] = {
                    "state": state,
                    "root_embedding": torch.tensor(state.root_embedding, dtype=torch.float16),
                }
                audit = state.audit_dict(include_embedding=False)
                root_state_rows.append(
                    {
                        "split": identity[0],
                        "dataset": identity[1],
                        "example_id": identity[2],
                        "initial_action": initial_id,
                        **{name: getattr(state, name) for name in ROOT_STATE_FEATURE_NAMES},
                        "root_ids": "|".join(state.root_ids),
                        "root_scores": "|".join(f"{value:.8g}" for value in state.root_scores),
                        "new_addresses": "|".join(state.new_addresses),
                        "root_embedding_width": audit["root_embedding"]["width"],
                        "root_embedding_norm": audit["root_embedding"]["norm"],
                    }
                )
                successors = _successor_rows(
                    feature=feature,
                    index=index,
                    token_index=token_index,
                    tokenizer=tokenizer,
                    query_ids=query_ids,
                    root_method=root_method,
                    root_scores=root["scores"],
                    selected_roots=root["selected"],
                    root_gold=root_gold,
                    all_gold=all_gold,
                    successor_k=args.successor_k,
                    traversal_cache=successor_cache,
                    candidate_cache=successor_candidate_cache,
                )
                for successor in successors:
                    action_rows.append(
                        {
                            "split": identity[0],
                            "dataset": identity[1],
                            "example_id": identity[2],
                            "initial_action": initial_id,
                            "complete_action": f"{initial_id}->{successor['successor_method']}",
                            "facet_mode": profile_name,
                            "requested_facet_count": requested_facet_count,
                            "actual_facet_count": actual_facet_count,
                            "root_method": root_method,
                            "successor_method": successor["successor_method"],
                            "root_count": args.root_count,
                            "successor_k": args.successor_k,
                            "hop_depth": 1,
                            "search_budget": args.root_count + args.successor_k,
                            "kv_budget": args.kv_budget,
                            **successor,
                            "root_latency_ms": root["latency_ms"],
                            "facet_latency_ms": tree.metrics.construction_latency_ms,
                            "graph_latency_ms": tree.metrics.graph_construction_ms
                            + tree.metrics.graph_clustering_ms,
                            "total_retrieval_latency_ms": tree.metrics.construction_latency_ms
                            + root["latency_ms"]
                            + successor["successor_latency_ms"],
                            "graph_calls": tree.metrics.graph_calls,
                            "graph_nodes": tree.metrics.graph_nodes,
                            "graph_edges": tree.metrics.graph_edges,
                            "graph_density": tree.metrics.graph_density,
                            "graph_pairwise_work": tree.metrics.pairwise_similarity_evaluations,
                            "facet_overlap": tree.metrics.mean_facet_overlap,
                            "logical_chunks": len(index.chunk_ids),
                            "active_fraction": successor["requested_chunks"] / max(len(index.chunk_ids), 1),
                            "requested_kv_tokens": sum(
                                index.records[index.chunk_ids.index(value)][1].token_count
                                for value in successor["selected_chunk_ids"].split("|")
                                if value
                            ),
                            "materialized_kv_tokens": 0,
                            "generation_performed": 0,
                        }
                    )
            facet_rows.append(
                {
                    "split": identity[0],
                    "dataset": identity[1],
                    "example_id": identity[2],
                    "facet_mode": profile_name,
                    "requested_facet_count": requested_facet_count,
                    "actual_facet_count": actual_facet_count,
                    **vars(tree.metrics),
                }
            )
        print(
            f"[request/reply surface {case_index}/{len(cases)}] "
            f"{identity[0]} {identity[1]} {identity[2]}",
            flush=True,
        )

    expected_rows = len(cases) * len(FACET_PROFILES) * len(ROOT_METHODS) * len(SUCCESSOR_CHANNELS)
    if len(action_rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} action rows, received {len(action_rows)}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "full_action_rows.csv", action_rows)
    _write_csv(args.output_dir / "root_state_rows.csv", root_state_rows)
    _write_csv(args.output_dir / "facet_construction_rows.csv", facet_rows)
    args.state_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_cache, args.state_cache)
    manifest = {
        "schema_version": "1.0",
        "study": "full_graph_faceted_request_reply_surface",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "cohort": {f"{split}:{dataset}": count for (split, dataset), count in sorted(expected.items())},
        "identities": len(cases),
        "validation_identities": sum(feature["split"] == "validation" for feature, _ in cases),
        "test_identities": sum(feature["split"] == "test" for feature, _ in cases),
        "facet_profiles": [
            {"name": name, "internal_mode": internal, "facet_count": count}
            for name, internal, count in FACET_PROFILES
        ],
        "root_methods": list(ROOT_METHODS),
        "successor_methods": list(SUCCESSOR_CHANNELS),
        "root_count": args.root_count,
        "successor_k": args.successor_k,
        "kv_budget": args.kv_budget,
        "action_rows": len(action_rows),
        "root_state_rows": len(root_state_rows),
        "state_cache": str(args.state_cache),
        "state_cache_sha256": _sha256(args.state_cache),
        "device": str(device),
        "materialization_performed": False,
        "generation_performed": False,
        "timing_scope": "synchronized only by surrounding experiment; prototype Python control flow",
    }
    (args.output_dir / "surface_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    output = ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra/root_callback/surface"
    inherited = ROOT.parent / "pdattention-iter-gist"
    primary = ROOT.parent / "pdattention"
    paper2 = primary / "docs/papers/shared/results/paper2_6_hybrid_pra/channel_selection"
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--root-count", type=int, default=2)
    parser.add_argument("--successor-k", type=int, default=2)
    parser.add_argument("--kv-budget", type=int, default=4)
    parser.add_argument("--graph-threshold", type=float, default=0.45)
    parser.add_argument("--graph-top-k", type=int, default=2)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument(
        "--paper2-feature-dir",
        type=Path,
        default=_resolve_local(
            ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
            "pdattention",
            "pdattention-iter-gist",
        ),
    )
    parser.add_argument(
        "--selector-features",
        type=Path,
        default=paper2 / "selector_observable_features.csv",
    )
    parser.add_argument(
        "--validation-query-states",
        type=Path,
        default=ROOT / "tmp/request_reply_callback/query_states_validation.pt",
    )
    parser.add_argument(
        "--heldout-query-states",
        type=Path,
        default=resolve_artifact(
            "docs/papers/shared/results/paper2_5_iterative_pra/query_entry_facets/query_entry_features.pt"
        ),
    )
    parser.add_argument(
        "--natural-features",
        type=Path,
        default=resolve_artifact(
            "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_features.pt"
        ),
    )
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=inherited / "data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument(
        "--twowiki-dev",
        type=Path,
        default=inherited / "data/.paper2_5_datasets/2wiki/dev.json",
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument(
        "--state-cache",
        type=Path,
        default=output / "root_states.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
