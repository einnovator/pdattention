"""Prepare frozen M9 tool/skill views, discovery, candidates, and costs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog
from data.declarative_skills import declarative_skill_catalog, skill_semantic_hard_queries
from data.semantic_hard_tools import semantic_hardness_queries
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_hf.agent_resources import DiscoveryRequest, PersistentResourceIndex, terms
from pra_hf.context_records import serialize_record, tool_definition_record
from pra_hf.progressive_disclosure import disclosure_cost
from pra_hf.semantic_resource_discovery import CompactEmbeddingEncoder


OUTPUT = ROOT / "docs/papers/shared/results/paper6_5_tools/progressive_disclosure"
AUTO_RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools/auto_discovery_ablation"
K_VALUES = (2, 4, 6, 8, 18)
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


@dataclass(frozen=True)
class ToolCase:
    query_id: str
    expected_arguments: Mapping[str, object]


TOOL_CASES = (
    ToolCase("get_user-h2-en-1", {"user_id": "u17"}),
    ToolCase("update_user-h4-en-1", {"user_id": "u17", "status": "active"}),
    ToolCase("search_document-h1-en-1", {"title": "PRA Notes"}),
    ToolCase("export_document-h2-en-1", {"document_id": "d42", "format": "pdf"}),
    ToolCase("create_issue-h1-en-1", {"repository_id": "repo9", "title": "Routing audit"}),
    ToolCase("update_issue-h2-en-1", {"issue_id": "issue-4", "status": "open"}),
    ToolCase("create_report-h2-en-1", {"artifact_id": "artifact-d42-pdf", "title": "PRA Digest"}),
    ToolCase("archive_report-h2-en-1", {"report_id": "report-7"}),
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rank(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=lambda index: (-float(values[index]), index)))


def _minmax(values: torch.Tensor) -> torch.Tensor:
    low = values.min()
    high = values.max()
    return (values - low) / (high - low).clamp_min(1e-8)


def _skill_auto_score(query: str, resource) -> float:
    query_terms = set(terms(query))
    evidence = set(resource.keywords) | set(resource.tags) | set(resource.auto_tags)
    evidence |= set(terms(" ".join(resource.aliases)))
    return len(query_terms & evidence) / max(len(query_terms), 1)


def _skill_discovery(skills, resources, encoder) -> tuple[list[dict], dict, dict]:
    queries = skill_semantic_hard_queries()
    texts = [
        f"{skill.name.replace('_', ' ')}. {skill.description} {skill.when_to_use}"
        for skill in skills
    ]
    embeddings = encoder.encode(texts)
    query_vectors = encoder.encode([query.query for query in queries], query=True)
    lexical = PersistentResourceIndex(resources)
    channels = {}
    for row_index, query in enumerate(queries):
        scores = lexical.score(
            DiscoveryRequest(query.query, tenant_id="paper6_5", top_k=len(resources)),
            channels=("index",),
        )
        bm25_by_uri = {row.uri: row.index for row in scores}
        bm25 = torch.tensor([bm25_by_uri.get(resource.uri, 0.0) for resource in resources])
        automatic = torch.tensor([_skill_auto_score(query.query, resource) for resource in resources])
        embedding = _minmax(embeddings @ query_vectors[row_index])
        channels[query.query_id] = {
            "bm25": bm25,
            "auto_semantic": automatic,
            "embedding": embedding,
        }

    validation = tuple(query for query in queries if query.split == "validation")
    weights = []
    for left in (0.0, 0.25, 0.5, 0.75, 1.0):
        for middle in (0.0, 0.25, 0.5, 0.75, 1.0):
            right = 1.0 - left - middle
            if right >= 0:
                weights.append((left, middle, right))
    target_index = {skill.name: index for index, skill in enumerate(skills)}
    selected_weights = max(
        weights,
        key=lambda value: (
            sum(
                _rank(
                    channels[query.query_id]["bm25"] * value[0]
                    + channels[query.query_id]["auto_semantic"] * value[1]
                    + channels[query.query_id]["embedding"] * value[2]
                )[0]
                == target_index[query.target_skill]
                for query in validation
            ),
            -value[2],
            value[1],
        ),
    )
    rows = []
    candidate_sets = []
    for query in queries:
        values = channels[query.query_id]
        fused = (
            values["bm25"] * selected_weights[0]
            + values["auto_semantic"] * selected_weights[1]
            + values["embedding"] * selected_weights[2]
        )
        policies = {**values, "fused": fused}
        for policy, scores in policies.items():
            order = _rank(scores)
            rank = order.index(target_index[query.target_skill]) + 1
            rows.append({
                "query_id": query.query_id,
                "split": query.split,
                "family": query.family,
                "policy": policy,
                "target_skill": query.target_skill,
                "rank": rank,
                "top1": int(rank == 1),
                "recall_at_2": int(rank <= 2),
                "recall_at_4": int(rank <= 4),
                "recall_at_6": int(rank <= 6),
                "recall_at_8": int(rank <= 8),
            })
        order = _rank(fused)
        for budget in (2, 4, 6, 8, len(skills)):
            candidate_sets.append({
                "resource_type": "skill",
                "query_id": query.query_id,
                "max_candidates": budget,
                "candidate_names": [skills[index].name for index in order[:budget]],
                "target_name": query.target_skill,
            })
    return rows, {"bm25": selected_weights[0], "auto_semantic": selected_weights[1], "embedding": selected_weights[2]}, {"rows": candidate_sets}


def _tool_candidates() -> tuple[list[dict], list[dict]]:
    query_by_id = {query.query_id: query for query in semantic_hardness_queries()}
    case_by_id = {case.query_id: case for case in TOOL_CASES}
    with (AUTO_RESULTS / "union_candidate_rows.csv").open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    selected = {
        (row["query_id"], int(row["max_candidates"])): row
        for row in source
        if row["split"] == "test"
        and row["strategy"] == "fused_score"
        and row["query_id"] in case_by_id
        and int(row["max_candidates"]) in {2, 4, 6, 8}
    }
    resources = realistic_tool_catalog()
    all_names = [resource.name for resource in resources]
    cases = []
    candidates = []
    for case in TOOL_CASES:
        query = query_by_id[case.query_id]
        cases.append({
            "query_id": case.query_id,
            "query": "\n".join(value for value in (query.context, query.query) if value),
            "target_name": query.required_tool,
            "expected_arguments": dict(case.expected_arguments),
            "hardness_level": query.hardness_level,
        })
        for budget in K_VALUES:
            names = all_names if budget == len(resources) else selected[(case.query_id, budget)]["candidate_names"].split("|")
            candidates.append({
                "resource_type": "tool",
                "query_id": case.query_id,
                "max_candidates": budget,
                "candidate_names": names,
                "target_name": query.required_tool,
            })
    return cases, candidates


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    token_counter = lambda text: _token_count(tokenizer, text)

    tool_resources = realistic_tool_catalog()
    tool_records = {resource.name: tool_definition_record(resource) for resource in tool_resources}
    tool_cases, tool_candidates = _tool_candidates()
    tool_selection_rows = []
    schema_sizes = []
    for resource in tool_resources:
        record = tool_records[resource.name]
        selection_text = serialize_record(record, view="selection")
        full_text = serialize_record(record, view="full")
        selection_tokens = token_counter(selection_text)
        full_tokens = token_counter(full_text)
        tool_selection_rows.append({
            "record_id": record.record_id,
            "name": resource.name,
            "selection_payload": record.materialize("selection").payload,
            "selection_tokens": selection_tokens,
            "full_tokens": full_tokens,
        })
        schema_sizes.append({
            "tool_name": resource.name,
            "selection_tokens": selection_tokens,
            "full_schema_tokens": full_tokens,
            "selection_full_ratio": selection_tokens / max(full_tokens, 1),
            "parameter_count": len(json.loads(resource.content)["parameters"]["properties"]),
            "required_parameter_count": len(json.loads(resource.content)["parameters"].get("required", ())),
            "side_effect": resource.side_effect_class.value,
        })
    _write_jsonl(args.output / "tool_selection_view_cases.jsonl", tool_selection_rows)
    _write_csv(args.output / "tool_schema_size_distribution.csv", schema_sizes)

    skills = declarative_skill_catalog()
    skill_resources = tuple(skill.to_agent_resource() for skill in skills)
    _write_jsonl(args.output / "skill_catalog.jsonl", [skill.full_payload for skill in skills])
    _write_jsonl(args.output / "skill_semantic_hard_queries.jsonl", [asdict(row) for row in skill_semantic_hard_queries()])
    encoder = CompactEmbeddingEncoder(
        str(args.model_root / "bge-small-en-v1.5"),
        revision=BGE_REVISION,
        device=args.device,
        query_prefix="Represent this sentence for searching relevant passages: ",
        pooling="cls",
    )
    skill_rows, skill_weights, skill_candidates = _skill_discovery(skills, skill_resources, encoder)
    _write_csv(args.output / "skill_discovery_results.csv", skill_rows)

    all_candidates = {"rows": [*tool_candidates, *skill_candidates["rows"]]}
    (args.output / "progressive_candidate_sets.json").write_text(
        json.dumps(all_candidates, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output / "tool_progressive_cases.json").write_text(
        json.dumps({"rows": tool_cases}, indent=2, sort_keys=True), encoding="utf-8"
    )

    costs = []
    skill_record_by_name = {skill.name: skill.to_context_record() for skill in skills}
    candidates_by = {
        (row["resource_type"], row["query_id"], row["max_candidates"]): row
        for row in all_candidates["rows"]
    }
    for resource_type, cases in (
        ("tool", tool_cases),
        ("skill", [asdict(row) for row in skill_semantic_hard_queries() if row.split == "test"]),
    ):
        selected_cases = cases if resource_type == "tool" else cases[:8]
        record_by_name = tool_records if resource_type == "tool" else skill_record_by_name
        for case in selected_cases:
            query_id = case["query_id"]
            target_name = case.get("target_name", case.get("target_skill"))
            for budget in K_VALUES if resource_type == "tool" else (2, 4, 6, 8, len(skills)):
                candidate = candidates_by[(resource_type, query_id, budget)]
                records = tuple(record_by_name[name] for name in candidate["candidate_names"])
                selected_id = record_by_name[target_name].record_id if target_name in candidate["candidate_names"] else records[0].record_id
                value = disclosure_cost(records, selected_record_id=selected_id, token_counter=token_counter)
                costs.append({
                    "resource_type": resource_type,
                    "query_id": query_id,
                    "max_candidates": budget,
                    "target_in_candidates": int(target_name in candidate["candidate_names"]),
                    **asdict(value),
                })
    _write_csv(args.output / "capability_disclosure_costs.csv", costs)

    full_tool_sizes = sorted(row["full_schema_tokens"] for row in schema_sizes)
    skill_view_sizes = [
        (
            token_counter(serialize_record(skill.to_context_record(), view="selection")),
            token_counter(serialize_record(skill.to_context_record(), view="full")),
        )
        for skill in skills
    ]
    full_skill_sizes = sorted(full for _, full in skill_view_sizes)
    manifest = {
        "schema_version": "1.0",
        "record_views": ["selection", "full"],
        "transition": "selection_to_full_between_model_invocations",
        "callback_behavior": "not_implemented",
        "paper7_dynamic_expansion": "not_implemented",
        "tool_cases": len(tool_cases),
        "skill_catalog_size": len(skills),
        "skill_queries": len(skill_semantic_hard_queries()),
        "candidate_budgets": list(K_VALUES),
        "skill_fusion_weights": skill_weights,
        "tool_schema_tokens": {
            "median": statistics.median(full_tool_sizes),
            "p90": full_tool_sizes[int(.9 * (len(full_tool_sizes) - 1))],
            "p95": full_tool_sizes[int(.95 * (len(full_tool_sizes) - 1))],
            "max": max(full_tool_sizes),
        },
        "skill_full_tokens": {
            "median": statistics.median(full_skill_sizes),
            "p90": full_skill_sizes[int(.9 * (len(full_skill_sizes) - 1))],
            "p95": full_skill_sizes[int(.95 * (len(full_skill_sizes) - 1))],
            "max": max(full_skill_sizes),
        },
        "skill_selection_full_ratio_median": statistics.median(
            selected / full for selected, full in skill_view_sizes
        ),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_revision": BGE_REVISION,
        "runtime": runtime_metadata(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "record_view_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / ".hf_models")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
