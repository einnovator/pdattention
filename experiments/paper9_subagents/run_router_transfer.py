"""Evaluate frozen Paper 9 descendant routers on unrelated repositories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from experiments.paper9_subagents.run_natural_repository_workload import QUESTIONS
from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.subagent_context import AgentStatus
from pra_hf.subagent_routing import (
    DescendantRecordRouter,
    DescendantRoutingExample,
    DescendantRoutingMode,
    RoutingCandidate,
)


def _candidate(path: Path, relative: str, prefix: str) -> RoutingCandidate:
    content = path.read_text(encoding="utf-8", errors="replace")
    payload = {"arguments": {"path": relative}, "output": content}
    return RoutingCandidate(
        ContextRecord(
            f"{prefix}:{relative}",
            RecordType.TOOL_RESPONSE,
            payload,
        ),
        json.dumps(payload, sort_keys=True),
        AgentStatus.STOPPED,
    )


def _development_router(repo: Path) -> tuple[DescendantRecordRouter, dict[str, str]]:
    candidates = tuple(
        _candidate(repo / question.path, question.path, "development")
        for question in QUESTIONS
    )
    by_path = {
        str(candidate.record.payload["arguments"]["path"]): candidate.record_uuid
        for candidate in candidates
    }
    examples = tuple(
        DescendantRoutingExample(
            question.query,
            candidates,
            frozenset({by_path[question.path]}),
        )
        for question in QUESTIONS
    )
    fingerprints = {
        question.path: hashlib.sha256((repo / question.path).read_bytes()).hexdigest()
        for question in QUESTIONS
    }
    return DescendantRecordRouter().fit(examples), fingerprints


def _git_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo_map(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("Repository mappings must use NAME=PATH.")
        result[name] = Path(path).resolve()
    return result


def evaluate(
    manifest: Mapping[str, object],
    repositories: Mapping[str, Path],
    development_repo: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    router, development_fingerprints = _development_router(development_repo)
    rows: list[dict[str, object]] = []
    fingerprints: dict[str, dict[str, object]] = {}
    modes = tuple(DescendantRoutingMode)
    for repository in manifest["repositories"]:
        repository_id = str(repository["repository_id"])
        repo = repositories[repository_id]
        expected_revision = str(repository["revision"])
        actual_revision = _git_revision(repo)
        if actual_revision != expected_revision:
            raise RuntimeError(
                f"{repository_id} revision {actual_revision} != frozen {expected_revision}"
            )
        candidates = tuple(
            _candidate(repo / relative, str(relative), repository_id)
            for relative in repository["candidate_paths"]
        )
        by_path = {
            str(candidate.record.payload["arguments"]["path"]): candidate.record_uuid
            for candidate in candidates
        }
        fingerprints[repository_id] = {
            "revision": actual_revision,
            "files": {
                str(relative): hashlib.sha256((repo / relative).read_bytes()).hexdigest()
                for relative in repository["candidate_paths"]
            },
        }
        for query in repository["queries"]:
            relevant_path = str(query["relevant_path"])
            relevant_ids = frozenset({by_path[relevant_path]})
            for mode in modes:
                route = router.route(
                    str(query["query"]),
                    candidates,
                    mode=mode,
                    top_k=1,
                    oracle_record_ids=relevant_ids,
                )
                rows.append(
                    {
                        "repository_id": repository_id,
                        "query_id": query["query_id"],
                        "query": query["query"],
                        "relevant_path": relevant_path,
                        "mode": mode.value,
                        "selected_path": str(
                            route.selected[0].record.payload["arguments"]["path"]
                        ),
                        "recall": route.recall(relevant_ids),
                        "precision": route.precision(relevant_ids),
                        "selected_tokens": route.selected_tokens,
                        "candidate_count": route.candidate_count,
                    }
                )
    summaries: dict[str, object] = {}
    for repository_id in (*repositories, "pooled"):
        subset = (
            rows
            if repository_id == "pooled"
            else [row for row in rows if row["repository_id"] == repository_id]
        )
        summaries[repository_id] = {}
        for mode in modes:
            selected = [row for row in subset if row["mode"] == mode.value]
            summaries[repository_id][mode.value] = {
                "queries": len(selected),
                "top1_recall": sum(float(row["recall"]) for row in selected)
                / max(1, len(selected)),
                "mean_selected_tokens": sum(int(row["selected_tokens"]) for row in selected)
                / max(1, len(selected)),
            }
    summary = {
        "protocol": "paper9-router-transfer-v1",
        "development_repository": manifest["development_repository"],
        "development_revision": _git_revision(development_repo),
        "development_file_sha256": development_fingerprints,
        "development_examples": len(QUESTIONS),
        "label_policy": "external labels frozen in manifest before evaluation",
        "repositories": fingerprints,
        "routing": summaries,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/paper9_subagents/benchmarks/router_transfer_v1.json"
        ),
    )
    parser.add_argument("--development-repo", type=Path, default=Path.cwd())
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/papers/shared/results/paper9_subagents/router_transfer_v1"
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    repositories = _repo_map(args.repo)
    expected = {str(row["repository_id"]) for row in manifest["repositories"]}
    if set(repositories) != expected:
        raise ValueError(f"Expected repository mappings for {sorted(expected)}")
    rows, summary = evaluate(manifest, repositories, args.development_repo.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "routing_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
