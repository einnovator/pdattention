"""Freeze router-selected records and verification candidates for Paper 9.

The resulting manifest is self-contained: every model and presentation arm
receives byte-identical selected evidence, while the mandatory-verification
arm can inspect only the same pinned candidate snapshots seen by the router.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def _git_text(repo: Path, revision: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative_path}"],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(
    benchmark_path: Path,
    routing_rows_path: Path,
    transfer_summary_path: Path,
    repositories: dict[str, Path],
) -> dict[str, object]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    transfer = json.loads(transfer_summary_path.read_text(encoding="utf-8"))
    with routing_rows_path.open(encoding="utf-8", newline="") as stream:
        selected = {
            (row["repository_id"], row["query_id"]): row
            for row in csv.DictReader(stream)
            if row["mode"] == "fielded_bm25"
        }

    frozen_repositories: dict[str, object] = {}
    cases: list[dict[str, object]] = []
    for repository in benchmark["repositories"]:
        repository_id = repository["repository_id"]
        revision = repository["revision"]
        repo_path = repositories[repository_id]
        expected_hashes = transfer["repositories"][repository_id]["files"]
        files: dict[str, dict[str, object]] = {}
        for relative_path in repository["candidate_paths"]:
            text = _git_text(repo_path, revision, relative_path)
            digest = _sha256(text)
            files[relative_path] = {
                "sha256": digest,
                # router_transfer_v1 hashed checkout bytes. That value can be
                # CRLF-sensitive on Windows, so retain it as source provenance
                # while using the normalized logical-text hash for this
                # cross-host manifest.
                "router_transfer_checkout_sha256": expected_hashes[relative_path],
                "text": text,
            }
        frozen_repositories[repository_id] = {
            "revision": revision,
            "files": files,
        }
        for query in repository["queries"]:
            key = (repository_id, query["query_id"])
            route = selected[key]
            selected_path = route["selected_path"]
            cases.append(
                {
                    "repository_id": repository_id,
                    "revision": revision,
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "relevant_path": query["relevant_path"],
                    "selected_path": selected_path,
                    "selected_sha256": files[selected_path]["sha256"],
                    "selection_correct": selected_path == query["relevant_path"],
                    "router": "fielded_bm25",
                    "router_selected_tokens": int(route["selected_tokens"]),
                }
            )

    return {
        "schema_version": "paper9.consumption_factorial_manifest.v1",
        "source_protocol": transfer["protocol"],
        "selection_policy": "frozen fielded-BM25 top-1 from router_transfer_v1",
        "selection_is_model_independent": True,
        "cases": cases,
        "repositories": frozen_repositories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("experiments/paper9_subagents/benchmarks/router_transfer_v1.json"),
    )
    parser.add_argument(
        "--routing-rows",
        type=Path,
        default=Path(
            "docs/papers/shared/results/paper9_subagents/router_transfer_v1/routing_rows.csv"
        ),
    )
    parser.add_argument(
        "--transfer-summary",
        type=Path,
        default=Path(
            "docs/papers/shared/results/paper9_subagents/router_transfer_v1/summary.json"
        ),
    )
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        help="Repository mapping as repository_id=/absolute/path.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repositories = {
        value.split("=", 1)[0]: Path(value.split("=", 1)[1]).resolve()
        for value in args.repo
    }
    manifest = build_manifest(
        args.benchmark,
        args.routing_rows,
        args.transfer_summary,
        repositories,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(manifest["cases"]),
                "repositories": sorted(manifest["repositories"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
