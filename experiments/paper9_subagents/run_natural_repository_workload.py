"""Run sequential/parallel subagents over real PRA repository source files.

This is a natural-source harness benchmark, not a SWE-bench success claim. It
measures actual file callbacks, cross-agent orientation reuse, completed-child
visibility, and fixed/learned/oracle descendant selection on tracked code.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.subagent_context import (
    ContextVisibilityPolicy,
    EffectType,
    ResourceIdentity,
    ToolEffectDescriptor,
)
from pra_hf.subagent_harness import DeclarativeTool, SubagentHarness, SubagentSpec
from pra_hf.subagent_routing import (
    DescendantRecordRouter,
    DescendantRoutingExample,
    DescendantRoutingMode,
    visible_routing_candidates,
)


SEEDS = (11, 23, 37, 71, 101)


@dataclass(frozen=True)
class RepositoryQuestion:
    path: str
    query: str


QUESTIONS = (
    RepositoryQuestion("src/pra_hf/subagent_context.py", "Where are unknown side effects rejected from cross-agent reuse?"),
    RepositoryQuestion("src/pra_hf/subagent_harness.py", "Which module owns bounded parallel child callback scheduling?"),
    RepositoryQuestion("src/pra_hf/subagent_routing.py", "Where are visible completed-child records ranked with pairwise logistic preferences?"),
    RepositoryQuestion("src/pra_hf/progressive_context.py", "Which module defines lazy native regions and context transport decisions?"),
    RepositoryQuestion("src/pra_hf/session_service.py", "Where are agent sessions persisted and recovered from local storage?"),
    RepositoryQuestion("src/pra_hf/task_context.py", "Which module models task relations, events, and task state?"),
    RepositoryQuestion("src/pra_hf/tool_records.py", "Where are Python callable parameters and return schemas represented?"),
    RepositoryQuestion("src/pra_hf/hybrid_discovery.py", "Which module combines token-native exact and semantic discovery candidates?"),
    RepositoryQuestion("src/pra_hf/context_records.py", "Where are typed record views, atomicity, and materialization policies defined?"),
    RepositoryQuestion("src/pra_hf/capability_runtime.py", "Which module lazily activates complete tool and skill definitions?"),
    RepositoryQuestion("src/pra_hf/context_store.py", "Where are scoped backing records authorized and loaded?"),
)


def _effect(path: str) -> ToolEffectDescriptor:
    return ToolEffectDescriptor(
        EffectType.READ,
        (ResourceIdentity("os.FILE", path),),
        reuse_enabled=True,
    )


def _read_tool(repo: Path, calls: list[str]) -> DeclarativeTool:
    def execute(arguments):
        relative = str(arguments["path"])
        calls.append(relative)
        return (repo / relative).read_text(encoding="utf-8")

    return DeclarativeTool(
        "tool://repository/read_text",
        execute,
        lambda arguments: _effect(str(arguments["path"])),
    )


def run_cohort(repo: Path, *, seed: int, parallel: bool) -> tuple[list[dict], dict]:
    randomizer = random.Random(seed)
    questions = list(QUESTIONS)
    randomizer.shuffle(questions)
    calls: list[str] = []
    harness = SubagentHarness(
        f"natural-{seed}-{parallel}",
        root_agent_uuid="root",
        root_context_policy=ContextVisibilityPolicy(descendant="completed_only"),
    )
    tool = _read_tool(repo, calls)

    # Common orientation is acquired once. Every child asks for it through the
    # normal tool API, exercising lineage-aware exact result reuse.
    orientation_paths = ("README.md", "pyproject.toml")
    for path in orientation_paths:
        harness.execute_tool("root", tool, {"path": path})

    path_by_task = {f"task-{index}": question.path for index, question in enumerate(questions)}
    specs = tuple(
        SubagentSpec(
            parent_task_uuid=task,
            context_policy=ContextVisibilityPolicy(ancestor="routable"),
        )
        for task in path_by_task
    )

    def child_runner(child, active_harness):
        for path in orientation_paths:
            active_harness.execute_tool(child.agent_uuid, tool, {"path": path})
        target = path_by_task[child.parent_task_uuid or ""]
        return active_harness.execute_tool(child.agent_uuid, tool, {"path": target})

    started = time.perf_counter()
    child_results = harness.run_subagents(
        "root", specs, child_runner, parallel=parallel, max_workers=min(8, len(specs))
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    if any(row.error for row in child_results):
        raise RuntimeError(f"Subagent callback failed: {[row.error for row in child_results]}")

    candidates = visible_routing_candidates(harness.graph, "root")
    evidence = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate.record.payload, dict)
        and str(candidate.record.payload.get("arguments", {}).get("path", ""))
        in {question.path for question in questions}
    )
    record_by_path = {
        str(candidate.record.payload["arguments"]["path"]): candidate.record_uuid
        for candidate in evidence
    }
    examples = tuple(
        DescendantRoutingExample(
            question.query,
            evidence,
            frozenset({record_by_path[question.path]}),
        )
        for question in questions
    )
    split = max(2, round(len(examples) * 0.65))
    router = DescendantRecordRouter().fit(examples[:split])
    rows: list[dict] = []
    for example in examples[split:]:
        for mode in DescendantRoutingMode:
            route = router.route(
                example.query,
                evidence,
                mode=mode,
                top_k=1,
                oracle_record_ids=example.relevant_record_ids,
            )
            rows.append(
                {
                    "seed": seed,
                    "scheduler": "parallel" if parallel else "sequential",
                    "query": example.query,
                    "mode": mode.value,
                    "recall": route.recall(example.relevant_record_ids),
                    "precision": route.precision(example.relevant_record_ids),
                    "selected_tokens": route.selected_tokens,
                    "candidate_count": route.candidate_count,
                    "all_valid_tokens": sum(row.token_count for row in evidence),
                }
            )
    stats = {
        "seed": seed,
        "scheduler": "parallel" if parallel else "sequential",
        "children": len(child_results),
        "wall_ms": wall_ms,
        "physical_file_reads": len(calls),
        "logical_file_reads": len(orientation_paths) + len(child_results) * (len(orientation_paths) + 1),
        "orientation_reuses": len(child_results) * len(orientation_paths),
        "candidate_records": len(evidence),
        "source_bytes": sum((repo / question.path).stat().st_size for question in questions),
    }
    return rows, stats


def summarize(rows: list[dict], scheduler_rows: list[dict]) -> dict:
    by_mode = {}
    for mode in DescendantRoutingMode:
        selected = [row for row in rows if row["mode"] == mode.value]
        by_mode[mode.value] = {
            "queries": len(selected),
            "evidence_recall": statistics.mean(row["recall"] for row in selected),
            "useful_precision": statistics.mean(row["precision"] for row in selected),
            "mean_selected_tokens": statistics.mean(row["selected_tokens"] for row in selected),
            "mean_all_valid_tokens": statistics.mean(row["all_valid_tokens"] for row in selected),
        }
    by_scheduler = {}
    for scheduler in ("sequential", "parallel"):
        selected = [row for row in scheduler_rows if row["scheduler"] == scheduler]
        by_scheduler[scheduler] = {
            "runs": len(selected),
            "mean_wall_ms": statistics.mean(row["wall_ms"] for row in selected),
            "mean_physical_file_reads": statistics.mean(row["physical_file_reads"] for row in selected),
            "mean_logical_file_reads": statistics.mean(row["logical_file_reads"] for row in selected),
            "orientation_reuses": sum(row["orientation_reuses"] for row in selected),
        }
    return {
        "protocol": "paper9-natural-repository-v1",
        "interpretation": "Natural tracked source files with deterministic harness callbacks; not an autonomous coding-agent success benchmark.",
        "seeds": list(SEEDS),
        "routing": by_mode,
        "scheduling": by_scheduler,
    }


def plot(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    modes = [mode.value for mode in DescendantRoutingMode]
    recall = [summary["routing"][mode]["evidence_recall"] for mode in modes]
    tokens = [summary["routing"][mode]["mean_selected_tokens"] for mode in modes]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].bar(modes, recall, color=("#68747d", "#24796b", "#b8872d", "#255b96"))
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Evidence recall")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(modes, tokens, color=("#68747d", "#24796b", "#b8872d", "#255b96"))
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Selected source tokens (log)")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output / "natural_repository_routing.pdf", bbox_inches="tight")
    figure.savefig(output / "natural_repository_routing.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/shared/results/paper9_subagents/natural_repository_v1"),
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    routing_rows: list[dict] = []
    scheduler_rows: list[dict] = []
    for seed in SEEDS:
        for parallel in (False, True):
            rows, stats = run_cohort(arguments.repo.resolve(), seed=seed, parallel=parallel)
            routing_rows.extend(rows)
            scheduler_rows.append(stats)
    summary = summarize(routing_rows, scheduler_rows)
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, rows in (("routing_rows.csv", routing_rows), ("scheduler_rows.csv", scheduler_rows)):
        with (arguments.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot(summary, arguments.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
