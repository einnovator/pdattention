"""Measure cumulative-context duplication removed by the shared visibility ledger."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pra_hf.deployment import (
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAWireRequest,
    PRAWireResource,
)
from pra_hf.gateway import PRAGateway


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/papers/shared/results/paper4_5_runtime/session_realization"


@dataclass
class _Adapter:
    requests: list[PRAWireRequest]

    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(
            adapter="session-realization-accounting",
            engine_type="custom",
            prefix_cache_mode="unknown",
            session_state=True,
            incremental_messages=True,
            text_fallback=True,
        )

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        return request.session_id

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        self.requests.append(request)
        return PRAEngineResult("correct", {})

    def stream(self, request: PRAWireRequest):
        raise NotImplementedError

    def close_session(self, session_id: str) -> None:
        return None


def _resource(seed: int, name: str, tokens: int) -> PRAWireResource:
    text = " ".join(f"{name.lower()}-{seed}-{index}" for index in range(tokens))
    return PRAWireResource(
        resource_id=name,
        uri=f"pra://tenant-a/{seed}/{name}",
        text=text,
        version="v1",
        authorization_scope="tenant-a",
        metadata={"tenant_id": "tenant-a", "version": "v1"},
    )


def _request(seed: int, turn: int, resources: tuple[PRAWireResource, ...]) -> PRAWireRequest:
    return PRAWireRequest(
        model="controlled/session-model",
        messages=({"role": "user", "content": f"question {seed}-{turn}"},),
        tenant_id="tenant-a",
        session_id=f"session-{seed}",
        resources=resources,
        history_mode="DELTA",
        allow_text_fallback=True,
        pra_policy={"profile": "BALANCED"},
    )


def run(*, seeds: tuple[int, ...] = (11, 23, 37, 53, 71)) -> Mapping[str, Any]:
    rows = []
    for seed in seeds:
        a, b, c, d = (
            _resource(seed, "A", 24),
            _resource(seed, "B", 16),
            _resource(seed, "C", 20),
            _resource(seed, "D", 12),
        )
        first_tokens = sum(len(resource.text.split()) for resource in (a, b, c))
        second_selected = sum(len(resource.text.split()) for resource in (b, d))
        all_unique = sum(len(resource.text.split()) for resource in (a, b, c, d))

        adapter = _Adapter([])
        gateway = PRAGateway(adapter, mode="G10")
        gateway.generate(_request(seed, 1, (a, b, c)))
        result = gateway.generate(_request(seed, 2, (b, d)))
        trace = next(row for row in result.trace if row.get("stage") == "gateway_session")

        conditions = (
            {
                "condition": "full_context",
                "visible_tokens": all_unique,
                "new_materialized_tokens": all_unique,
                "logical_reuse_tokens": 0,
            },
            {
                "condition": "selected_context_without_logical_reuse",
                "visible_tokens": first_tokens + second_selected,
                "new_materialized_tokens": second_selected,
                "logical_reuse_tokens": 0,
            },
            {
                "condition": "session_aware_selected_context",
                "visible_tokens": first_tokens + len(d.text.split()),
                "new_materialized_tokens": trace["new_materialized_tokens"],
                "logical_reuse_tokens": trace["visible_reuse_tokens"],
            },
            {
                "condition": "native_memory",
                "visible_tokens": None,
                "new_materialized_tokens": None,
                "logical_reuse_tokens": None,
            },
        )
        for condition in conditions:
            rows.append(
                {
                    "seed": seed,
                    **condition,
                    "prefix_cached_tokens": None,
                    "native_reuse_tokens": None,
                    "quality": 1.0 if condition["condition"] != "native_memory" else None,
                    "ttft_ms": None,
                    "measurement_status": (
                        "CONTROLLED_ACCOUNTING"
                        if condition["condition"] != "native_memory"
                        else "NOT_MEASURED"
                    ),
                }
            )
    aggregates = []
    for condition in dict.fromkeys(row["condition"] for row in rows):
        group = [row for row in rows if row["condition"] == condition]

        def mean(name: str):
            values = [float(row[name]) for row in group if row[name] is not None]
            return statistics.fmean(values) if values else None

        aggregates.append(
            {
                "condition": condition,
                "seeds": len(group),
                "visible_tokens": mean("visible_tokens"),
                "new_materialized_tokens": mean("new_materialized_tokens"),
                "logical_reuse_tokens": mean("logical_reuse_tokens"),
                "prefix_cached_tokens": mean("prefix_cached_tokens"),
                "native_reuse_tokens": mean("native_reuse_tokens"),
                "quality": mean("quality"),
                "ttft_ms": mean("ttft_ms"),
                "measurement_status": group[0]["measurement_status"],
            }
        )
    return {
        "schema_version": "1.0",
        "experiment": "paper4_5_session_aware_selected_context_v1",
        "evidence_tier": "CONTROLLED_ACCOUNTING",
        "claim_boundary": (
            "Logical visible-token accounting only; prefix-cache compute reuse, "
            "native-state reuse, and TTFT remain separate and unmeasured."
        ),
        "turns": ["turn 1 selects A/B/C", "turn 2 selects B/D"],
        "rows": rows,
        "aggregates": aggregates,
    }


def write(payload: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "session_realization_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    columns = tuple(payload["rows"][0])
    with (output / "session_realization_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(payload["rows"])
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & Visible & New materialization & Logical reuse & Prefix cached & Native reused & Quality \\",
        r"\midrule",
    ]
    for row in payload["aggregates"]:
        def value(name: str) -> str:
            item = row[name]
            return r"\textsc{Not measured}" if item is None else f"{item:.1f}"

        name = str(row["condition"]).replace("_", r"\_")
        lines.append(
            f"{name} & {value('visible_tokens')} & {value('new_materialized_tokens')} & "
            f"{value('logical_reuse_tokens')} & {value('prefix_cached_tokens')} & "
            f"{value('native_reuse_tokens')} & {value('quality')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "generated_session_realization_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run()
    write(payload, args.output)
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
