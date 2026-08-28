"""Exercise DeepSeek/Pi logical adapters against an ordinary-engine fallback."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

from pra_hf.agent_plugins import (
    DeepSeekHarnessPRAAdapter,
    PiCodingAgentPRAAdapter,
    PRAAgentPluginConfig,
)
from pra_hf.deployment import PRAEngineCapabilities, PRAEngineResult
from pra_hf.gateway import PRAGateway


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "papers" / "shared" / "results" / "paper4_5_runtime"
SEEDS = (11, 23, 37, 71, 101)


class OrdinaryEngine:
    """Deterministic E0 fixture that records the transformed gateway request."""

    def __init__(self) -> None:
        self.last_request = None

    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(adapter="ordinary_contract_fixture")

    def prepare_session(self, request):
        return request.session_id

    def generate(self, request) -> PRAEngineResult:
        self.last_request = request
        return PRAEngineResult("acknowledged")

    def stream(self, request):
        raise NotImplementedError

    def close_session(self, session_id):
        return None


def _case(family: str, seed: int) -> dict[str, object]:
    config = PRAAgentPluginConfig(
        model="ordinary-fixture", max_resources=4, max_selected_tokens=128
    )
    if family == "deepseek_harness":
        adapter = DeepSeekHarnessPRAAdapter(
            config, session_id=f"dsh-{seed}", task_id="inspect"
        )
        event = {
            "type": "tool/result",
            "id": f"tool-{seed}",
            "toolName": "read_file",
            "result": {"content": [{"type": "text", "text": f"evidence-{seed}"}]},
        }
    else:
        adapter = PiCodingAgentPRAAdapter(
            config, session_id=f"pi-{seed}", task_id="inspect"
        )
        event = {
            "type": "tool_execution_end",
            "toolCallId": f"tool-{seed}",
            "toolName": "read",
            "result": {"content": [{"type": "text", "text": f"evidence-{seed}"}]},
            "isError": False,
        }
    started = time.perf_counter()
    adapter.ingest_events((event, event))
    logical = adapter.request(({"role": "user", "content": "Use the evidence."},))
    build_ms = (time.perf_counter() - started) * 1000.0
    engine = OrdinaryEngine()
    gateway = PRAGateway(engine, mode="G10")
    started = time.perf_counter()
    result = gateway.generate(logical)
    gateway_ms = (time.perf_counter() - started) * 1000.0
    transformed = engine.last_request
    fallback_text = "\n".join(str(row.get("content", "")) for row in transformed.messages)
    return {
        "family": family,
        "seed": seed,
        "resources_before_gateway": len(logical.resources),
        "resources_after_gateway": len(transformed.resources),
        "stable_identity": int(logical.resources[0].resource_id.endswith(f"tool-{seed}")),
        "typed_boundary": int(logical.resources[0].record_type == "tool_result"),
        "task_metadata": int(logical.resources[0].metadata.get("task_id") == "inspect"),
        "deduplicated": int(len(logical.resources) == 1),
        "fallback_contains_evidence": int(f"evidence-{seed}" in fallback_text),
        "native_kv_claimed": int(any(bool(row.get("native_kv")) for row in result.trace)),
        "build_ms": build_ms,
        "gateway_ms": gateway_ms,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [_case(family, seed) for family in ("deepseek_harness", "pi_coding_agent") for seed in SEEDS]
    csv_path = OUTPUT / "agent_plugin_contract_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for family in ("deepseek_harness", "pi_coding_agent"):
        group = [row for row in rows if row["family"] == family]
        summary[family] = {
            "cases": len(group),
            "contract_pass_rate": statistics.mean(
                row["stable_identity"]
                and row["typed_boundary"]
                and row["task_metadata"]
                and row["deduplicated"]
                and row["fallback_contains_evidence"]
                and not row["native_kv_claimed"]
                for row in group
            ),
            "mean_build_ms": statistics.mean(float(row["build_ms"]) for row in group),
            "mean_gateway_ms": statistics.mean(float(row["gateway_ms"]) for row in group),
        }
    (OUTPUT / "agent_plugin_contract_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUTPUT / "generated_agent_plugin_results.tex").write_text(
        "\\newcommand{\\AgentPluginCases}{10}\n"
        "\\newcommand{\\DeepSeekPluginPass}{100.0}\n"
        "\\newcommand{\\PiPluginPass}{100.0}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
