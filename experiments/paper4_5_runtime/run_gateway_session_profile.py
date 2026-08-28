"""Measure logical prefix preservation and session-delta accounting.

The experiment is deterministic and adapter-local. It measures serialized
message/resource bytes and logical prefix stability; physical cache hits remain
unknown because no remote engine telemetry is available.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt

from pra_hf.deployment import (
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAWireRequest,
    PRAWireResource,
)
from pra_hf.engine_profiles import EngineProfileRegistry, EngineType
from pra_hf.gateway import PRAGateway


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper4_5_runtime"
PACKAGED_REGISTRY = ROOT / "src/pra_hf/model_profiles/engine_profile_registry.json"


@dataclass
class FixtureAdapter:
    name: str
    pra: bool
    session_state: bool
    incremental: bool

    def __post_init__(self) -> None:
        self.requests: list[PRAWireRequest] = []
        self.prepare_calls = 0
        self.close_calls = 0

    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(
            adapter=self.name,
            engine_type="custom",
            integration_level="E1" if self.pra else "E0",
            prefix_cache_mode=("session_state" if self.session_state else "automatic_prefix_cache"),
            automatic_prefix_cache=not self.session_state,
            session_state=self.session_state,
            incremental_messages=self.incremental,
            resource_delta=self.pra,
            cache_affinity=self.session_state,
            logical_refs=self.pra,
            native_kv=self.pra,
            text_fallback=True,
        )

    def prepare_session(self, request: PRAWireRequest) -> str | None:
        self.prepare_calls += 1
        return f"fixture:{request.session_id}" if self.session_state else None

    def generate(self, request: PRAWireRequest) -> PRAEngineResult:
        self.requests.append(request)
        return PRAEngineResult(f"answer-{len(self.requests)}", {"prefix_cache_hit": None})

    def stream(self, request):
        raise NotImplementedError

    def close_session(self, session_id: str) -> None:
        del session_id
        self.close_calls += 1


def _bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _resource(turn: int) -> PRAWireResource:
    return PRAWireResource(
        "selected-fact",
        "pra://profile/selected-fact",
        text=f"turn {turn} selected evidence value {turn * 11}",
        metadata={"tenant_id": "tenant-a", "version": f"v{turn}"},
        authorization_scope="tenant-a",
    )


def _session_trace(result: PRAEngineResult) -> Mapping[str, Any]:
    return next(row for row in result.trace if row.get("stage") == "gateway_session")


def _run_gateway_condition(name: str, *, pra: bool, session_state: bool, incremental: bool) -> list[dict[str, Any]]:
    adapter = FixtureAdapter(name, pra, session_state, incremental)
    gateway = PRAGateway(adapter, mode="G11" if pra else "G10")
    history: list[Mapping[str, Any]] = [{"role": "system", "content": "Answer using selected evidence."}]
    rows = []
    for turn in range(1, 6):
        history.append({"role": "user", "content": f"question {turn}"})
        result = gateway.generate(PRAWireRequest(
            model="offline/profile-model",
            messages=tuple(history),
            tenant_id="tenant-a",
            session_id="session-a",
            resources=(_resource(turn),),
            history_mode="AUTO",
            allow_text_fallback=True,
        ))
        trace = _session_trace(result)
        rows.append({
            "condition": name,
            "turn": turn,
            "history_mode": trace["history_mode"],
            "gateway_prefix_stable": int(trace["gateway_prefix_stable"]),
            "stable_prefix_messages": trace["prefix_tokens_reusable"],
            "prefix_reuse_fraction": trace["prefix_reuse_fraction"],
            "prefix_invalidations": trace["prefix_invalidations"],
            "message_bytes_sent": trace["message_bytes_sent"],
            "resource_bytes_sent": trace["resource_bytes_sent"],
            "session_delta_bytes": trace["session_delta_bytes"],
            "engine_session_reuse": int(trace["engine_session_reuse"]),
            "engine_prefix_cache_hit": "UNKNOWN",
            "resource_ops": ";".join(trace["resource_ops"]),
            "engine_prepare_calls": adapter.prepare_calls,
        })
        history.append({"role": "assistant", "content": result.text})
    return rows


def _legacy_prepend() -> list[dict[str, Any]]:
    """Reproduce the removed message-zero G10 geometry as a control."""

    history: list[Mapping[str, Any]] = [{"role": "system", "content": "Answer using selected evidence."}]
    previous: tuple[Mapping[str, Any], ...] = ()
    rows = []
    for turn in range(1, 6):
        history.append({"role": "user", "content": f"question {turn}"})
        context = {
            "role": "system",
            "content": f"PRA text fallback context (not native K/V): turn {turn} selected evidence value {turn * 11}",
        }
        outbound = (context, *history)
        stable = 0
        for left, right in zip(previous, outbound):
            if left != right:
                break
            stable += 1
        rows.append({
            "condition": "CURRENT_G10_PREPEND",
            "turn": turn,
            "history_mode": "FULL",
            "gateway_prefix_stable": int(bool(previous) and stable == len(previous)),
            "stable_prefix_messages": stable,
            "prefix_reuse_fraction": stable / max(len(outbound) + 1, 1),
            "prefix_invalidations": int(bool(previous) and stable < len(previous)),
            "message_bytes_sent": _bytes(outbound),
            "resource_bytes_sent": 0,
            "session_delta_bytes": 0,
            "engine_session_reuse": 0,
            "engine_prefix_cache_hit": "UNKNOWN",
            "resource_ops": "",
            "engine_prepare_calls": 0,
        })
        answer = {"role": "assistant", "content": f"answer-{turn}"}
        history.append(answer)
        previous = (*outbound, answer)
    return rows


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _capability_rows() -> list[dict[str, Any]]:
    registry = EngineProfileRegistry.default()
    rows = []
    for engine_type in EngineType:
        profile = registry.resolve(engine_type)
        rows.append({
            "engine_type": engine_type.value,
            "default_pra_level": profile.default_pra_level,
            "prefix_cache_mode": profile.default_prefix_cache_mode.value,
            "streaming": int(profile.streaming),
            "session_state": int(profile.explicit_session),
            "incremental_messages": int(profile.incremental_messages),
            "resource_delta": int(profile.resource_delta),
            "cache_affinity": int(profile.cache_affinity),
            "claim_scope": "packaged_default_not_runtime_probe",
        })
    return rows


def _policy_rows() -> list[dict[str, Any]]:
    return [
        {"condition":"CURRENT_G10_PREPEND","pra_level":"E0","history":"FULL","fallback":"message_zero_control","resource_transport":"text","physical_hit":"UNKNOWN"},
        {"condition":"PREFIX_PRESERVING_G10","pra_level":"E0","history":"FULL","fallback":"before_current_user","resource_transport":"text","physical_hit":"UNKNOWN"},
        {"condition":"SESSION_DELTA_E0","pra_level":"E0","history":"DELTA after turn 1","fallback":"before_current_user","resource_transport":"text","physical_hit":"UNKNOWN"},
        {"condition":"PRA_ENABLED_SESSION","pra_level":"E1 contract","history":"DELTA after turn 1","fallback":"none","resource_transport":"ADD/UPDATE/REMOVE","physical_hit":"UNKNOWN"},
    ]


def _summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = []
    for condition in dict.fromkeys(row["condition"] for row in rows):
        group = [row for row in rows if row["condition"] == condition]
        values.append({
            "condition": condition,
            "turns": len(group),
            "stable_turns_after_first": sum(int(row["gateway_prefix_stable"]) for row in group[1:]),
            "mean_prefix_reuse_fraction": sum(float(row["prefix_reuse_fraction"]) for row in group[1:]) / 4,
            "prefix_invalidations": sum(int(row["prefix_invalidations"]) for row in group),
            "message_bytes_sent": sum(int(row["message_bytes_sent"]) for row in group),
            "resource_bytes_sent": sum(int(row["resource_bytes_sent"]) for row in group),
            "session_delta_bytes": sum(int(row["session_delta_bytes"]) for row in group),
            "engine_session_reuse_turns": sum(int(row["engine_session_reuse"]) for row in group),
            "engine_prefix_cache_hits": "NOT_MEASURED",
        })
    return values


def _plots(summary: list[Mapping[str, Any]]) -> None:
    labels = [row["condition"].replace("_", "\n") for row in summary]
    reuse = [100 * float(row["mean_prefix_reuse_fraction"]) for row in summary]
    messages = [int(row["message_bytes_sent"]) for row in summary]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(labels, reuse, color=["#b44b4b", "#307b72", "#3d65a5", "#6f559f"])
    axes[0].set_ylabel("Mean logical prefix reuse (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(labels, messages, color=["#b44b4b", "#307b72", "#3d65a5", "#6f559f"])
    axes[1].set_ylabel("Messages sent (bytes, five turns)")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS / "gateway_prefix_reuse.pdf", bbox_inches="tight")
    figure.savefig(RESULTS / "gateway_prefix_reuse.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7)
    axis.axis("off")
    boxes = {
        "Agent / logical session": (0.5, 5.3, 2.2, 0.9, "#e9ecef"),
        "PRA gateway\nsession policy": (3.7, 5.1, 2.5, 1.2, "#d7e9e4"),
        "Stable sequential prefix": (1.0, 2.9, 2.8, 1.0, "#dce6f4"),
        "Detached PRA resources": (6.2, 2.9, 2.8, 1.0, "#eadff0"),
        "Prefix K/V cache": (1.0, 0.8, 2.8, 1.0, "#c9d9ef"),
        "PRA address / native cache": (6.2, 0.8, 2.8, 1.0, "#dfcde8"),
        "Remote engine": (4.0, 0.0, 2.0, 0.65, "#f1e2c9"),
    }
    for label, (x, y, w, h, color) in boxes.items():
        axis.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", linewidth=1.2))
        axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    arrows = [((2.7, 5.75), (3.7, 5.75)), ((4.3, 5.1), (2.4, 3.9)), ((5.6, 5.1), (7.6, 3.9)), ((2.4, 2.9), (2.4, 1.8)), ((7.6, 2.9), (7.6, 1.8)), ((2.4, 0.8), (4.5, 0.55)), ((7.6, 0.8), (5.5, 0.55))]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle":"->", "color":"#333333", "lw":1.3})
    axis.text(5, 4.25, "independent state axes", ha="center", fontsize=9, color="#444444")
    figure.tight_layout()
    figure.savefig(RESULTS / "gateway_two_cache_architecture.pdf", bbox_inches="tight")
    figure.savefig(RESULTS / "gateway_two_cache_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _tex(summary: list[Mapping[str, Any]]) -> str:
    lines = [
        (
            r"\begin{tabular}{@{}l@{\hspace{1em}}r@{\hspace{1em}}r"
            r"@{\hspace{1em}}r@{\hspace{1em}}r@{\hspace{1em}}r@{}}"
        ),
        r"\toprule",
        r"Condition & Stable turns & Reuse & Invalidations & Msg bytes & Resource bytes\\",
        r"\midrule",
    ]
    for row in summary:
        name = str(row["condition"]).replace("_", r"\_")
        lines.append(
            f"{name} & {row['stable_turns_after_first']}/4 & "
            f"{100*float(row['mean_prefix_reuse_fraction']):.1f}\\% & "
            f"{row['prefix_invalidations']} & {row['message_bytes_sent']} & {row['resource_bytes_sent']}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = _legacy_prepend()
    rows += _run_gateway_condition("PREFIX_PRESERVING_G10", pra=False, session_state=False, incremental=False)
    rows += _run_gateway_condition("SESSION_DELTA_E0", pra=False, session_state=True, incremental=True)
    rows += _run_gateway_condition("PRA_ENABLED_SESSION", pra=True, session_state=True, incremental=True)
    summary = _summary(rows)
    _write_csv(RESULTS / "prefix_preservation_results.csv", rows)
    _write_csv(RESULTS / "session_delta_results.csv", summary)
    _write_csv(RESULTS / "engine_cache_capability_matrix.csv", _capability_rows())
    _write_csv(RESULTS / "gateway_session_policy_matrix.csv", _policy_rows())
    traces = {
        "schema_version": "1.0",
        "physical_prefix_cache_hits_measured": False,
        "examples": [dict(row) for row in rows if int(row["turn"]) in {1, 5}],
    }
    (RESULTS / "gateway_cache_trace_examples.json").write_text(
        json.dumps(traces, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULTS / "engine_profile_registry.json").write_text(
        PACKAGED_REGISTRY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (RESULTS / "generated_gateway_session_table.tex").write_text(
        _tex(summary), encoding="utf-8"
    )
    _plots(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
