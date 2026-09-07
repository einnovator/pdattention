"""Measure live MLX prefill with frozen shared evidence across child streams.

The selector is intentionally absent: every condition receives identical token
IDs. The benchmark separates conventional prefill, ordinary payload
memoization, typed-record reuse with re-prefill, and native K/V reuse.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pra_hf.subagent_context import ContextVisibilityPolicy, EffectType, ResourceIdentity, ToolEffectDescriptor
from pra_hf.subagent_harness import DeclarativeTool, SubagentHarness, SubagentSpec
from pra_hf.subagent_mlx_native import MLXSubagentNativePort


SEEDS = (11, 23, 37, 71, 101)
INTEGER_FIELDS = {
    "seed", "target_shared_tokens", "actual_shared_tokens", "query_tokens", "fanout",
    "child_index", "native_kv_bytes", "physical_input_tokens", "native_tokens_reused",
    "argmax_matches_host_split", "payload_hit", "native_hit", "tool_calls_so_far",
}
FLOAT_FIELDS = {
    "one_time_native_encode_ms", "model_ms", "route_attach_ms", "request_ms",
    "max_abs_logit_delta_vs_host_split",
}


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def load_rows(path: Path) -> list[dict]:
    """Restore numeric CSV fields for analysis without loading a model."""

    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        for name in INTEGER_FIELDS:
            row[name] = int(row[name])
        for name in FLOAT_FIELDS:
            row[name] = float(row[name])
        rows.append(row)
    return rows


def timed_forward(model, token_ids: list[int], *, cache=None) -> tuple[float, np.ndarray]:
    import mlx.core as mx

    started = time.perf_counter()
    logits = model(mx.array(token_ids, dtype=mx.int32)[None], cache=cache)
    final = logits[0, -1]
    mx.eval(final)
    return (time.perf_counter() - started) * 1000.0, np.asarray(final.astype(mx.float32))


def timed_split_forward(model, source_ids: list[int], query_ids: list[int]) -> tuple[float, np.ndarray]:
    """Prefill source then query through an ordinary host prompt cache."""

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    started = time.perf_counter()
    model(mx.array(source_ids, dtype=mx.int32)[None], cache=cache)
    logits = model(mx.array(query_ids, dtype=mx.int32)[None], cache=cache)
    final = logits[0, -1]
    mx.eval(final)
    return (time.perf_counter() - started) * 1000.0, np.asarray(final.astype(mx.float32))


def shared_text(tokenizer, target_tokens: int, seed: int) -> str:
    line = (
        f"Repository orientation seed {seed}: ContextRecord preserves resource identity, "
        "causal position, tool validity, and immutable native attention state. "
    )
    text = line * max(2, target_tokens // 12)
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text += line
        ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)


def run_point(model, tokenizer, model_id: str, seed: int, target_tokens: int, fanout: int) -> list[dict]:
    import mlx.core as mx

    content = shared_text(tokenizer, target_tokens, seed)
    port = MLXSubagentNativePort(model, tokenizer, model_id=model_id, revision="frozen")
    harness = SubagentHarness(
        f"mlx-{seed}-{target_tokens}-{fanout}",
        root_agent_uuid="root",
        native_port=port,
    )
    calls = 0

    def execute(_):
        nonlocal calls
        calls += 1
        return content

    effect = ToolEffectDescriptor(
        EffectType.READ,
        (ResourceIdentity("paper9.REPOSITORY", f"orientation-{seed}-{target_tokens}"),),
        reuse_enabled=True,
    )
    tool = DeclarativeTool("tool://paper9/orientation", execute, lambda _: effect)
    encode_started = time.perf_counter()
    parent = harness.execute_tool("root", tool, {})
    encode_ms = (time.perf_counter() - encode_started) * 1000.0
    native = port.memories[parent.record.record_id]
    source_ids = tokenizer.encode(content, add_special_tokens=False)
    rows: list[dict] = []
    for child_index in range(fanout):
        child = harness.spawn_subagent(
            "root",
            SubagentSpec(context_policy=ContextVisibilityPolicy(ancestor="routable")),
            agent_uuid=f"child-{child_index}",
        )
        query = f"\nChild {child_index}: name the record field that preserves causal order. Answer:"
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        full_ids = [*source_ids, *query_ids]

        # A: no reuse. This repeats the exact selected source in each child.
        isolated_ms, isolated_logits = timed_forward(model, full_ids)
        split_ms, split_logits = timed_split_forward(model, source_ids, query_ids)

        # B: ordinary result memoization still serializes and prefills the text.
        lookup_started = time.perf_counter()
        payload = harness.execute_tool(
            child.agent_uuid,
            tool,
            {},
            allow_native_kv_reuse=False,
        )
        memo_lookup_ms = (time.perf_counter() - lookup_started) * 1000.0
        memo_ms, memo_logits = timed_forward(model, full_ids)

        # C: a visible typed record is reused, but execution remains E0 text.
        record_lookup_started = time.perf_counter()
        record_reuse = harness.execute_tool(
            child.agent_uuid,
            tool,
            {},
            allow_native_kv_reuse=False,
        )
        record_lookup_ms = (time.perf_counter() - record_lookup_started) * 1000.0
        record_ms, record_logits = timed_forward(model, full_ids)

        # D: the same typed record attaches immutable host-native K/V by ref.
        native_lookup_started = time.perf_counter()
        native_reuse = harness.execute_tool(
            child.agent_uuid,
            tool,
            {},
            requested_native=port.requested_handle(),
        )
        native_lookup_ms = (time.perf_counter() - native_lookup_started) * 1000.0
        native_ms, native_logits = timed_forward(
            model, query_ids, cache=port.prompt_cache(child.agent_uuid)
        )
        mx.clear_cache()

        common = {
            "seed": seed,
            "target_shared_tokens": target_tokens,
            "actual_shared_tokens": len(source_ids),
            "query_tokens": len(query_ids),
            "fanout": fanout,
            "child_index": child_index,
            "one_time_native_encode_ms": encode_ms,
            "native_kv_bytes": native.nbytes,
        }
        values = (
            ("isolated_text", isolated_ms, 0.0, len(full_ids), isolated_logits),
            ("host_split_text", split_ms, 0.0, len(full_ids), split_logits),
            ("harness_memo_text", memo_ms, memo_lookup_ms, len(full_ids), memo_logits),
            ("pra_record_reprefill", record_ms, record_lookup_ms, len(full_ids), record_logits),
            ("pra_native_kv", native_ms, native_lookup_ms, len(query_ids), native_logits),
        )
        for condition, model_ms, route_ms, physical_tokens, logits in values:
            rows.append(
                {
                    **common,
                    "condition": condition,
                    "model_ms": model_ms,
                    "route_attach_ms": route_ms,
                    "request_ms": model_ms + route_ms,
                    "physical_input_tokens": physical_tokens,
                    "native_tokens_reused": len(source_ids) if condition == "pra_native_kv" else 0,
                    "argmax_matches_host_split": int(np.argmax(logits) == np.argmax(split_logits)),
                    "max_abs_logit_delta_vs_host_split": float(np.max(np.abs(logits - split_logits))),
                    "payload_hit": int(condition != "isolated_text" and not payload.executed),
                    "native_hit": int(condition == "pra_native_kv" and native_reuse.reuse.kv_reused),
                    "tool_calls_so_far": calls,
                }
            )
        assert not payload.executed and not record_reuse.executed
        harness.stop_subagent(child.agent_uuid)
    return rows


def summarize(rows: list[dict], model_id: str, seeds: tuple[int, ...]) -> dict:
    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        latencies = [row["request_ms"] for row in selected]
        by_condition[condition] = {
            "requests": len(selected),
            "mean_request_ms": statistics.mean(latencies),
            "p50_request_ms": percentile(latencies, 50),
            "p95_request_ms": percentile(latencies, 95),
            "p99_request_ms": percentile(latencies, 99),
            "physical_input_tokens": sum(row["physical_input_tokens"] for row in selected),
            "native_tokens_reused": sum(row["native_tokens_reused"] for row in selected),
            "argmax_parity_vs_host_split": statistics.mean(row["argmax_matches_host_split"] for row in selected),
            "max_abs_logit_delta_vs_host_split": max(row["max_abs_logit_delta_vs_host_split"] for row in selected),
        }
    sessions = build_session_rows(rows)
    return {
        "protocol": "paper9-mlx-live-reuse-v1",
        "model": model_id,
        "engine": "mlx-lm",
        "seeds": list(seeds),
        "shared_token_targets": sorted({row["target_shared_tokens"] for row in rows}),
        "fanouts": sorted({row["fanout"] for row in rows}),
        "scope": "Live host-model forward/prefill and post-RoPE native K/V; not an HTTP serving TTFT benchmark.",
        "conditions": by_condition,
        "session_economics": {
            "points": len(sessions),
            "mean_amortized_speedup": statistics.mean(row["amortized_speedup"] for row in sessions),
            "max_amortized_speedup": max(row["amortized_speedup"] for row in sessions),
            "mean_physical_token_reduction": statistics.mean(row["physical_token_reduction"] for row in sessions),
        },
    }


def build_session_rows(rows: list[dict]) -> list[dict]:
    """Charge source encoding once and aggregate all child requests."""

    points = sorted({
        (int(row["seed"]), int(row["target_shared_tokens"]), int(row["fanout"]))
        for row in rows
    })
    sessions = []
    for seed, size, fanout in points:
        selected = [
            row for row in rows
            if int(row["seed"]) == seed
            and int(row["target_shared_tokens"]) == size
            and int(row["fanout"]) == fanout
        ]
        text = [row for row in selected if row["condition"] == "host_split_text"]
        native = [row for row in selected if row["condition"] == "pra_native_kv"]
        encode_ms = float(native[0]["one_time_native_encode_ms"])
        text_ms = sum(float(row["request_ms"]) for row in text)
        native_ms = encode_ms + sum(float(row["request_ms"]) for row in native)
        text_tokens = sum(int(row["physical_input_tokens"]) for row in text)
        native_tokens = int(native[0]["actual_shared_tokens"]) + sum(
            int(row["physical_input_tokens"]) for row in native
        )
        sessions.append({
            "seed": seed,
            "target_shared_tokens": size,
            "actual_shared_tokens": int(native[0]["actual_shared_tokens"]),
            "fanout": fanout,
            "host_split_session_ms": text_ms,
            "native_session_ms": native_ms,
            "one_time_native_encode_ms": encode_ms,
            "amortized_speedup": text_ms / native_ms,
            "host_split_physical_tokens": text_tokens,
            "native_physical_tokens": native_tokens,
            "physical_token_reduction": 1.0 - native_tokens / text_tokens,
            "duplicate_kv_bytes_avoided": max(0, fanout - 1) * int(native[0]["native_kv_bytes"]),
            "native_argmax_parity": statistics.mean(int(row["argmax_matches_host_split"]) for row in native),
        })
    return sessions


def plot(rows: list[dict], session_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    colors = {"harness_memo_text": "#68747d", "pra_record_reprefill": "#b8872d", "pra_native_kv": "#24796b"}
    available_fanouts = sorted({row["fanout"] for row in rows})
    shown_fanouts = [fanout for fanout in (1, 4, 16) if fanout in available_fanouts]
    if not shown_fanouts:
        shown_fanouts = available_fanouts
    for condition in colors:
        for fanout in shown_fanouts:
            points = []
            for size in sorted({row["target_shared_tokens"] for row in rows}):
                selected = [
                    row for row in rows
                    if row["condition"] == condition and row["fanout"] == fanout
                    and row["target_shared_tokens"] == size
                ]
                points.append(statistics.mean(row["request_ms"] for row in selected))
            axes[0].plot(
                sorted({row["target_shared_tokens"] for row in rows}),
                points,
                marker="o",
                color=colors[condition],
                alpha=0.45 + fanout / 32,
                label=f"{condition}, N={fanout}",
            )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Shared source tokens")
    axes[0].set_ylabel("Per-child request latency (ms)")
    axes[0].legend(fontsize=6, ncol=2)

    sizes = sorted({row["target_shared_tokens"] for row in rows})
    ratios = []
    for size in sizes:
        selected = [row["amortized_speedup"] for row in session_rows if row["target_shared_tokens"] == size and row["fanout"] == max(available_fanouts)]
        ratios.append(statistics.mean(selected))
    axes[1].plot(sizes, ratios, marker="o", color="#255b96")
    axes[1].axhline(1.0, color="#333333", linewidth=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Shared source tokens")
    axes[1].set_ylabel(f"Amortized speedup at N={max(available_fanouts)}")
    figure.tight_layout()
    figure.savefig(output / "mlx_live_reuse.pdf", bbox_inches="tight")
    figure.savefig(output / "mlx_live_reuse.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--shared-tokens", default="512,2048,8192,32768")
    parser.add_argument("--fanouts", default="1,2,4,8,16")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-existing", action="store_true")
    arguments = parser.parse_args()
    seeds = tuple(int(value) for value in arguments.seeds.split(",") if value)
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.analyze_existing:
        rows = load_rows(arguments.output / "rows.csv")
    else:
        from mlx_lm import load

        model, tokenizer = load(arguments.model)
        rows: list[dict] = []
        for seed in seeds:
            for target_tokens in (int(value) for value in arguments.shared_tokens.split(",")):
                for fanout in (int(value) for value in arguments.fanouts.split(",")):
                    print(f"seed={seed} shared={target_tokens} fanout={fanout}", flush=True)
                    rows.extend(run_point(model, tokenizer, arguments.model, seed, target_tokens, fanout))
        with (arguments.output / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    session_rows = build_session_rows(rows)
    with (arguments.output / "session_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(session_rows[0]))
        writer.writeheader()
        writer.writerows(session_rows)
    summary = summarize(rows, arguments.model, seeds)
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(rows, session_rows, arguments.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
