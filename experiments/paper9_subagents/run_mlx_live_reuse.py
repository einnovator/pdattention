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


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def timed_forward(model, token_ids: list[int], *, cache=None) -> tuple[float, np.ndarray]:
    import mlx.core as mx

    started = time.perf_counter()
    logits = model(mx.array(token_ids, dtype=mx.int32)[None], cache=cache)
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
                    "argmax_matches_isolated": int(np.argmax(logits) == np.argmax(isolated_logits)),
                    "max_abs_logit_delta": float(np.max(np.abs(logits - isolated_logits))),
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
            "argmax_parity": statistics.mean(row["argmax_matches_isolated"] for row in selected),
            "max_abs_logit_delta": max(row["max_abs_logit_delta"] for row in selected),
        }
    return {
        "protocol": "paper9-mlx-live-reuse-v1",
        "model": model_id,
        "engine": "mlx-lm",
        "seeds": list(seeds),
        "shared_token_targets": sorted({row["target_shared_tokens"] for row in rows}),
        "fanouts": sorted({row["fanout"] for row in rows}),
        "scope": "Live host-model forward/prefill and post-RoPE native K/V; not an HTTP serving TTFT benchmark.",
        "conditions": by_condition,
    }


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    colors = {"harness_memo_text": "#68747d", "pra_record_reprefill": "#b8872d", "pra_native_kv": "#24796b"}
    for condition in colors:
        for fanout in (1, 4, 16):
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
        text = [row["request_ms"] for row in rows if row["condition"] == "pra_record_reprefill" and row["target_shared_tokens"] == size]
        native = [row["request_ms"] for row in rows if row["condition"] == "pra_native_kv" and row["target_shared_tokens"] == size]
        ratios.append(statistics.mean(text) / statistics.mean(native))
    axes[1].plot(sizes, ratios, marker="o", color="#255b96")
    axes[1].axhline(1.0, color="#333333", linewidth=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Shared source tokens")
    axes[1].set_ylabel("Record re-prefill / native latency")
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
    arguments = parser.parse_args()
    from mlx_lm import load

    model, tokenizer = load(arguments.model)
    rows: list[dict] = []
    seeds = tuple(int(value) for value in arguments.seeds.split(",") if value)
    for seed in seeds:
        for target_tokens in (int(value) for value in arguments.shared_tokens.split(",")):
            for fanout in (int(value) for value in arguments.fanouts.split(",")):
                print(f"seed={seed} shared={target_tokens} fanout={fanout}", flush=True)
                rows.extend(run_point(model, tokenizer, arguments.model, seed, target_tokens, fanout))
    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, arguments.model, seeds)
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(rows, arguments.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
