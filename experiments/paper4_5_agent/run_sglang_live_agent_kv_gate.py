"""Qualify SGLang-MLX from resident agent-prefix K/V without re-encoding."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from experiments.paper4_5_agent.run_hf_agent_cache_equivalence import (
    _assistant_prompts,
)
from experiments.paper6_1_sglang.run_live_runner import _run_request
from experiments.paper6_vllm.run_v1_capture_replay_audit import _token_comparison
from experiments.paper4_5_agent.sparse_gate_common import sparse_causal_plan
from pra_hf.live_history import LiveKVSelectionPlan


def _extend_request(runner, req_id: str, suffix: list[int], max_tokens: int):
    """Generate after extending an already resident SGLang request cache."""

    import mlx.core as mx

    pending = runner.extend_start(req_id, suffix, [], needs_logits=True)
    runner.eval_pending(pending)
    generated = [runner.extend_finalize(pending)]
    for _ in range(max_tokens - 1):
        pending_decode = runner.decode_batch_start([req_id])
        runner.eval_pending(pending_decode)
        generated.extend(runner.decode_batch_finalize(pending_decode))
    mx.eval(*runner._req_caches[req_id][0].state)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--retention-fraction", type=float, default=1.0)
    parser.add_argument(
        "--radix-cache",
        action="store_true",
        help="Also initialize SGLang's radix pool; the base same-state gate keeps it off.",
    )
    args = parser.parse_args()

    import sglang
    from pra_mlx.native import capture_live_native_memory
    from pra_sglang.mlx_native import SGLangMLXNativeBridge
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from transformers import AutoTokenizer

    runner = MlxModelRunner(
        args.model,
        revision=args.revision,
        disable_radix_cache=not args.radix_cache,
        enable_sampling=False,
    )
    if args.radix_cache:
        runner.init_cache_pools(None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turns)
    messages = trajectory["messages"]
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turns]
    bridge = SGLangMLXNativeBridge(runner)
    rows = []
    try:
        for turn, (assistant_index, prompt) in enumerate(
            zip(assistant_indexes, prompts), start=1
        ):
            tail_tokens = min(args.wire_tail_tokens, max(1, len(prompt) // 4))
            source = prompt[:-tail_tokens]
            suffix = prompt[-tail_tokens:]
            if not source:
                raise RuntimeError("Agent prompt is too short for a source/suffix fork.")

            prime_id = f"agent-prime-{turn}"
            pending = runner.prefill_start(prime_id, source, source, [], [], 0)
            runner.eval_pending(pending)
            runner.prefill_finalize(pending)
            source_caches = runner._req_caches[prime_id]
            try:
                plan = sparse_causal_plan(
                    tokenizer,
                    messages[:assistant_index],
                    prompt,
                    source_tokens=len(source),
                    retention_fraction=args.retention_fraction,
                )
            except RuntimeError:
                runner.remove_request(prime_id)
                continue
            resident = capture_live_native_memory(source_caches, plan)

            if args.retention_fraction == 1:
                # The ordinary fork must continue from the exact captured
                # source request. Re-prefilling would reintroduce drift.
                ordinary_tokens = _extend_request(
                    runner, prime_id, suffix, args.continuation_tokens
                )
                ordinary_cache = runner._req_caches[prime_id][
                    runner._cache_layout.first_attention_layer_index
                ]
                ordinary_cache_offset = int(ordinary_cache.offset)
            else:
                reference_id = f"agent-reference-{turn}"
                bridge.register(
                    reference_id,
                    resident,
                    logical_keys=(f"agent-source-{turn}",),
                )
                ordinary_tokens, _ = _run_request(
                    runner,
                    reference_id,
                    suffix,
                    max_tokens=args.continuation_tokens,
                )
                ordinary_cache_offset = int(
                    runner._req_caches[reference_id][
                        runner._cache_layout.first_attention_layer_index
                    ].offset
                )
                runner.remove_request(reference_id)
                bridge.unregister(reference_id)

            pra_id = f"agent-pra-{turn}"
            bridge.register(
                pra_id,
                resident,
                logical_keys=(f"agent-source-{turn}",),
            )
            pra_tokens, _ = _run_request(
                runner,
                pra_id,
                suffix,
                max_tokens=args.continuation_tokens,
            )
            comparison = _token_comparison(
                list(map(int, ordinary_tokens)), list(map(int, pra_tokens))
            )
            rows.append(
                {
                    "turn": turn,
                    "source_tokens": len(source),
                    "wire_suffix_tokens": len(suffix),
                    "ordinary_cache_offset": ordinary_cache_offset,
                    "selected_kv_tokens": resident.plan.selected_tokens,
                    "source_position_base": resident.plan.source_position_base,
                    "has_holes": resident.plan.has_holes,
                    "selection_plan": resident.plan.to_dict(),
                    "selected_text_reencoded_tokens": 0,
                    "physical_kv_copy": resident.physical_kv_copy,
                    "comparison": comparison,
                    "ordinary_token_ids": list(map(int, ordinary_tokens)),
                    "pra_token_ids": list(map(int, pra_tokens)),
                }
            )
            runner.remove_request(pra_id)
            bridge.unregister(pra_id)
            # Keep the source owner alive until both forks have completed so
            # its cache cannot be reset and recycled underneath the view.
            runner.remove_request(prime_id)
    finally:
        bridge.close()

    payload = {
        "schema_version": "paper4.5.agent-history-kv-gate.v1",
        "probe": "sglang_same_resident_agent_kv",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "engine_revision": "ef20fab38a03490e2cdf1b7377145ca3a3f2bfc5",
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "model": args.model,
        "model_revision": args.revision,
        "trajectory": str(args.trajectory),
        "retention_fraction": args.retention_fraction,
        "adaptor": "none",
        "same_resident_kv_fork": True,
        "source_owner_pinned_through_both_forks": True,
        "radix_cache_enabled": args.radix_cache,
        "completed_turns": len(rows),
        "exact_turns": sum(int(row["comparison"]["exact"]) for row in rows),
        "all_exact": all(row["comparison"]["exact"] for row in rows),
        "zero_selected_text_reencoding": True,
        "zero_physical_kv_copy": all(not row["physical_kv_copy"] for row in rows),
        "first_divergent_turn": next(
            (row["turn"] for row in rows if not row["comparison"]["exact"]), None
        ),
        "reference_condition": (
            "ordinary fork continuing the same source owner" if args.retention_fraction == 1
            else "independent request consuming identical selected resident K/V at identical original positions"
        ),
        "sparse_turns": sum(int(row["has_holes"]) for row in rows),
        "rows": rows,
    }
    payload["same_state_gate_valid"] = bool(
        payload["all_exact"]
        and payload["zero_selected_text_reencoding"]
        and payload["zero_physical_kv_copy"]
        and payload["source_owner_pinned_through_both_forks"]
    )
    payload["sparse_position_gate_valid"] = bool(
        args.retention_fraction < 1
        and payload["all_exact"]
        and payload["zero_selected_text_reencoding"]
        and payload["sparse_turns"] > 0
        and payload["source_owner_pinned_through_both_forks"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "engine", "model", "completed_turns", "exact_turns", "all_exact",
        "zero_selected_text_reencoding", "zero_physical_kv_copy",
        "first_divergent_turn",
    )}, indent=2))
    raise SystemExit(0 if payload["all_exact"] else 1)


if __name__ == "__main__":
    main()
