"""Qualify SGLang-MLX from resident agent-prefix K/V without re-encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper4_5_agent.run_hf_agent_cache_equivalence import (
    _assistant_prompts,
)
from experiments.paper6_1_sglang.run_live_runner import _run_request
from experiments.paper6_vllm.run_v1_capture_replay_audit import _token_comparison
from pra_hf.live_history import LiveKVSelectionPlan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    args = parser.parse_args()

    import sglang
    from pra_mlx.native import capture_live_native_memory
    from pra_sglang.mlx_native import SGLangMLXNativeBridge
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from transformers import AutoTokenizer

    runner = MlxModelRunner(
        args.model,
        revision=args.revision,
        disable_radix_cache=False,
        enable_sampling=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turns)
    bridge = SGLangMLXNativeBridge(runner)
    rows = []
    try:
        for turn, prompt in enumerate(prompts, start=1):
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
            resident = capture_live_native_memory(
                source_caches, LiveKVSelectionPlan.full(len(source))
            )
            runner.remove_request(prime_id)

            ordinary_id = f"agent-ordinary-{turn}"
            ordinary_tokens, _ = _run_request(
                runner,
                ordinary_id,
                prompt,
                max_tokens=args.continuation_tokens,
            )
            ordinary_cache = runner._req_caches[ordinary_id][
                runner._cache_layout.first_attention_layer_index
            ]

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
                    "ordinary_cache_offset": int(ordinary_cache.offset),
                    "selected_kv_tokens": resident.plan.selected_tokens,
                    "selected_text_reencoded_tokens": 0,
                    "physical_kv_copy": resident.physical_kv_copy,
                    "comparison": comparison,
                    "ordinary_token_ids": list(map(int, ordinary_tokens)),
                    "pra_token_ids": list(map(int, pra_tokens)),
                }
            )
            runner.remove_request(ordinary_id)
            runner.remove_request(pra_id)
            bridge.unregister(pra_id)
    finally:
        bridge.close()

    payload = {
        "schema_version": "paper4.5.agent-history-kv-gate.v1",
        "probe": "sglang_same_resident_agent_kv_100",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "model": args.model,
        "model_revision": args.revision,
        "trajectory": str(args.trajectory),
        "retention_fraction": 1.0,
        "adaptor": "none",
        "same_resident_kv_fork": True,
        "completed_turns": len(rows),
        "exact_turns": sum(int(row["comparison"]["exact"]) for row in rows),
        "all_exact": all(row["comparison"]["exact"] for row in rows),
        "zero_selected_text_reencoding": True,
        "zero_physical_kv_copy": all(not row["physical_kv_copy"] for row in rows),
        "first_divergent_turn": next(
            (row["turn"] for row in rows if not row["comparison"]["exact"]), None
        ),
        "rows": rows,
    }
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

