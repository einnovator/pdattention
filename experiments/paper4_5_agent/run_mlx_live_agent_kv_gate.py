"""Qualify MLX agent-history PRA from the same resident prompt cache."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from pra_hf.live_history import LiveKVSelectionPlan

from .run_mlx_agent_cache_equivalence import (
    CapturedKVCache,
    LayerKV,
    _assistant_prompts,
    _capture,
    _common_prefix,
    _generate,
    _max_delta,
)
from .sparse_gate_common import sparse_causal_plan


def _ordinary_cache(model, layers: tuple[LayerKV, ...], source_tokens: int):
    from mlx_lm.models.cache import make_prompt_cache

    local = make_prompt_cache(model)
    return [
        CapturedKVCache(cache, layer, source_tokens)
        for cache, layer in zip(local, layers)
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx.native import capture_live_native_memory, make_native_prompt_cache

    model, tokenizer = load(args.model)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turns)
    messages = trajectory["messages"]
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turns]
    prior_ids: list[int] = []
    source_cache = None
    rows = []
    for turn, (assistant_index, prompt_ids) in enumerate(
        zip(assistant_indexes, prompts), start=1
    ):
        tail_tokens = 0
        if args.retention_fraction < 1:
            tail_tokens = min(args.wire_tail_tokens, max(1, len(prompt_ids) // 4))
        source_ids = prompt_ids[:-tail_tokens] if tail_tokens else prompt_ids
        wire_tail = prompt_ids[-tail_tokens:] if tail_tokens else []
        common = _common_prefix(prior_ids, source_ids)
        if source_cache is None or common == 0:
            source_cache = make_prompt_cache(model)
        else:
            remove = len(prior_ids) - common
            if remove:
                for cache in source_cache:
                    if int(cache.trim(remove)) != remove:
                        raise RuntimeError("could not crop incremental MLX cache")
        tail = source_ids[common:] if common else source_ids
        if not tail:
            raise RuntimeError("agent prompt did not extend after cache crop")
        source_logits = model(mx.array([tail], dtype=mx.int32), cache=source_cache)
        mx.eval(source_logits)

        try:
            plan = sparse_causal_plan(
                tokenizer,
                messages[:assistant_index],
                prompt_ids,
                source_tokens=len(source_ids),
                retention_fraction=args.retention_fraction,
            )
        except RuntimeError:
            prior_ids = source_ids
            continue
        resident = capture_live_native_memory(source_cache, plan)
        if wire_tail:
            reference = capture_live_native_memory(source_cache, plan)
            reference_cache = make_native_prompt_cache(
                model,
                reference.memory,
                query_position_base=plan.source_position_base,
            )
            reference_logits = model(
                mx.array([wire_tail], dtype=mx.int32), cache=reference_cache
            )
            ordinary_tokens, ordinary_logits = _generate(
                model, reference_logits, reference_cache, args.continuation_tokens
            )
            pra_cache = make_native_prompt_cache(
                model,
                resident.memory,
                query_position_base=resident.plan.source_position_base,
            )
            pra_logits_source = model(
                mx.array([wire_tail], dtype=mx.int32), cache=pra_cache
            )
            pra_tokens, pra_logits = _generate(
                model, pra_logits_source, pra_cache, args.continuation_tokens
            )
        else:
            captured = _capture(source_cache, len(source_ids))
            ordinary_tokens, ordinary_logits = _generate(
                model,
                source_logits,
                _ordinary_cache(model, captured, len(source_ids)),
                args.continuation_tokens,
            )
            pra_tokens, pra_logits = _generate(
                model,
                source_logits,
                make_native_prompt_cache(
                    model,
                    resident.memory,
                    query_position_base=resident.plan.source_position_base,
                ),
                args.continuation_tokens,
            )
        rows.append(
            {
                "turn": turn,
                "source_tokens": len(source_ids),
                "wire_suffix_tokens": len(wire_tail),
                "source_prefix_reused_tokens": common,
                "selected_kv_tokens": resident.plan.selected_tokens,
                "source_position_base": resident.plan.source_position_base,
                "has_holes": resident.plan.has_holes,
                "selection_plan": resident.plan.to_dict(),
                "selected_text_reencoded_tokens": resident.selected_text_reencoded_tokens,
                "physical_kv_copy": resident.physical_kv_copy,
                "token_exact": ordinary_tokens == pra_tokens,
                "max_abs_logit_delta": _max_delta(ordinary_logits, pra_logits),
                "ordinary_token_ids": ordinary_tokens,
                "pra_token_ids": pra_tokens,
            }
        )
        prior_ids = source_ids
        mx.clear_cache()

    result = {
        "schema_version": "paper4.5.agent-history-kv-gate.v1",
        "probe": "mlx_same_state_live_agent_kv",
        "engine": "mlx-lm",
        "model": args.model,
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "hardware": args.hardware_label,
        "trajectory": str(args.trajectory),
        "retention_fraction": args.retention_fraction,
        "adaptor": "none",
        "same_resident_kv_fork": True,
        "completed_turns": len(rows),
        "exact_turns": sum(int(row["token_exact"]) for row in rows),
        "all_exact": all(row["token_exact"] for row in rows),
        "zero_selected_text_reencoding": all(
            row["selected_text_reencoded_tokens"] == 0 for row in rows
        ),
        "zero_physical_kv_copy": all(
            not row["physical_kv_copy"] for row in rows
        ),
        "first_divergent_turn": next(
            (row["turn"] for row in rows if not row["token_exact"]), None
        ),
        "reference_condition": "detached clone consuming identical selected resident K/V at identical original positions",
        "sparse_turns": sum(int(row["has_holes"]) for row in rows),
        "rows": rows,
    }
    result["sparse_position_gate_valid"] = bool(
        args.retention_fraction < 1
        and result["all_exact"]
        and result["zero_selected_text_reencoding"]
        and result["sparse_turns"] > 0
        and all(
            row["source_position_base"] == row["source_tokens"] for row in rows
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--retention-fraction", type=float, default=1.0)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--hardware-label", default="unspecified")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in (
        "engine", "model", "completed_turns", "exact_turns", "all_exact",
        "zero_selected_text_reencoding", "first_divergent_turn",
    )}, indent=2))
    raise SystemExit(0 if result["all_exact"] else 1)


if __name__ == "__main__":
    main()
