"""Qualify HF agent-history PRA from the same resident DynamicCache."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pra_hf.hf_live_kv import (
    dense_reference_attention_mask,
    enable_qwen_sparse_live_kv,
    pack_segmented_dynamic_cache_reference,
    select_dynamic_cache,
    select_full_dynamic_cache_noop,
)
from pra_hf.live_history import LiveKVSelectionPlan

from .run_hf_agent_cache_equivalence import (
    _assistant_prompts,
    _clone_cache,
    _common_prefix,
    _generate_from_logits,
    _max_delta,
    _prefill,
)
from .sparse_gate_common import sparse_causal_plan


def _cache_bytes(cache: object) -> int:
    """Count a measurement fork without charging it to PRA attachment."""

    total = 0
    for layer in getattr(cache, "layers", ()):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue
        total += int(keys.numel()) * int(keys.element_size())
        total += int(values.numel()) * int(values.element_size())
    return total


@torch.inference_mode()
def _positioned_tail_generation(
    model,
    cache,
    tail: list[int],
    *,
    position_base: int,
    continuation_tokens: int,
    device: torch.device,
    dense_reference_plan: LiveKVSelectionPlan | None = None,
):
    """Consume a wire tail while keeping queries in the source position frame."""

    positions = torch.arange(
        position_base,
        position_base + len(tail),
        dtype=torch.long,
        device=device,
    )
    output = model(
        input_ids=torch.tensor([tail], dtype=torch.long, device=device),
        past_key_values=cache,
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        attention_mask=(
            dense_reference_attention_mask(
                dense_reference_plan,
                positions,
                dtype=next(model.parameters()).dtype,
                device=device,
            )
            if dense_reference_plan is not None
            else None
        ),
        use_cache=True,
        return_dict=True,
    )
    generated: list[int] = []
    step_logits: list[torch.Tensor] = []
    logits = output.logits
    cache = output.past_key_values
    for step in range(continuation_tokens):
        current = logits[:, -1, :]
        step_logits.append(current.detach().float().cpu())
        token = int(torch.argmax(current, dim=-1).item())
        generated.append(token)
        if step + 1 == continuation_tokens:
            break
        position = torch.tensor(
            [position_base + len(tail) + step], dtype=torch.long, device=device
        )
        output = model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            past_key_values=cache,
            position_ids=position.unsqueeze(0),
            cache_position=position,
            attention_mask=(
                dense_reference_attention_mask(
                    dense_reference_plan,
                    position,
                    dtype=next(model.parameters()).dtype,
                    device=device,
                )
                if dense_reference_plan is not None
                else None
            ),
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits
        cache = output.past_key_values
    return generated, step_logits


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    load_kwargs = {"local_files_only": args.local_files_only}
    if args.revision:
        load_kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        **load_kwargs,
    ).to(device).eval()
    enable_qwen_sparse_live_kv(model)
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
            source = _prefill(model, source_ids, device)
        else:
            source_cache.crop(common)
            tail = source_ids[common:]
            if not tail:
                raise RuntimeError("agent prompt did not extend after cache crop")
            source = model(
                input_ids=torch.tensor([tail], dtype=torch.long, device=device),
                past_key_values=source_cache,
                use_cache=True,
                return_dict=True,
            )
        source_cache = source.past_key_values
        try:
            plan = sparse_causal_plan(
                tokenizer,
                messages[:assistant_index],
                prompt_ids,
                source_tokens=len(source_ids),
                retention_fraction=args.retention_fraction,
            )
        except RuntimeError:
            # Early turns may contain only the mandatory preamble and active
            # groups. They are not evidence for a sparse mechanism gate.
            prior_ids = source_ids
            continue
        same_state_fork_bytes = 0
        dense_semantic_noop_identity = False
        if plan.full_retention:
            if wire_tail:
                raise RuntimeError("PRA-100 same-state gate unexpectedly split a wire tail.")
            # The two clones are measurement-only forks from one resident K/V
            # state. They are not part of production selection or attachment.
            ordinary_cache = _clone_cache(source_cache)
            candidate_cache = _clone_cache(source_cache)
            same_state_fork_bytes = _cache_bytes(ordinary_cache) + _cache_bytes(
                candidate_cache
            )
            selected = select_full_dynamic_cache_noop(candidate_cache, plan)
            dense_semantic_noop_identity = selected.cache is candidate_cache
            ordinary_tokens, ordinary_logits = _generate_from_logits(
                model,
                source.logits,
                ordinary_cache,
                args.continuation_tokens,
                device,
            )
            pra_tokens, pra_logits = _generate_from_logits(
                model,
                source.logits,
                selected.cache,
                args.continuation_tokens,
                device,
            )
        else:
            selected = select_dynamic_cache(source_cache, plan)
        if wire_tail:
            reference = pack_segmented_dynamic_cache_reference(
                _clone_cache(source_cache), plan
            )
            ordinary_tokens, ordinary_logits = _positioned_tail_generation(
                model,
                reference.cache,
                wire_tail,
                position_base=plan.source_position_base,
                continuation_tokens=args.continuation_tokens,
                device=device,
                dense_reference_plan=None,
            )
            pra_tokens, pra_logits = _positioned_tail_generation(
                model,
                selected.cache,
                wire_tail,
                position_base=plan.source_position_base,
                continuation_tokens=args.continuation_tokens,
                device=device,
            )
        elif not plan.full_retention:
            ordinary_tokens, ordinary_logits = _generate_from_logits(
                model,
                source.logits,
                _clone_cache(source_cache),
                args.continuation_tokens,
                device,
            )
            pra_tokens, pra_logits = _generate_from_logits(
                model,
                source.logits,
                selected.cache,
                args.continuation_tokens,
                device,
            )
        rows.append(
            {
                "turn": turn,
                "source_tokens": len(source_ids),
                "wire_suffix_tokens": len(wire_tail),
                "source_prefix_reused_tokens": common,
                "new_history_encoded_tokens": len(source_ids) - common,
                "selected_kv_tokens": plan.selected_tokens,
                "source_position_base": plan.source_position_base,
                "decode_position_start": plan.source_position_base,
                "decode_position_end": (
                    plan.source_position_base
                    + len(wire_tail)
                    + args.continuation_tokens
                    - 1
                ),
                "has_holes": plan.has_holes,
                "selection_plan": plan.to_dict(),
                "selected_text_reencoded_tokens": selected.selected_text_reencoded_tokens,
                "physical_kv_copy": bool(
                    selected.physical_kv_copy
                    or selected.transient_kv_copy_bytes > 0
                ),
                "persistent_interval_pack": selected.physical_kv_copy,
                "interval_pack_bytes": selected.interval_pack_bytes,
                "selected_history_kv_copy_bytes": (
                    int(selected.interval_pack_bytes)
                    if selected.physical_kv_copy
                    else 0
                ),
                "measurement_state_fork_copy_bytes": same_state_fork_bytes,
                "measurement_state_fork_copy_scope": (
                    "two isolated oracle branches; excluded from production PRA attachment"
                ),
                "dense_semantic_noop_identity": dense_semantic_noop_identity,
                "transient_attention_bytes": selected.transient_attention_bytes,
                "transient_kv_copy_bytes": selected.transient_kv_copy_bytes,
                "max_transient_kv_tile_bytes": selected.max_transient_kv_tile_bytes,
                "request_tail_copy_bytes": selected.request_tail_copy_bytes,
                "fused_attention_calls": selected.fused_attention_calls,
                "token_exact": ordinary_tokens == pra_tokens,
                "max_abs_logit_delta": _max_delta(ordinary_logits, pra_logits),
                "ordinary_token_ids": ordinary_tokens,
                "pra_token_ids": pra_tokens,
            }
        )
        prior_ids = source_ids

    result = {
        "schema_version": "paper4.5.agent-history-kv-gate.v2",
        "probe": (
            "hf_same_state_live_agent_kv_100"
            if args.retention_fraction == 1
            else "hf_same_state_live_agent_kv_sparse"
        ),
        "engine": "transformers-pytorch",
        "model": args.model_label or args.model,
        "model_source": args.model,
        "model_revision": args.revision or getattr(model.config, "_commit_hash", None),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "trajectory": str(args.trajectory),
        "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(inspect.getfile(select_dynamic_cache)).read_bytes()
        ).hexdigest(),
        "retention_fraction": args.retention_fraction,
        "adaptor": (
            "dense_semantic_noop"
            if args.retention_fraction == 1
            else "original_position_sparse_views"
        ),
        "same_resident_kv_fork": True,
        "completed_turns": len(rows),
        "exact_turns": sum(int(row["token_exact"]) for row in rows),
        "all_exact": all(row["token_exact"] for row in rows),
        "zero_selected_text_reencoding": all(
            row["selected_text_reencoded_tokens"] == 0 for row in rows
        ),
        "selected_history_reencoded_tokens": sum(
            int(row["selected_text_reencoded_tokens"]) for row in rows
        ),
        "selected_history_kv_copy_bytes": sum(
            int(row["selected_history_kv_copy_bytes"]) for row in rows
        ),
        "zero_physical_kv_copy": all(
            not row["physical_kv_copy"] for row in rows
        ),
        "zero_persistent_interval_pack": all(
            not row["persistent_interval_pack"] for row in rows
        ),
        "zero_interval_pack_bytes": all(
            row["interval_pack_bytes"] == 0 for row in rows
        ),
        "transient_attention_bytes": sum(
            int(row["transient_attention_bytes"]) for row in rows
        ),
        "transient_kv_copy_bytes": sum(
            int(row["transient_kv_copy_bytes"]) for row in rows
        ),
        "max_transient_kv_tile_bytes": max(
            (int(row["max_transient_kv_tile_bytes"]) for row in rows), default=0
        ),
        "request_tail_copy_bytes": sum(
            int(row["request_tail_copy_bytes"]) for row in rows
        ),
        "fused_attention_calls": sum(
            int(row["fused_attention_calls"]) for row in rows
        ),
        "first_divergent_turn": next(
            (row["turn"] for row in rows if not row["token_exact"]), None
        ),
        "measurement_state_fork_copy_bytes": sum(
            int(row["measurement_state_fork_copy_bytes"]) for row in rows
        ),
        "measurement_state_fork_copy_scope": (
            "two isolated oracle branches from one resident cache; excluded from production PRA attachment"
        ),
        "reference_condition": (
            "ordinary dense continuation and PRA-100 dense semantic no-op forked from the identical resident DynamicCache"
            if args.retention_fraction == 1
            else "packed-value oracle preserving the candidate segment boundaries and original positions under the identical sparse consumer"
        ),
        "consumer_implementation": (
            "ordinary_dense_attention_semantic_noop"
            if args.retention_fraction == 1
            else (
                "fused_interval_addressed_triton"
                if device.type == "cuda"
                else "two_pass_segmented_sdpa"
            )
        ),
        "sparse_numerical_policy": (
            "one fused Triton online softmax over canonical source intervals and request-local K/V"
            if device.type == "cuda"
            else "two-pass native SDPA over storage-alias K/V views with fp32 cross-segment normalization and zero K/V casts"
        ),
        "sparse_turns": sum(int(row["has_holes"]) for row in rows),
        "rows": rows,
    }
    result["max_abs_logit_delta"] = max(
        (float(row["max_abs_logit_delta"]) for row in rows), default=0.0
    )
    result["copy_accounting"] = {
        "selected_history_reencoded_tokens": result[
            "selected_history_reencoded_tokens"
        ],
        "selected_history_kv_copy_bytes": result[
            "selected_history_kv_copy_bytes"
        ],
        "physical_kv_copy_bytes": 0 if result["zero_physical_kv_copy"] else None,
        "host_to_device_bytes": 0,
        "interval_pack_bytes": sum(int(row["interval_pack_bytes"]) for row in rows),
        "measurement_state_fork_copy_bytes": result[
            "measurement_state_fork_copy_bytes"
        ],
        "measurement_state_fork_in_production_path": False,
    }
    result["logit_tolerance"] = args.max_logit_delta
    result["logits_within_tolerance"] = bool(
        result["max_abs_logit_delta"] <= args.max_logit_delta
    )
    result["pra100_same_state_gate_valid"] = bool(
        args.retention_fraction == 1
        and len(rows) == args.turns
        and result["all_exact"]
        and result["max_abs_logit_delta"] == 0.0
        and result["zero_selected_text_reencoding"]
        and result["zero_physical_kv_copy"]
        and result["zero_interval_pack_bytes"]
        and all(row["selection_plan"]["full_retention"] for row in rows)
        and all(row["dense_semantic_noop_identity"] for row in rows)
    )
    result["sparse_position_gate_valid"] = bool(
        args.retention_fraction < 1
        and result["all_exact"]
        and result["logits_within_tolerance"]
        and result["zero_selected_text_reencoding"]
        and result["zero_interval_pack_bytes"]
        and result["sparse_turns"] > 0
        and all(
            row["source_position_base"] == row["source_tokens"] for row in rows
        )
    )
    result["zero_copy_engine_gate_valid"] = bool(
        (
            result["pra100_same_state_gate_valid"]
            or result["sparse_position_gate_valid"]
        )
        and result["zero_physical_kv_copy"]
    )
    result["qualification_blockers"] = []
    if not result["logits_within_tolerance"]:
        result["qualification_blockers"].append(
            "max_abs_logit_delta_exceeds_tolerance"
        )
    if not result["zero_physical_kv_copy"]:
        result["qualification_blockers"].append(
            "bounded_fp32_kv_tile_copy_requires_fused_native_sparse_attention"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-label")
    parser.add_argument("--revision")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--continuation-tokens", type=int, default=32)
    parser.add_argument("--retention-fraction", type=float, default=1.0)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--max-logit-delta", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in (
        "engine", "model", "completed_turns", "exact_turns", "all_exact",
        "zero_selected_text_reencoding", "first_divergent_turn",
        "max_abs_logit_delta", "logits_within_tolerance",
    )}, indent=2))
    raise SystemExit(0 if result["zero_copy_engine_gate_valid"] else 1)


if __name__ == "__main__":
    main()
