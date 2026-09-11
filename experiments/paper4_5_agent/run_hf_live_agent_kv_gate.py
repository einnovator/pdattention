"""Qualify HF agent-history PRA from the same resident DynamicCache."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pra_hf.hf_live_kv import select_dynamic_cache
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


@torch.inference_mode()
def _positioned_tail_generation(
    model,
    cache,
    tail: list[int],
    *,
    position_base: int,
    continuation_tokens: int,
    device: torch.device,
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
        position = torch.tensor(
            [position_base + len(tail) + step], dtype=torch.long, device=device
        )
        output = model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            past_key_values=cache,
            position_ids=position.unsqueeze(0),
            cache_position=position,
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device).eval()
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
        selected = select_dynamic_cache(source_cache, plan)
        if wire_tail:
            reference = select_dynamic_cache(_clone_cache(source_cache), plan)
            ordinary_tokens, ordinary_logits = _positioned_tail_generation(
                model,
                reference.cache,
                wire_tail,
                position_base=plan.source_position_base,
                continuation_tokens=args.continuation_tokens,
                device=device,
            )
            pra_tokens, pra_logits = _positioned_tail_generation(
                model,
                selected.cache,
                wire_tail,
                position_base=plan.source_position_base,
                continuation_tokens=args.continuation_tokens,
                device=device,
            )
        else:
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
                "selected_kv_tokens": plan.selected_tokens,
                "source_position_base": plan.source_position_base,
                "has_holes": plan.has_holes,
                "selection_plan": plan.to_dict(),
                "selected_text_reencoded_tokens": selected.selected_text_reencoded_tokens,
                "physical_kv_copy": selected.physical_kv_copy,
                "token_exact": ordinary_tokens == pra_tokens,
                "max_abs_logit_delta": _max_delta(ordinary_logits, pra_logits),
                "ordinary_token_ids": ordinary_tokens,
                "pra_token_ids": pra_tokens,
            }
        )
        prior_ids = source_ids

    result = {
        "schema_version": "paper4.5.agent-history-kv-gate.v1",
        "probe": "hf_same_state_live_agent_kv",
        "engine": "transformers-pytorch",
        "model": args.model,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "dtype": args.dtype,
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--continuation-tokens", type=int, default=32)
    parser.add_argument("--retention-fraction", type=float, default=1.0)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
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
    )}, indent=2))
    raise SystemExit(0 if result["all_exact"] else 1)


if __name__ == "__main__":
    main()
