"""Qualify HF 100% agent-history PRA from the same resident DynamicCache."""

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

    prior_ids: list[int] = []
    source_cache = None
    rows = []
    for turn, prompt_ids in enumerate(prompts, start=1):
        common = _common_prefix(prior_ids, prompt_ids)
        if source_cache is None or common == 0:
            source = _prefill(model, prompt_ids, device)
        else:
            source_cache.crop(common)
            tail = prompt_ids[common:]
            if not tail:
                raise RuntimeError("agent prompt did not extend after cache crop")
            source = model(
                input_ids=torch.tensor([tail], dtype=torch.long, device=device),
                past_key_values=source_cache,
                use_cache=True,
                return_dict=True,
            )
        source_cache = source.past_key_values
        plan = LiveKVSelectionPlan.full(len(prompt_ids))
        selected = select_dynamic_cache(source_cache, plan)

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
                "source_tokens": len(prompt_ids),
                "source_prefix_reused_tokens": common,
                "selected_kv_tokens": plan.selected_tokens,
                "selected_text_reencoded_tokens": selected.selected_text_reencoded_tokens,
                "physical_kv_copy": selected.physical_kv_copy,
                "token_exact": ordinary_tokens == pra_tokens,
                "max_abs_logit_delta": _max_delta(ordinary_logits, pra_logits),
                "ordinary_token_ids": ordinary_tokens,
                "pra_token_ids": pra_tokens,
            }
        )
        prior_ids = prompt_ids

    result = {
        "schema_version": "paper4.5.agent-history-kv-gate.v1",
        "probe": "hf_same_state_live_agent_kv_100",
        "engine": "transformers-pytorch",
        "model": args.model,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "dtype": args.dtype,
        "trajectory": str(args.trajectory),
        "retention_fraction": 1.0,
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
        "rows": rows,
    }
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

