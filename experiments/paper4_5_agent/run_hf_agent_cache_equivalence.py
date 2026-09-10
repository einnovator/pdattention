"""Compare incremental-cache and cold-prefill execution on frozen agent turns."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def _common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _assistant_prompts(tokenizer, trajectory: dict, turns: int) -> list[list[int]]:
    messages = trajectory["messages"]
    indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][:turns]
    prompts: list[list[int]] = []
    for index in indexes:
        rendered = tokenizer.apply_chat_template(
            messages[:index], tokenize=True, add_generation_prompt=True
        )
        # Transformers 4 returns token IDs here.  Some Transformers 5
        # tokenizers instead return the rendered string despite tokenize=True.
        if isinstance(rendered, str):
            rendered = tokenizer.encode(rendered, add_special_tokens=False)
        elif isinstance(rendered, Mapping):
            rendered = rendered["input_ids"]
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], list):
            if len(rendered) != 1:
                raise ValueError("chat template returned a batched prompt")
            rendered = rendered[0]
        prompts.append([int(token) for token in rendered])
    return prompts


@torch.inference_mode()
def _prefill(model, ids: list[int], device: torch.device):
    return model(
        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
        use_cache=True,
        return_dict=True,
    )


@torch.inference_mode()
def _generate_from_logits(model, logits, cache, tokens: int, device: torch.device):
    generated: list[int] = []
    step_logits: list[torch.Tensor] = []
    for _ in range(tokens):
        current = logits[:, -1, :]
        step_logits.append(current.detach().float().cpu())
        token = int(torch.argmax(current, dim=-1).item())
        generated.append(token)
        output = model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits
        cache = output.past_key_values
    return generated, step_logits


def _max_delta(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return max(float(torch.max(torch.abs(a - b)).item()) for a, b in zip(left, right))


def _clone_cache(cache):
    """Clone a Transformers cache without relying on Tensor.__deepcopy__."""
    cloned = copy.copy(cache)
    if hasattr(cache, "layers"):
        cloned.layers = []
        for layer in cache.layers:
            cloned_layer = copy.copy(layer)
            for name, value in vars(layer).items():
                if torch.is_tensor(value):
                    value = value.clone()
                elif isinstance(value, list):
                    value = [item.clone() if torch.is_tensor(item) else item for item in value]
                setattr(cloned_layer, name, value)
            cloned.layers.append(cloned_layer)
        return cloned
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cloned.key_cache = [tensor.clone() for tensor in cache.key_cache]
        cloned.value_cache = [tensor.clone() for tensor in cache.value_cache]
        return cloned
    raise TypeError(f"unsupported cache type: {type(cache)!r}")


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=args.local_files_only
    ).to(device).eval()
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turns)

    prior_ids: list[int] = []
    incremental_cache = None
    rows = []
    for turn, prompt_ids in enumerate(prompts, start=1):
        common = _common_prefix(prior_ids, prompt_ids)
        if incremental_cache is None or common == 0:
            started = time.perf_counter()
            incremental = _prefill(model, prompt_ids, device)
        else:
            incremental_cache.crop(common)
            tail = prompt_ids[common:]
            if not tail:
                raise RuntimeError("agent prompt did not extend after cache crop")
            started = time.perf_counter()
            incremental = model(
                input_ids=torch.tensor([tail], dtype=torch.long, device=device),
                past_key_values=incremental_cache,
                use_cache=True,
                return_dict=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        incremental_prefill_ms = (time.perf_counter() - started) * 1000
        incremental_cache = incremental.past_key_values

        # Generate from the live-cache snapshot before building the cold cache.
        # Holding the persistent state, its generation clone, and a dense cold
        # state simultaneously exceeds an 8 GB agent-smoke GPU at long turns.
        cached_tokens, cached_logits = _generate_from_logits(
            model, incremental.logits, _clone_cache(incremental_cache),
            args.continuation_tokens, device,
        )
        started = time.perf_counter()
        cold = _prefill(model, prompt_ids, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cold_prefill_ms = (time.perf_counter() - started) * 1000
        cold_tokens, cold_logits = _generate_from_logits(
            model, cold.logits, cold.past_key_values,
            args.continuation_tokens, device,
        )
        rows.append({
            "turn": turn,
            "prompt_tokens": len(prompt_ids),
            "reused_prefix_tokens": common,
            "new_prompt_tokens": len(prompt_ids) - common,
            "token_exact": cached_tokens == cold_tokens,
            "max_abs_logit_delta": _max_delta(cached_logits, cold_logits),
            "cached_token_ids": cached_tokens,
            "cold_token_ids": cold_tokens,
            "incremental_prefill_ms": incremental_prefill_ms,
            "cold_prefill_ms": cold_prefill_ms,
        })
        prior_ids = prompt_ids

    result = {
        "schema_version": 1,
        "probe": "hf_frozen_agent_incremental_cache_equivalence",
        "model": args.model,
        "engine": "transformers-pytorch",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "dtype": args.dtype,
        "trajectory": str(args.trajectory),
        "requested_turns": args.turns,
        "continuation_tokens": args.continuation_tokens,
        "completed_turns": len(rows),
        "exact_turns": sum(int(row["token_exact"]) for row in rows),
        "all_exact": all(row["token_exact"] for row in rows),
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
    parser.add_argument("--continuation-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in (
        "model", "device", "dtype", "completed_turns", "exact_turns",
        "all_exact", "first_divergent_turn",
    )}, indent=2))
    raise SystemExit(0 if result["all_exact"] else 1)


if __name__ == "__main__":
    main()
