"""Compare incremental-cache and cold-prefill MLX execution on frozen agent turns."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LayerKV:
    keys: object
    values: object


class CapturedKVCache:
    """Immutable captured prefix followed by a fresh request-local MLX cache."""

    def __init__(self, local_cache: object, memory: LayerKV, position_base: int):
        self.local_cache = local_cache
        self.memory = memory
        self.position_base = int(position_base)

    @property
    def offset(self) -> int:
        return self.position_base + int(self.local_cache.offset)

    @property
    def state(self):
        local = self.local_cache.state
        if not isinstance(local, tuple):
            local = tuple(local)
        return self.memory.keys, self.memory.values, *local

    @state.setter
    def state(self, value) -> None:
        raise RuntimeError("captured prefix is immutable")

    @property
    def nbytes(self) -> int:
        return int(self.memory.keys.nbytes + self.memory.values.nbytes) + int(
            getattr(self.local_cache, "nbytes", 0)
        )

    def empty(self) -> bool:
        return False

    def is_trimmable(self) -> bool:
        operation = getattr(self.local_cache, "is_trimmable", None)
        return bool(operation and operation())

    def trim(self, n: int) -> int:
        return int(self.local_cache.trim(n))

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        local_keys, local_values = self.local_cache.update_and_fetch(keys, values)
        return (
            mx.concatenate((self.memory.keys, local_keys), axis=2),
            mx.concatenate((self.memory.values, local_values), axis=2),
        )

    def make_mask(self, n: int, return_array: bool = False, window_size=None, **kwargs):
        import mlx.core as mx
        from mlx_lm.models.base import create_causal_mask

        local_offset = int(self.local_cache.offset)
        if window_size is None:
            local = create_causal_mask(n, local_offset)
        else:
            local = create_causal_mask(
                n, local_offset, window_size=min(int(window_size), local_offset + n)
            )
        memory = mx.ones((n, int(self.memory.keys.shape[2])), dtype=mx.bool_)
        return mx.concatenate((memory, local), axis=1)


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
    prompts = []
    for index in indexes:
        rendered = tokenizer.apply_chat_template(
            messages[:index], tokenize=True, add_generation_prompt=True
        )
        if isinstance(rendered, str):
            rendered = tokenizer.encode(rendered, add_special_tokens=False)
        prompts.append([int(token) for token in rendered])
    return prompts


def _capture(caches: list[object], token_count: int) -> tuple[LayerKV, ...]:
    import mlx.core as mx

    layers = []
    for cache in caches:
        state = cache.state
        if not isinstance(state, tuple) or len(state) < 2:
            raise RuntimeError("MLX exposed a non-attention cache layer")
        keys, values = state[:2]
        layers.append(LayerKV(keys[:, :, :token_count, :], values[:, :, :token_count, :]))
    mx.eval([(layer.keys, layer.values) for layer in layers])
    return tuple(layers)


def _make_captured_cache(model, layers: tuple[LayerKV, ...], token_count: int):
    from mlx_lm.models.cache import make_prompt_cache

    local = make_prompt_cache(model)
    if len(local) != len(layers):
        raise RuntimeError("captured state does not match model layer count")
    return [
        CapturedKVCache(cache, layer, token_count)
        for cache, layer in zip(local, layers)
    ]


def _generate(model, logits, cache, tokens: int):
    import mlx.core as mx

    generated = []
    step_logits = []
    for _ in range(tokens):
        current = logits[0, -1]
        mx.eval(current)
        step_logits.append(np.asarray(current.astype(mx.float32)))
        token = int(mx.argmax(current).item())
        generated.append(token)
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    return generated, step_logits


def _max_delta(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    return max(float(np.max(np.abs(a - b))) for a, b in zip(left, right))


def run(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(args.model)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turns)
    prior_ids: list[int] = []
    live_cache = None
    rows = []
    for turn, prompt_ids in enumerate(prompts, start=1):
        common = _common_prefix(prior_ids, prompt_ids)
        if live_cache is None or common == 0:
            live_cache = make_prompt_cache(model)
        else:
            remove = len(prior_ids) - common
            if remove:
                for cache in live_cache:
                    if int(cache.trim(remove)) != remove:
                        raise RuntimeError("could not crop incremental MLX cache")
        tail = prompt_ids[common:] if common else prompt_ids
        if not tail:
            raise RuntimeError("agent prompt did not extend after cache crop")
        started = time.perf_counter()
        live_logits = model(mx.array([tail], dtype=mx.int32), cache=live_cache)
        mx.eval(live_logits)
        live_prefill_ms = (time.perf_counter() - started) * 1000

        captured = _capture(live_cache, len(prompt_ids))
        live_generation_cache = _make_captured_cache(model, captured, len(prompt_ids))

        # Finish the live-cache continuation before allocating a full cold
        # cache.  Three simultaneous long-context caches can force a 16 GB
        # unified-memory Mac into swap and make the smoke test unresponsive.
        live_tokens, live_steps = _generate(
            model, live_logits, live_generation_cache, args.continuation_tokens
        )
        del live_generation_cache

        cold_cache = make_prompt_cache(model)
        started = time.perf_counter()
        cold_logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cold_cache)
        mx.eval(cold_logits)
        cold_prefill_ms = (time.perf_counter() - started) * 1000

        cold_tokens, cold_steps = _generate(
            model, cold_logits, cold_cache, args.continuation_tokens
        )
        rows.append({
            "turn": turn,
            "prompt_tokens": len(prompt_ids),
            "reused_prefix_tokens": common,
            "new_prompt_tokens": len(prompt_ids) - common,
            "token_exact": live_tokens == cold_tokens,
            "max_abs_logit_delta": _max_delta(live_steps, cold_steps),
            "live_token_ids": live_tokens,
            "cold_token_ids": cold_tokens,
            "incremental_prefill_ms": live_prefill_ms,
            "cold_prefill_ms": cold_prefill_ms,
        })
        prior_ids = prompt_ids
        mx.clear_cache()

    result = {
        "schema_version": 1,
        "probe": "mlx_frozen_agent_incremental_cache_equivalence",
        "model": args.model,
        "engine": "mlx-lm",
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "hardware": args.hardware_label,
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
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--hardware-label", default="unspecified")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in (
        "model", "hardware", "completed_turns", "exact_turns",
        "all_exact", "first_divergent_turn",
    )}, indent=2))
    raise SystemExit(0 if result["all_exact"] else 1)


if __name__ == "__main__":
    main()
