"""Diagnose cached-versus-cold llama.cpp consumption on a frozen agent history.

The two phases deliberately run sequentially on the same explicit server slot.
Interleaving a second slot is not a valid control when llama.cpp runs with a
unified KV cache because idle-slot eviction can erase the supposedly cached
sequence.  Every phase receives the same chat-template-rendered prompt, seed,
and generation parameters.  The cached phase must also demonstrate a physical
cache hit after its first request; cold-vs-cold agreement is not qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .context_treatment import CONSUMPTION_POLICIES, apply_consumption_policy


def _post(base_url: str, path: str, payload: Mapping[str, Any]) -> Any:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def _erase(base_url: str, slot: int) -> None:
    _post(base_url, f"/slots/{slot}?action=erase", {})


def _digest(tokens: Sequence[int]) -> str:
    encoded = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _probability_at(raw: Mapping[str, Any], index: int | None) -> Any:
    if index is None:
        return None
    rows = raw.get("completion_probabilities")
    if not isinstance(rows, list) or index >= len(rows):
        return None
    return rows[index]


def _minimum_top2_margin(rows: Any) -> float | None:
    if not isinstance(rows, list):
        return None
    margins: list[float] = []
    for row in rows:
        top = row.get("top_logprobs") if isinstance(row, Mapping) else None
        if isinstance(top, list) and len(top) >= 2:
            margins.append(float(top[0]["logprob"]) - float(top[1]["logprob"]))
    return min(margins) if margins else None


def _run_phase(
    *,
    base_url: str,
    slot: int,
    prompts: Sequence[str],
    cache_prompt: bool,
    max_tokens: int,
    seed: int,
    n_probs: int,
    first_turn: int = 1,
    expected_by_turn: Mapping[int, Sequence[int]] | None = None,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    _erase(base_url, slot)
    rows: list[dict[str, Any]] = []
    for turn, prompt in enumerate(prompts, start=first_turn):
        if not cache_prompt:
            _erase(base_url, slot)
        raw = _post(base_url, "/completion", {
            "prompt": prompt,
            "id_slot": slot,
            "n_predict": max_tokens,
            "cache_prompt": cache_prompt,
            "temperature": 0,
            "seed": seed,
            "return_tokens": True,
            "n_probs": n_probs,
        })
        timings = raw.get("timings") if isinstance(raw.get("timings"), Mapping) else {}
        tokens = [int(token) for token in raw.get("tokens", ())]
        rows.append({
            "turn": turn,
            "tokens": tokens,
            "token_sha256": _digest(tokens),
            "content": str(raw.get("content", "")),
            "cache_n": timings.get("cache_n"),
            "prompt_n": timings.get("prompt_n"),
            "predicted_n": timings.get("predicted_n"),
            "completion_probabilities": raw.get("completion_probabilities"),
        })
        if (
            fail_fast
            and expected_by_turn is not None
            and tokens != list(expected_by_turn.get(turn, ()))
        ):
            break
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    messages = trajectory["messages"]
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][:args.turns]
    prompts: list[str] = []
    policy_tokens: list[int] = []
    for index in assistant_indexes:
        prefix = messages[:index]
        presented, overhead = apply_consumption_policy(
            {"messages": prefix}, args.consumption_policy,
            logical_messages=prefix,
        )
        prompts.append(_post(args.base_url, "/apply-template", {
            "messages": presented["messages"],
            "add_generation_prompt": True,
        })["prompt"])
        policy_tokens.append(overhead)

    cached = _run_phase(
        base_url=args.base_url,
        slot=args.slot,
        prompts=prompts,
        cache_prompt=True,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n_probs=args.n_probs,
    )
    cold_prompts = prompts
    cold_first_turn = 1
    if args.compare_only_turn is not None:
        if not 1 <= args.compare_only_turn <= len(prompts):
            raise ValueError("compare-only-turn exceeds the requested trajectory prefix")
        cold_prompts = [prompts[args.compare_only_turn - 1]]
        cold_first_turn = args.compare_only_turn
    cold = _run_phase(
        base_url=args.base_url,
        slot=args.slot,
        prompts=cold_prompts,
        cache_prompt=False,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n_probs=args.n_probs,
        first_turn=cold_first_turn,
        expected_by_turn={int(row["turn"]): row["tokens"] for row in cached},
        fail_fast=args.fail_fast,
    )

    comparisons: list[dict[str, Any]] = []
    cold_by_turn = {int(row["turn"]): row for row in cold}
    for cached_row in cached:
        turn = int(cached_row["turn"])
        cold_row = cold_by_turn.get(turn)
        if cold_row is None:
            continue
        index = turn - 1
        cached_tokens = cached_row["tokens"]
        cold_tokens = cold_row["tokens"]
        first_mismatch = next((
            token_index
            for token_index, pair in enumerate(zip(cached_tokens, cold_tokens))
            if pair[0] != pair[1]
        ), None)
        if first_mismatch is None and len(cached_tokens) != len(cold_tokens):
            first_mismatch = min(len(cached_tokens), len(cold_tokens))
        comparisons.append({
            "turn": turn,
            "trajectory_message_index": assistant_indexes[index],
            "prompt_characters": len(prompts[index]),
            "cached_tokens": len(cached_tokens),
            "cold_tokens": len(cold_tokens),
            "cached_sha256": cached_row["token_sha256"],
            "cold_sha256": cold_row["token_sha256"],
            "exact": cached_tokens == cold_tokens,
            "first_mismatch": first_mismatch,
            "cached_cache_n": cached_row["cache_n"],
            "cached_prompt_n": cached_row["prompt_n"],
            "cold_cache_n": cold_row["cache_n"],
            "cold_prompt_n": cold_row["prompt_n"],
            "cached_probability_at_mismatch": _probability_at(cached_row, first_mismatch),
            "cold_probability_at_mismatch": _probability_at(cold_row, first_mismatch),
            "cached_minimum_top2_logprob_margin": _minimum_top2_margin(
                cached_row.get("completion_probabilities")
            ),
            "cold_minimum_top2_logprob_margin": _minimum_top2_margin(
                cold_row.get("completion_probabilities")
            ),
            "cached_content": cached_row["content"],
            "cold_content": cold_row["content"],
        })

    positive_cache_hits = sum(
        int((row.get("cache_n") or 0) > 0) for row in cached[1:]
    )
    result = {
        "schema_version": 1,
        "probe": "llamacpp_frozen_agent_prefix_cache_equivalence",
        "trajectory": str(args.trajectory),
        "base_url": args.base_url,
        "slot": args.slot,
        "seed": args.seed,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "consumption_policy": args.consumption_policy,
        "consumption_policy_tokens_estimate_by_turn": policy_tokens,
        "requested_turns": args.turns,
        "completed_turns": len(comparisons),
        "positive_cached_requests_after_first": positive_cache_hits,
        "cache_physically_exercised": positive_cache_hits == max(0, len(cached) - 1),
        "exact_turns": sum(int(row["exact"]) for row in comparisons),
        "all_exact": bool(comparisons) and all(row["exact"] for row in comparisons),
        "first_divergent_turn": next(
            (row["turn"] for row in comparisons if not row["exact"]), None
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--slot", type=int, default=1)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-probs", type=int, default=5)
    parser.add_argument(
        "--consumption-policy", choices=CONSUMPTION_POLICIES, default="standard",
    )
    parser.add_argument(
        "--compare-only-turn", type=int,
        help="Run the cached prefix through this turn but cold-prefill only this turn.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop the cold phase at its first mismatch with the completed cached phase.",
    )
    args = parser.parse_args()
    result = run(args)
    compact = {key: result[key] for key in (
        "completed_turns", "positive_cached_requests_after_first",
        "cache_physically_exercised", "exact_turns", "all_exact",
        "first_divergent_turn",
    )}
    print(json.dumps(compact, indent=2))
    raise SystemExit(0 if result["cache_physically_exercised"] and result["all_exact"] else 1)


if __name__ == "__main__":
    main()
