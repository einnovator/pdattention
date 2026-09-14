"""Test whether llama.cpp sequence attachment preserves live generated state.

This is deliberately different from dense-vs-split prefill parity.  Each case
first creates a live sequence by generating tokens.  The same suffix is then
decoded from (1) a destination sequence attached with ``llama_memory_seq_cp``
and (2) the untouched source sequence.  A 100% pass-through agent profile is
qualified only when those continuations are token-exact.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def _request(base_url: str, path: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _erase(base_url: str, slot: int) -> None:
    _request(base_url, f"/slots/{slot}?action=erase", {})


def _completion(prompt: object, slot: int, tokens: int) -> dict[str, object]:
    return {
        "prompt": prompt,
        "id_slot": slot,
        "n_predict": tokens,
        "cache_prompt": True,
        "temperature": 0,
        "return_tokens": True,
    }


def _first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def run(args: argparse.Namespace) -> dict[str, object]:
    prompt_line = (
        "Repository state: preserve causal order, exact command output, immutable "
        "attention positions, and the complete generated agent trajectory.\n"
    )
    rows = []
    for seed in args.seeds:
        _erase(args.base_url, args.source_slot)
        _erase(args.base_url, args.destination_slot)
        initial_prompt = (f"Seed {seed}.\n" + prompt_line * args.prompt_repetitions)
        history = _request(
            args.base_url,
            "/completion",
            _completion(initial_prompt, args.source_slot, args.history_tokens),
        )
        suffix = (
            "\n<returncode>0</returncode>\n"
            "<output>django/db/models/fields/__init__.py</output>\nContinue:"
        )
        logical_prompt = initial_prompt + str(history.get("content", "")) + suffix
        logical_tokens = _request(
            args.base_url, "/tokenize", {"content": logical_prompt, "add_special": True}
        ).get("tokens", [])
        resident_tokens = (
            int(history.get("tokens_evaluated", 0))
            + max(int(history.get("tokens_predicted", 0)) - 1, 0)
        )
        wire_tokens = logical_tokens[resident_tokens:]
        attached_payload = _completion(
            wire_tokens, args.destination_slot, args.continuation_tokens
        )
        attached_payload["pra_resource_slot"] = args.source_slot
        started = time.perf_counter()
        attached = _request(args.base_url, "/completion", attached_payload)
        attached_ms = (time.perf_counter() - started) * 1000.0

        # A slot becomes resource-pinned when it is used as an attachment
        # source, and llama-server correctly rejects ordinary generation on a
        # pinned slot.  Recreate the deterministic live history on a clean
        # source slot for the control continuation.
        _erase(args.base_url, args.source_slot)
        _erase(args.base_url, args.destination_slot)
        live_history = _request(
            args.base_url,
            "/completion",
            _completion(initial_prompt, args.source_slot, args.history_tokens),
        )
        # llama-server's cache API accepts the complete logical prompt and
        # discovers the resident prefix; an append-only suffix is interpreted
        # as a replacement prompt.  Reconstruct the full prompt exactly as a
        # real chat client does and verify cache_n below.
        live_prompt = initial_prompt + str(live_history.get("content", "")) + suffix
        started = time.perf_counter()
        live = _request(
            args.base_url,
            "/completion",
            _completion(live_prompt, args.source_slot, args.continuation_tokens),
        )
        live_ms = (time.perf_counter() - started) * 1000.0
        live_tokens = [int(token) for token in live.get("tokens", [])]
        attached_tokens = [int(token) for token in attached.get("tokens", [])]
        history_tokens = [int(token) for token in history.get("tokens", [])]
        live_history_tokens = [int(token) for token in live_history.get("tokens", [])]
        rows.append({
            "seed": seed,
            "initial_prompt_tokens": history.get("tokens_evaluated"),
            "generated_history_tokens": history.get("tokens_predicted"),
            "logical_prompt_tokens": len(logical_tokens),
            "bridge_and_suffix_tokens": len(wire_tokens),
            "native_tokens": (attached.get("pra") or {}).get("native_tokens"),
            "wire_tokens": (attached.get("pra") or {}).get("wire_tokens"),
            "physical_kv_copy": (attached.get("pra") or {}).get("physical_kv_copy"),
            "history_replay_exact": history_tokens == live_history_tokens,
            "live_cached_tokens": (live.get("timings") or {}).get("cache_n"),
            "live_vs_attached_exact": live_tokens == attached_tokens,
            "first_divergent_token": _first_difference(live_tokens, attached_tokens),
            "live_token_ids": live_tokens,
            "attached_token_ids": attached_tokens,
            "live_text": live.get("content", ""),
            "attached_text": attached.get("content", ""),
            "live_ms": live_ms,
            "attached_ms": attached_ms,
        })

    exact = sum(
        bool(
            row["history_replay_exact"]
            and row["live_cached_tokens"] == row["native_tokens"]
            and row["live_vs_attached_exact"]
        )
        for row in rows
    )
    artifact = {
        "schema_version": "paper6.7-llamacpp-sequential-state-equivalence-v1",
        "experiment": "live_generated_state_vs_native_sequence_attach",
        "engine": "llama.cpp",
        "base_url": args.base_url,
        "seeds": list(args.seeds),
        "history_tokens": args.history_tokens,
        "continuation_tokens": args.continuation_tokens,
        "exact_pairs": exact,
        "total_pairs": len(rows),
        "pass_through_qualified": exact == len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-slot", type=int, default=0)
    parser.add_argument("--destination-slot", type=int, default=1)
    parser.add_argument("--prompt-repetitions", type=int, default=16)
    parser.add_argument("--history-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 23, 37, 71, 101))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
