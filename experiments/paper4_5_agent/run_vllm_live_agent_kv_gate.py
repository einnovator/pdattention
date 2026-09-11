"""Qualify vLLM-Metal by borrowing agent-prefix pages without K/V copying."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from experiments.paper4_5_agent.run_hf_agent_cache_equivalence import (
    _assistant_prompts,
)
from experiments.paper6_vllm.run_matched_e0_e2 import _run
from experiments.paper6_vllm.run_v1_capture_replay_audit import _token_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm
    from pra_vllm.v1_native import VLLMMetalV1NativeBridge
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=16)
    tokenizer = llm.get_tokenizer()
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turns)
    generation = SamplingParams(temperature=0, max_tokens=args.continuation_tokens)
    prime = SamplingParams(temperature=0, max_tokens=1)
    rows = []
    try:
        for turn, prompt in enumerate(prompts, start=1):
            page_tokens = (len(prompt) // bridge.block_size) * bridge.block_size
            if page_tokens == 0 or page_tokens == len(prompt):
                raise RuntimeError("Agent prompt did not provide a page prefix plus suffix.")
            source = prompt[:page_tokens]
            suffix = prompt[page_tokens:]
            salt = f"paper45-agent-live-turn-{turn}"

            observation_start = len(bridge.prefill_page_observations())
            _prime_id, _prime_output, _ = _run(
                llm, bridge, prime, source, cache_salt=salt
            )
            observations = bridge.prefill_page_observations()[observation_start:]
            fresh = [row for row in observations if row["scheduler_cache_start"] == 0]
            if not fresh:
                raise RuntimeError("vLLM did not expose the live source prefill pages.")
            page_count = math.ceil(len(source) / bridge.block_size)
            source_blocks = tuple(fresh[0]["block_ids_by_group"][0][:page_count])

            _ordinary_id, ordinary, _ = _run(
                llm, bridge, generation, prompt, cache_salt=salt
            )
            key = f"agent-live-turn-{turn}"
            bridge.borrow_resident_pages(
                key, source_blocks, selected_token_count=len(source)
            )
            try:
                _native_id, native, _ = _run(
                    llm,
                    bridge,
                    generation,
                    suffix,
                    key=key,
                    source_tokens=len(source),
                    source_position_base=len(source),
                )
            finally:
                bridge.release(key)

            ordinary_tokens = list(map(int, ordinary.outputs[0].token_ids))
            native_tokens = list(map(int, native.outputs[0].token_ids))
            comparison = _token_comparison(ordinary_tokens, native_tokens)
            rows.append(
                {
                    "turn": turn,
                    "source_tokens": len(source),
                    "wire_suffix_tokens": len(suffix),
                    "source_block_ids": list(source_blocks),
                    "ordinary_prefix_cached_tokens": int(ordinary.num_cached_tokens),
                    "selected_kv_tokens": len(source),
                    "selected_text_reencoded_tokens": 0,
                    "physical_kv_copy": False,
                    "comparison": comparison,
                    "ordinary_token_ids": ordinary_tokens,
                    "pra_token_ids": native_tokens,
                }
            )
    finally:
        bridge.close()

    payload = {
        "schema_version": "paper4.5.agent-history-kv-gate.v1",
        "probe": "vllm_same_resident_page_agent_kv_100",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model": args.model,
        "trajectory": str(args.trajectory),
        "retention_fraction": 1.0,
        "adaptor": "none",
        "same_resident_kv_fork": True,
        "page_aligned_prefix_with_wire_tail": True,
        "completed_turns": len(rows),
        "exact_turns": sum(int(row["comparison"]["exact"]) for row in rows),
        "all_exact": all(row["comparison"]["exact"] for row in rows),
        "zero_selected_text_reencoding": True,
        "zero_physical_kv_copy": True,
        "first_divergent_turn": next(
            (row["turn"] for row in rows if not row["comparison"]["exact"]), None
        ),
        "known_constraint": "borrowed history is complete-page aligned; the final partial page remains in the ordinary request suffix",
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

