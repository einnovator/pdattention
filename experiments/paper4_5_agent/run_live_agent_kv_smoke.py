"""Real llama.cpp smoke for record-aware agent-history K/V selection."""

from __future__ import annotations

import argparse
import json
from typing import Any

from pra_hf.deployment import PRAWireRequest
from pra_llamacpp import LlamaCppEngineAdapter, LlamaCppNativeServerExecutor

from .context_treatment import ContextTreatment, transform_chat_payload
from .serve_llamacpp_pra import (
    CausalChatNativePromptMixin,
    HybridLlamaCppAdapter,
    PlainSlotExecutor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://192.168.1.6:18083")
    parser.add_argument("--model", default="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    args = parser.parse_args()

    executor_type = type(
        "AgentNativeExecutor",
        (CausalChatNativePromptMixin, LlamaCppNativeServerExecutor),
        {},
    )
    executor = executor_type(
        args.base_url,
        resource_slot=0,
        request_slot=1,
        model_fingerprint=args.model,
    )
    executor._delete_resource(0)
    executor._delete_resource(1)
    native = LlamaCppEngineAdapter(
        args.base_url,
        model_fingerprint=args.model,
        native_executor=executor,
    )
    adapter = HybridLlamaCppAdapter(
        native,
        PlainSlotExecutor(executor, prefix_caching=True),
        prefix_caching=True,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Reply with one short word."},
        {"role": "user", "content": "The task key is ALPHA."},
    ]
    turns: list[dict[str, Any]] = []

    def run(
        *, frozen: list[tuple[str, str]] | None = None, phase: str = "sparse"
    ) -> str:
        payload, trace = transform_chat_payload(
            {
                "model": args.model,
                "messages": messages,
                "max_tokens": 4,
                "temperature": 0,
                "seed": 123,
            },
            mode=ContextTreatment.DIRECT_NATIVE_PRA,
            budget_fraction=0.25,
            frozen_selection=frozen,
            recent_completed_turns=0,
            recent_mutation_turns=0,
            recent_verification_turns=0,
        )
        result = adapter.generate(PRAWireRequest.from_openai(payload))
        pra = dict(result.raw.get("pra") or {})
        turns.append({
            "turn": len(turns) + 1,
            "phase": phase,
            "selected_resource_ids": [
                row["resource_id"] for row in payload.get("pra", {}).get("resources", [])
            ],
            "logical_tokens_estimate": trace.logical_input_tokens_estimate,
            "selected_tokens_estimate": trace.selected_tokens_estimate,
            "model_text": result.text,
            "kv_source": pra.get("kv_source"),
            "source_tokens": pra.get("source_tokens"),
            "selected_kv_tokens": pra.get("selected_kv_tokens"),
            "selected_text_reencoded_tokens": pra.get(
                "selected_text_reencoded_tokens"
            ),
            "commit_succeeded": pra.get("commit_succeeded"),
        })
        return result.text

    answer1 = run()
    messages.extend([
        {"role": "assistant", "content": answer1},
        {
            "role": "user",
            "content": "Observation: " + " ".join(
                f"irrelevant-{index}" for index in range(120)
            ),
        },
    ])
    answer2 = run()
    messages.extend([
        {"role": "assistant", "content": answer2},
        {"role": "user", "content": "Return the task key."},
    ])
    run(frozen=[("m1-0-user", "The task key is ALPHA.")])

    # A parser-rejected assistant response is not present in the next client
    # transcript. Prove that the engine removes only that discarded source
    # tail and evaluates the replacement recovery message without rebuilding
    # the stable system/task prefix.
    executor._delete_resource(0)
    executor._delete_resource(1)
    adapter = HybridLlamaCppAdapter(
        native,
        PlainSlotExecutor(executor, prefix_caching=True),
        prefix_caching=True,
    )
    messages = [
        {"role": "system", "content": "Reply with one short word."},
        {"role": "user", "content": "The task key is ALPHA."},
    ]
    rejected = run(phase="rollback")
    source_before_rollback = len(next(iter(adapter._live_session_tokens.values())))
    messages.append({
        "role": "user",
        "content": "FORMAT ERROR: ignore the rejected response and return ALPHA.",
    })
    run(phase="rollback")
    rollback_row = turns[-1]
    rollback = {
        "discarded_model_text": rejected,
        "source_tokens_before": source_before_rollback,
        "source_prefix_tokens_after": rollback_row["source_tokens"],
        "source_was_rolled_back": (
            rollback_row["source_tokens"] < source_before_rollback
        ),
        "selected_text_reencoded_tokens": rollback_row[
            "selected_text_reencoded_tokens"
        ],
        "commit_succeeded": rollback_row["commit_succeeded"],
        "recovery_text": rollback_row["model_text"],
    }

    summary = {
        "schema_version": "1.0",
        "purpose": "correctness smoke, not a task-quality or speed result",
        "engine": "llama.cpp",
        "model": args.model,
        "turns": turns,
        "rejected_action_rollback": rollback,
        "zero_selected_text_reencoding": all(
            row["selected_text_reencoded_tokens"] == 0
            for row in turns if row["kv_source"] is not None
        ),
        "sparse_turn_observed": any(
            row["selected_kv_tokens"] is not None
            and row["source_tokens"] is not None
            and row["selected_kv_tokens"] < row["source_tokens"]
            for row in turns[1:]
        ),
    }
    print(json.dumps(summary, indent=2))
    if not summary["zero_selected_text_reencoding"]:
        raise SystemExit("selected history was re-encoded")
    if not summary["sparse_turn_observed"]:
        raise SystemExit("smoke did not exercise a sparse live-prefix selection")
    if not rollback["source_was_rolled_back"] or not rollback["commit_succeeded"]:
        raise SystemExit("rejected-action source rollback failed")


if __name__ == "__main__":
    main()
