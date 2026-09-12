"""Check append-only agent turns with the pinned model's real chat template."""

from __future__ import annotations

import argparse
import json

from pra_sglang.agent_executor import (
    _common_prefix,
    _render,
    configure_append_stable_template,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--chat-template-profile",
        choices=("native", "qwen3-stable-no-thinking", "pure-chatml-stable"),
        default="native",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    template_options = {"enable_thinking": False}
    base = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test."},
    ]
    assistant = {"role": "assistant", "content": "Inspect the repository first."}
    next_messages = [
        *base,
        assistant,
        {
            "role": "user",
            "content": (
                "Chunk ID: 123\nProcess exited with code 0\n"
                "Final output:\nfile.py"
            ),
        },
    ]
    stock_first = _render(
        tokenizer,
        base,
        generation_prompt=True,
        chat_template_kwargs=template_options,
    )
    template_digest = configure_append_stable_template(
        tokenizer, args.chat_template_profile
    )
    first = _render(
        tokenizer,
        base,
        generation_prompt=True,
        chat_template_kwargs=template_options,
    )
    closed = _render(
        tokenizer,
        [*base, assistant],
        generation_prompt=False,
        chat_template_kwargs=template_options,
    )
    next_prompt = _render(
        tokenizer,
        next_messages,
        generation_prompt=True,
        chat_template_kwargs=template_options,
    )
    if closed[: len(first)] != first:
        raise RuntimeError(
            "The real chat template rewrites the generation-prompt prefix when "
            "the assistant record is closed."
        )
    canonical = [*first, *closed[len(first):]]
    common = _common_prefix(canonical, next_prompt)
    reencoded = max(0, min(len(canonical), len(next_prompt)) - common)
    result = {
        "model": args.model,
        "revision": args.revision,
        "chat_template_profile": args.chat_template_profile,
        "chat_template_digest": template_digest,
        "first_turn_equal_to_stock": first == stock_first,
        "first_prompt_tokens": len(first),
        "assistant_and_closing_tokens": len(closed) - len(first),
        "canonical_tokens": len(canonical),
        "next_prompt_tokens": len(next_prompt),
        "common_prefix_tokens": common,
        "new_suffix_tokens": len(next_prompt) - common,
        "selected_history_reencoded_tokens": reencoded,
        "canonical_is_exact_next_prefix": canonical
        == next_prompt[: len(canonical)],
        "eos_token_id": tokenizer.eos_token_id,
        "first_tail_ids": first[-16:],
        "closing_tail_ids": closed[-4:],
        "first_tail_text": tokenizer.decode(first[-16:]),
        "closed_tail_text": tokenizer.decode(closed[-16:]),
        "next_at_divergence_text": tokenizer.decode(
            next_prompt[max(0, common - 8): common + 16]
        ),
    }
    print(json.dumps(result, indent=2))
    preferred_equality_failed = (
        args.chat_template_profile == "qwen3-stable-no-thinking"
        and not result["first_turn_equal_to_stock"]
    )
    if (
        reencoded
        or not result["canonical_is_exact_next_prefix"]
        or preferred_equality_failed
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
