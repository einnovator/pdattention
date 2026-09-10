"""Replay an agent trajectory through selection and the model chat template."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .context_treatment import ContextTreatment, transform_chat_payload
from .serve_llamacpp_pra import CausalChatNativePromptMixin


class TemplateValidator(CausalChatNativePromptMixin):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request_json(
        self, path: str, payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))


def validate(
    trajectory: Path, *, base_url: str, budget_fraction: float,
    segment_tokens: int,
) -> dict[str, Any]:
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    messages = list(payload["messages"])
    validator = TemplateValidator(base_url)
    rows: list[dict[str, Any]] = []
    for end in range(2, len(messages) + 1):
        if messages[end - 1].get("role") not in {"user", "tool"}:
            continue
        transformed, trace = transform_chat_payload(
            {"model": payload.get("model", "model"), "messages": messages[:end]},
            mode=ContextTreatment.DIRECT_NATIVE_PRA,
            budget_fraction=budget_fraction,
            segment_tokens=segment_tokens,
        )
        resources = tuple(
            SimpleNamespace(
                resource_id=row["resource_id"],
                text=row["text"],
                metadata=row["metadata"],
            )
            for row in transformed["pra"]["resources"]
        )
        request = SimpleNamespace(
            request_id=f"trajectory-prefix-{end}",
            resources=resources,
            messages=tuple(transformed["messages"]),
            tools=(),
        )
        pair = validator._causal_prompt_pair(request)
        if pair is None:
            full_messages = [dict(message) for message in request.messages]
            validator._validate_causal_messages(full_messages)
            prefix = ""
            suffix = validator._render_chat(full_messages, request, generate=True)
        else:
            prefix, suffix = pair
        rows.append({
            "message_count": end,
            "selected_segments": trace.selected_segments,
            "candidate_segments": trace.candidate_segments,
            "saving_fraction": trace.token_saving_fraction_estimate,
            "rendered_prefix_characters": len(prefix),
            "rendered_suffix_characters": len(suffix),
        })
    return {
        "schema_version": 2,
        "selection_contract": "v4-progress-spine",
        "active_action_observation_tail_mandatory": True,
        "recent_completed_turns_pinned": 2,
        "latest_mutation_and_verification_turns_pinned": True,
        "trajectory": str(trajectory),
        "budget_fraction": budget_fraction,
        "segment_tokens": segment_tokens,
        "validated_request_prefixes": len(rows),
        "all_model_templates_valid": True,
        "requests": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--budget-fraction", type=float, required=True)
    parser.add_argument("--segment-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(
        args.trajectory,
        base_url=args.base_url,
        budget_fraction=args.budget_fraction,
        segment_tokens=args.segment_tokens,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
