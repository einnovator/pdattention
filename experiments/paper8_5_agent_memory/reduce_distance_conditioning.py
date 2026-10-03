"""Reduce distance-conditioned next-action diagnostics without overclaiming."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from .miniswe_semantics import bash_action_contract


def _action_contract(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("action_contract")
    if isinstance(value, Mapping):
        return value
    command = row.get("command")
    return bash_action_contract(str(command)) if command else None


def _logprob_tokens(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = row.get("choice_logprobs")
    content = value.get("content") if isinstance(value, Mapping) else None
    return [item for item in content if isinstance(item, Mapping)] if isinstance(content, list) else []


def _candidate_logprob(item: Mapping[str, Any], token: str) -> float | None:
    for candidate in item.get("top_logprobs") or ():
        if isinstance(candidate, Mapping) and str(candidate.get("token")) == token:
            value = candidate.get("logprob")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def _top_margin(item: Mapping[str, Any]) -> float | None:
    values = [
        float(row["logprob"])
        for row in item.get("top_logprobs") or ()
        if isinstance(row, Mapping) and isinstance(row.get("logprob"), (int, float))
    ]
    values.sort(reverse=True)
    return values[0] - values[1] if len(values) >= 2 else None


def _pairwise_full_token_divergences(
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        left_tokens = _logprob_tokens(left)
        if not left_tokens:
            continue
        for right in rows[left_index + 1:]:
            right_tokens = _logprob_tokens(right)
            if not right_tokens:
                continue
            common = 0
            for left_token, right_token in zip(left_tokens, right_tokens):
                if str(left_token.get("token")) != str(right_token.get("token")):
                    break
                common += 1
            divergence = None
            if common < min(len(left_tokens), len(right_tokens)):
                left_item = left_tokens[common]
                right_item = right_tokens[common]
                left_choice = str(left_item.get("token"))
                right_choice = str(right_item.get("token"))
                divergence = {
                    "token_index": common,
                    "left_token": left_choice,
                    "right_token": right_choice,
                    "left_chosen_logprob": left_item.get("logprob"),
                    "right_chosen_logprob": right_item.get("logprob"),
                    "left_logprob_for_right_token": _candidate_logprob(
                        left_item, right_choice
                    ),
                    "right_logprob_for_left_token": _candidate_logprob(
                        right_item, left_choice
                    ),
                    "left_top1_margin": _top_margin(left_item),
                    "right_top1_margin": _top_margin(right_item),
                }
            result.append({
                "left_repeat": left.get("repeat"),
                "right_repeat": right.get("repeat"),
                "same_request_sha256": (
                    left.get("request_sha256") == right.get("request_sha256")
                ),
                "common_prefix_tokens": common,
                "left_token_count": len(left_tokens),
                "right_token_count": len(right_tokens),
                "first_divergence": divergence,
            })
    return result


def reduce_distance_conditioning(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    arms = payload.get("arms")
    if not isinstance(rows, list) or not isinstance(arms, Mapping):
        raise ValueError("distance diagnostic must contain rows and arms")
    full_rows = [row for row in rows if row.get("arm") == "FULL"]
    full_selected_digests = {
        str(row.get("selected_messages_sha256")) for row in full_rows
    }
    full_command_hashes = {
        str(row.get("command_sha256"))
        for row in full_rows if row.get("command_sha256")
    }
    full_response_hashes = {
        str(row.get("response_content_sha256"))
        for row in full_rows if row.get("response_content_sha256")
    }
    full_transport_clean = bool(full_rows) and all(
        not row.get("transport_error") for row in full_rows
    )
    full_action_valid = bool(full_rows) and all(
        row.get("action_valid") is True for row in full_rows
    )
    full_request_exact = len(full_selected_digests) == 1
    full_command_stable = bool(full_command_hashes) and len(full_command_hashes) == 1
    causal_admission = bool(
        full_transport_clean
        and full_action_valid
        and full_request_exact
        and full_command_stable
    )
    by_repeat = {
        int(row["repeat"]): row
        for row in full_rows if isinstance(row.get("repeat"), int)
    }
    summaries: dict[str, Any] = {}
    for arm, arm_metadata in arms.items():
        arm_rows = [row for row in rows if row.get("arm") == arm]
        comparable = [
            row for row in arm_rows
            if int(row.get("repeat", -1)) in by_repeat
            and row.get("command_sha256")
            and by_repeat[int(row["repeat"])].get("command_sha256")
        ]
        matched = sum(
            row["command_sha256"]
            == by_repeat[int(row["repeat"])]["command_sha256"]
            for row in comparable
        )
        operation_comparable = [
            row for row in arm_rows
            if int(row.get("repeat", -1)) in by_repeat
            and row.get("operation_class")
            and by_repeat[int(row["repeat"])].get("operation_class")
        ]
        operation_matched = sum(
            row["operation_class"]
            == by_repeat[int(row["repeat"])]["operation_class"]
            for row in operation_comparable
        )
        contract_comparable = [
            row for row in arm_rows
            if int(row.get("repeat", -1)) in by_repeat
            and _action_contract(row) is not None
            and _action_contract(by_repeat[int(row["repeat"])]) is not None
        ]
        contract_matched = sum(
            _action_contract(row)
            == _action_contract(by_repeat[int(row["repeat"])])
            for row in contract_comparable
        )
        valid = sum(row.get("action_valid") is True for row in arm_rows)
        summaries[str(arm)] = {
            "attempts": len(arm_rows),
            "transport_errors": sum(bool(row.get("transport_error")) for row in arm_rows),
            "valid_actions": valid,
            "valid_action_fraction": valid / len(arm_rows) if arm_rows else None,
            "unique_command_hashes": len({
                row.get("command_sha256") for row in arm_rows
                if row.get("command_sha256")
            }),
            "unique_response_hashes": len({
                row.get("response_content_sha256") for row in arm_rows
                if row.get("response_content_sha256")
            }),
            "operation_classes": dict(Counter(
                str(row.get("operation_class")) for row in arm_rows
                if row.get("operation_class")
            )),
            "paired_command_matches": matched,
            "paired_comparable": len(comparable),
            "paired_command_match_fraction": (
                matched / len(comparable) if comparable else None
            ),
            "paired_operation_matches": operation_matched,
            "paired_operation_comparable": len(operation_comparable),
            "paired_operation_match_fraction": (
                operation_matched / len(operation_comparable)
                if operation_comparable else None
            ),
            "paired_action_contract_matches": contract_matched,
            "paired_action_contract_comparable": len(contract_comparable),
            "paired_action_contract_match_fraction": (
                contract_matched / len(contract_comparable)
                if contract_comparable else None
            ),
            "saving_fraction": (
                arm_metadata.get("saving_fraction")
                if isinstance(arm_metadata, Mapping) else None
            ),
            "exact_noop": (
                arm_metadata.get("exact_noop")
                if isinstance(arm_metadata, Mapping) else None
            ),
            "causal_interpretation": (
                "admitted_against_stable_full"
                if causal_admission else "withheld_full_unstable_or_incomplete"
            ),
        }
    return {
        "schema_version": 1,
        "study": "paper8_5_distance_conditioning_reduction",
        "source_study": payload.get("study"),
        "full_control": {
            "attempts": len(full_rows),
            "transport_clean": full_transport_clean,
            "action_valid": full_action_valid,
            "selected_request_exact": full_request_exact,
            "unique_command_hashes": len(full_command_hashes),
            "unique_response_hashes": len(full_response_hashes),
            "causal_admission": causal_admission,
            "pairwise_token_divergences": _pairwise_full_token_divergences(
                full_rows
            ),
            "note": (
                "Stable byte-identical FULL commands admit arm attribution. "
                "Temperature zero alone does not."
            ),
        },
        "arms": summaries,
        "guardrail": (
            "This reduction concerns the next action only. Calls-to-solution "
            "and official resolution require autonomous paired executions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = reduce_distance_conditioning(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
