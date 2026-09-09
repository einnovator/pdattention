"""Contracts for Paper 9's frozen selected-record consumption factorial."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.paper9_subagents.run_consumption_factorial import (
    CONDITIONS,
    _identity_digest,
    _model_digests,
    run_case,
    summarize,
)


MANIFEST = Path(
    "experiments/paper9_subagents/benchmarks/consumption_factorial_v1.json"
)


class AnswerClient:
    def __init__(self, path: str) -> None:
        self.path = path

    def chat(self, messages, tools):
        return (
            {"role": "assistant", "content": f"The answer is in `{self.path}`."},
            {"prompt_tokens": 10, "generated_tokens": 5, "model_seconds": 0.01},
        )


class VerificationClient:
    def __init__(self, path: str) -> None:
        self.path = path

    def chat(self, messages, tools):
        if messages[-1]["role"] != "tool":
            return (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_text",
                                "arguments": {"path": self.path},
                            }
                        }
                    ],
                },
                {"prompt_tokens": 10, "generated_tokens": 2, "model_seconds": 0.01},
            )
        return (
            {"role": "assistant", "content": f"Verified in `{self.path}`."},
            {"prompt_tokens": 20, "generated_tokens": 5, "model_seconds": 0.01},
        )


def test_manifest_freezes_one_model_independent_selection_per_case() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["selection_is_model_independent"] is True
    assert len(manifest["cases"]) == 16
    assert sum(bool(case["selection_correct"]) for case in manifest["cases"]) == 14
    for case in manifest["cases"]:
        record = manifest["repositories"][case["repository_id"]]["files"][
            case["selected_path"]
        ]
        assert record["sha256"] == case["selected_sha256"]


def test_factorial_separates_direct_consumption_from_required_verification() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case = next(case for case in manifest["cases"] if not case["selection_correct"])
    direct = run_case(
        AnswerClient(case["selected_path"]),
        manifest,
        case,
        CONDITIONS[0],
        model="fake",
        max_steps=3,
    )
    verified = run_case(
        VerificationClient(case["relevant_path"]),
        manifest,
        case,
        CONDITIONS[2],
        model="fake",
        max_steps=3,
    )

    assert direct["selected_sha256"] == verified["selected_sha256"]
    assert direct["selected_path_hit"] is True
    assert direct["relevant_path_hit"] is False
    assert direct["tool_calls"] == 0
    assert verified["relevant_path_hit"] is True
    assert verified["recovered_from_wrong_selection"] is True
    assert verified["verification_compliant"] is True
    assert verified["tool_calls"] == 1

    report = summarize([direct, verified])
    assert {row["condition"] for row in report} == {
        "direct_injection",
        "mandatory_tool_verification",
    }


def test_execution_identity_tracks_exact_model_digests() -> None:
    runtime = {
        "models": {
            "model-a": {"digest": "sha256-a"},
            "model-b": {"digest": "sha256-b"},
        }
    }
    assert _model_digests(runtime) == {
        "model-a": "sha256-a",
        "model-b": "sha256-b",
    }
    identity = {"models": ["model-a", "model-b"], "digests": _model_digests(runtime)}
    assert _identity_digest(identity) == _identity_digest(dict(reversed(tuple(identity.items()))))
