"""Reproduction gates for the long-running Paper 4.5 agent campaign."""

from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pra_hf.deployment import (
    PRAEngineCapabilities,
    PRAEngineResult,
    PRAGatewayMode,
    PRAWireRequest,
    PRAWireResource,
)
from pra_hf.gateway_session import ResourceDelta, ResourceOperation
from pra_hf.gateway import PRAGateway
from experiments.paper4_5_agent.reproduction import OfficialResult, review_result
from experiments.paper4_5_agent.reports import _next_tier, _success_band
from experiments.paper4_5_agent.benchmark import (
    load_benchmark_card,
    precision_diagnostic_ids,
)
from experiments.paper4_5_agent.build_easy_cohorts import (
    DIFFICULTY,
    build_card,
    ids_digest,
    select_easy_rows,
)
from experiments.paper4_5_agent.build_swebench_lite_cohort import (
    build_card as build_lite_card,
    select_rows as select_lite_rows,
)
from experiments.paper4_5_agent.promote_nested_baseline import promote
from experiments.paper4_5_agent.probe_frozen_prefix_behavior import (
    run as run_frozen_prefix_probe,
)
from experiments.paper4_5_agent.analyze_easy_frontier import (
    _bucket_effects,
    summarize as summarize_frontier,
)
from experiments.paper4_5_agent.analyze_transport_equivalence import (
    compare as compare_transport_histories,
)
from experiments.paper4_5_agent.context_treatment import (
    ContextTreatment,
    TreatmentProxy,
    _load_selection_fixture,
    _selection_digest,
    apply_consumption_policy,
    enforce_consumption_action,
    transform_chat_payload,
)
from experiments.paper4_5_agent.serve_llamacpp_pra import (
    CausalChatNativePromptMixin,
    HybridLlamaCppAdapter,
    PlainSlotExecutor,
    SessionClosedError,
    SessionCommitConflict,
    _direct_handler,
    parse_args as parse_llamacpp_server_args,
)
from experiments.paper4_5_agent.summarize_easy50_strata import stratified_outcomes
from experiments.paper4_5_agent.harness_matrix import (
    HarnessMatrixConfig,
    MatrixModel,
    MatrixRoute,
    _completed_attempt,
    _invalid_trial_reason,
    _load_baseline_admission,
    _paired_route_comparisons,
    _route_preflight,
    harbor_command,
    matrix_cells,
    run_matrix,
)
from experiments.paper4_5_agent.runners.r2egym import (
    locate_official_report,
    normalize_official_report,
    trajectories_to_predictions,
    write_task_results,
)
from experiments.paper4_5_agent.summarize_baseline import summarize
from experiments.paper4_5_agent.run_campaign import (
    _campaign_environment,
    _expand_command,
    record_result,
    run_campaign,
)
from experiments.paper4_5_agent.schema import (
    CampaignConfig,
    PublishedBaseline,
    ReproductionStatus,
)
from experiments.paper4_5_agent.runners.swebench_verified import (
    _aggregate_traces,
    _chunk_receipt_reusable,
    _completion_token_overrides,
    _cleanup_owned_containers,
    _container_environment,
    derive_task_card,
    _execute_chunks,
    _grader_error_type,
    _is_h100_80gb,
    _normalize_report,
    _prepull_swebench_images,
    _raise_on_agent_infrastructure_error,
    _trajectory_metrics,
    _write_empty_predictions,
    gateway_preflight,
    package_versions,
    treatment_placement,
)
from experiments.paper4_5_agent.runners.local_qwen_swebench import (
    official_image_platform,
)
from experiments.agents.schema import BenchmarkManifest


ROOT = Path(__file__).parents[1]


def test_verification_guard_changes_presentation_after_selection_only() -> None:
    resources = [{"resource_id": "m1-0-user", "text": "retained history"}]
    payload = {
        "messages": [
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "current observation"},
        ],
        "pra": {
            "resources": resources,
            "metadata": {"selection_complete": False},
        },
    }
    logical_messages = [
        {"role": "system", "content": "base system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": (
                "THOUGHT: edit\n```mswea_bash_command\n"
                "sed -i 's/old/new/' source.py\n```"
            ),
        },
        {"role": "user", "content": "<returncode>0</returncode>"},
    ]

    transformed, overhead = apply_consumption_policy(
        payload, "verification-guard-v1", logical_messages=logical_messages,
    )

    assert payload["messages"][0]["content"] == "base system"
    assert transformed["pra"]["resources"] == resources
    assert transformed["pra"]["metadata"]["selection_complete"] is False
    assert transformed["pra"]["metadata"]["consumption_policy"] == (
        "verification-guard-v1"
    )
    assert "source mutation is still unverified" in transformed["messages"][-1]["content"]
    assert transformed["messages"][0]["content"] == "base system"
    assert overhead > 0

    unchanged, no_overhead = apply_consumption_policy(
        payload, "verification-guard-v1", logical_messages=logical_messages[:2],
    )
    assert unchanged["messages"] == payload["messages"]
    assert no_overhead == 0


_MUTATION_STEP = ("sed -i 's/old/new/' source.py", "<returncode>0</returncode>")
_DIFF_STEP = (
    "git diff -- source.py",
    "<returncode>0</returncode><output>diff --git a/source.py b/source.py</output>",
)
_TEST_STEP = ("python -m pytest tests/test_source.py -q", "<returncode>0</returncode>")
_PATCH_STEP = ("git diff -- source.py > patch.txt", "<returncode>0</returncode>")


@pytest.mark.parametrize(("steps", "expected"), [
    ([_MUTATION_STEP, ("sed -n '1,20p' source.py", "<returncode>0</returncode>")],
     "still unverified"),
    ([_MUTATION_STEP, _DIFF_STEP], "narrowest relevant"),
    ([
        _MUTATION_STEP,
        ("git diff -- source.py", "<returncode>0</returncode><output>\n</output>"),
    ], "silent no-op"),
    ([_MUTATION_STEP, _DIFF_STEP, _TEST_STEP], "Create patch.txt"),
    ([_MUTATION_STEP, _DIFF_STEP, _TEST_STEP, _PATCH_STEP], "Inspect patch.txt"),
    ([_MUTATION_STEP, _DIFF_STEP, _TEST_STEP, _PATCH_STEP,
      ("cat patch.txt", "<returncode>0</returncode>")], "Submit it now"),
])
def test_verification_guard_advances_only_at_completed_boundaries(
    steps: list[tuple[str, str]], expected: str,
) -> None:
    logical = [{"role": "user", "content": "task"}]
    for command, observation in steps:
        logical.extend(({
            "role": "assistant",
            "content": f"THOUGHT: next\n```mswea_bash_command\n{command}\n```",
        }, {"role": "user", "content": observation}))

    transformed, overhead = apply_consumption_policy(
        {"messages": [{"role": "user", "content": steps[-1][1]}]},
        "verification-guard-v1", logical_messages=logical,
    )

    assert expected in transformed["messages"][0]["content"]
    assert overhead > 0


def _response(command: str) -> dict[str, object]:
    return {
        "id": "test",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"THOUGHT: next\n```mswea_bash_command\n{command}\n```",
            },
        }],
    }


def _history(steps: list[tuple[str, str]]) -> list[dict[str, str]]:
    logical = [{"role": "user", "content": "task"}]
    for command, observation in steps:
        logical.extend((
            {
                "role": "assistant",
                "content": f"THOUGHT: next\n```mswea_bash_command\n{command}\n```",
            },
            {"role": "user", "content": observation},
        ))
    return logical


@pytest.mark.parametrize(("steps", "proposal", "expected", "reason"), [
    ([_MUTATION_STEP], "sed -n '1,20p' source.py", "git diff",
     "diff_required_after_mutation"),
    ([_MUTATION_STEP, ("git diff", "<returncode>0</returncode><output></output>")],
     "sed -n '1,20p' source.py", "PRA_AGENT_BLOCKED",
     "repair_required_after_empty_diff"),
    ([_MUTATION_STEP, _DIFF_STEP], "grep -R symbol .", "PRA_AGENT_BLOCKED",
     "focused_test_required"),
    ([_MUTATION_STEP, _DIFF_STEP, _TEST_STEP], "grep -R symbol .", "git diff > patch.txt",
     "patch_creation_required"),
    ([_MUTATION_STEP, _DIFF_STEP, _TEST_STEP, _PATCH_STEP], "git status", "cat patch.txt",
     "patch_inspection_required"),
    ([_MUTATION_STEP, _DIFF_STEP, _TEST_STEP, _PATCH_STEP,
      ("cat patch.txt", "<returncode>0</returncode>")], "git status",
     "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "submission_required"),
])
def test_enforced_policy_replaces_out_of_order_boundary_action(
    steps: list[tuple[str, str]], proposal: str, expected: str, reason: str,
) -> None:
    original = _response(proposal)
    transformed, metadata = enforce_consumption_action(
        original, "verification-enforced-v1", logical_messages=_history(steps),
    )

    content = transformed["choices"][0]["message"]["content"]
    assert expected in content
    assert metadata["action_enforced"] is True
    assert metadata["enforcement_reason"] == reason
    assert metadata["original_content_sha256"]
    assert proposal in metadata["original_content"]
    assert transformed["pra"]["agent"]["enforced_command"]
    assert "original_content" not in transformed["pra"]["agent"]
    assert "pra" not in original


def test_enforced_policy_preserves_compliant_or_semantic_action() -> None:
    compliant = _response("git diff -- source.py")
    transformed, metadata = enforce_consumption_action(
        compliant, "verification-enforced-v1", logical_messages=_history([_MUTATION_STEP]),
    )
    assert transformed == compliant
    assert metadata["action_enforced"] is False

    reproduction = _response("python -c \"from source import reproduce; reproduce()\"")
    transformed, metadata = enforce_consumption_action(
        reproduction,
        "verification-enforced-v1",
        logical_messages=_history([_MUTATION_STEP, _DIFF_STEP]),
    )
    assert transformed == reproduction
    assert metadata["action_enforced"] is False

    failed_test = _history([
        _MUTATION_STEP,
        _DIFF_STEP,
        ("pytest tests/test_source.py -q", "<returncode>1</returncode>"),
    ])
    repair = _response("sed -i 's/new/fixed/' source.py")
    transformed, metadata = enforce_consumption_action(
        repair, "verification-enforced-v1", logical_messages=failed_test,
    )
    assert transformed == repair
    assert metadata["action_enforced"] is False
CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/fim14b_r2egym.yaml"
MATRIX_CONFIG = ROOT / "experiments/paper4_5_agent/configs/harness_matrices/qwen3_coder_30b_pilot.yaml"
SWEBENCH_CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/swebench_pra_frontier.yaml"
EASY20_CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/swebench_easy20_calibration.yaml"
EASY50_CONFIG = ROOT / "experiments/paper4_5_agent/configs/campaigns/swebench_easy50_pra_frontier.yaml"
FIXED50 = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_verified_fixed50.json"
EASY20 = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_verified_easy20.json"
EASY50 = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_verified_easy50.json"
INCREMENTAL_ENGINE_MATRIX = ROOT / (
    "docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/"
    "swebench_verified_easy50/incremental_engine_matrix.json"
)
LITE50 = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_lite50.json"
EASY20_RESULT = ROOT / (
    "docs/papers/shared/results/paper4_5_runtime_productization/coding_agents/"
    "swebench_verified_easy20/no_pra/official_result.json"
)


def test_local_qwen_runner_requests_official_x86_images_on_arm() -> None:
    assert official_image_platform("arm64") == "linux/amd64"
    assert official_image_platform("aarch64") == "linux/amd64"


def test_incremental_engine_matrix_crosses_every_pra_level_with_adaptor() -> None:
    matrix = json.loads(INCREMENTAL_ENGINE_MATRIX.read_text(encoding="utf-8"))
    arms = {
        (float(row["retention_fraction"]), str(row["adaptor"])): row
        for row in matrix["arms"]
        if row["pra"]
    }
    for retention in (1.0, 0.9, 0.75):
        assert (retention, "none") in arms
        assert (retention, "frozen_bundle") in arms
    assert arms[(1.0, "none")]["label"] == "pra_100_no_adaptor"
    assert arms[(1.0, "frozen_bundle")]["label"] == "pra_100_model_adaptor"
    assert all(
        arms[(0.75, adaptor)]["enabled"] is False
        for adaptor in ("none", "frozen_bundle")
    )
    assert matrix["factor_contract"]["gateway"] is False


def test_task_major_runner_derives_locked_single_task_without_resampling() -> None:
    parent = load_benchmark_card(
        ROOT / "experiments/paper4_5_agent/benchmarks/"
        "swebench_verified_easy50_baseline_success14.json"
    )

    task = derive_task_card(parent, 2, source="success14.json")

    assert task["instance_ids"] == ["django__django-15368"]
    assert task["expected_count"] == 1
    assert task["parent_cohort_ids_sha256"] == parent["canonical_ids_sha256"]
    assert task["canonical_ids_sha256"] == ids_digest(task["instance_ids"])
    assert task["task_index"] == 2


def test_task_major_runner_rejects_out_of_range_index() -> None:
    parent = {"instance_ids": ["repo__task-1"], "canonical_ids_sha256": "digest"}
    with pytest.raises(ValueError, match="1..1"):
        derive_task_card(parent, 2, source="parent.json")
    assert official_image_platform("x86_64") is None


def test_easy50_completion_ceiling_is_forwarded_to_minisweagent() -> None:
    assert _completion_token_overrides(
        SimpleNamespace(max_completion_tokens=4096)
    ) == ["-c", "model.model_kwargs.max_tokens=4096"]
    assert _completion_token_overrides(SimpleNamespace()) == []
    with pytest.raises(ValueError, match="must be positive"):
        _completion_token_overrides(SimpleNamespace(max_completion_tokens=0))


def test_fixed50_card_is_exact_unique_and_digest_protected() -> None:
    card = load_benchmark_card(FIXED50)
    assert len(card["instance_ids"]) == len(set(card["instance_ids"])) == 50
    assert card["source_revision"] == "8f894c2284b9f73a515024d7c1f32e4d0fb14a04"
    assert card["canonical_ids_sha256"] == (
        "20acb5f7e30fb3c854091e47c4214afb7304a5d47f353408a71ffaa418318131"
    )
    assert precision_diagnostic_ids(card) == tuple(card["instance_ids"][:10])


def test_easy_cards_are_nested_frozen_and_digest_protected() -> None:
    easy20 = load_benchmark_card(EASY20)
    easy50 = load_benchmark_card(EASY50)
    assert easy20["instance_ids"] == easy50["instance_ids"][:20]
    assert len(easy20["instance_ids"]) == 20
    assert len(easy50["instance_ids"]) == 50
    assert {row["difficulty"] for row in easy50["task_metadata"]} == {DIFFICULTY}


def test_lite50_card_is_frozen_and_outcome_blind() -> None:
    card = load_benchmark_card(LITE50)
    assert len(card["instance_ids"]) == 50
    assert card["source_revision"] == "b0dde1093fe417d83b7184254edf8199c1f0dff5"
    assert card["canonical_ids_sha256"] == (
        "3dab12f2b8e1fe3cbddeb20cc7522991ad76f25eba9dbd147223297e555ab93d"
    )
    rows = [
        {
            "instance_id": f"repo__project-{index}", "repo": "repo/project",
            "base_commit": str(index), "version": "1", "model_outcome": index % 2,
        }
        for index in range(60)
    ]
    assert select_lite_rows(rows, limit=10) == select_lite_rows(reversed(rows), limit=10)
    assert "model_outcome" not in json.dumps(build_lite_card(rows, count=10))


def test_easy_selection_is_deterministic_and_outcome_blind() -> None:
    rows = [
        {
            "instance_id": f"repo__project-{index}",
            "repo": "repo/project",
            "base_commit": str(index),
            "version": "1",
            "difficulty": DIFFICULTY if index != 2 else "1-4 hours",
            "model_outcome": index % 2,
        }
        for index in range(8)
    ]
    first = select_easy_rows(rows, limit=4)
    second = select_easy_rows(reversed(rows), limit=4)
    assert [row["instance_id"] for row in first] == [row["instance_id"] for row in second]
    card = build_card(rows, count=4, eligible_count=7)
    assert "model_outcome" not in json.dumps(card)


def test_campaign_children_use_scheduler_interpreter() -> None:
    environment = _campaign_environment({"PRA_TEST_VALUE": "present"})
    selected = Path(environment["PATH"].split(os.pathsep)[0]).resolve()
    assert selected == Path(sys.executable).absolute().parent.resolve()
    assert environment["PRA_TEST_VALUE"] == "present"


def test_nested_baseline_promotion_reuses_only_exact_completed_prefix(tmp_path: Path) -> None:
    source_ids = ["repo__project-1", "repo__project-2"]
    destination_ids = [*source_ids, "repo__project-3"]
    source_card = tmp_path / "source.json"
    destination_card = tmp_path / "destination.json"
    for path, ids in ((source_card, source_ids), (destination_card, destination_ids)):
        path.write_text(json.dumps({
            "expected_count": len(ids),
            "canonical_ids_sha256": ids_digest(ids),
            "instance_ids": ids,
        }), encoding="utf-8")
    source_output = tmp_path / "source"
    for index, instance_id in enumerate(source_ids):
        chunk = source_output / f"chunk_{index:02d}"
        chunk.mkdir(parents=True)
        (chunk / "official_chunk_result.json").write_text(json.dumps({
            "submitted_ids": [instance_id], "resolved_ids": [], "error_ids": [],
        }), encoding="utf-8")
    destination_output = tmp_path / "destination"

    manifest = promote(
        source_card, destination_card, source_output, destination_output,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["copied_task_ids"] == source_ids
    assert (destination_output / "chunk_01" / "official_chunk_result.json").is_file()
    assert not (destination_output / "chunk_02").exists()


def test_easy_frontier_summary_preserves_paired_outcomes_and_missing_metrics(tmp_path: Path) -> None:
    baseline = tmp_path / "no_pra"
    treatment = tmp_path / "truncation_50"
    baseline.mkdir()
    treatment.mkdir()
    rows = [
        {"instance_id": "a", "resolved": True, "mode": "no-pra",
         "context_budget_fraction": 1.0, "physical_input_tokens": 100,
         "logical_input_tokens": 100, "cumulative_prompt_tokens": 100},
        {"instance_id": "b", "resolved": False, "mode": "no-pra",
         "context_budget_fraction": 1.0, "physical_input_tokens": 200,
         "logical_input_tokens": 200, "cumulative_prompt_tokens": 200},
    ]
    (baseline / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    treated = [
        {**rows[0], "resolved": False, "mode": "truncation",
         "context_budget_fraction": 0.5, "physical_input_tokens": 50},
        {**rows[1], "resolved": True, "mode": "truncation",
         "context_budget_fraction": 0.5, "physical_input_tokens": 100},
    ]
    (treatment / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in treated) + "\n", encoding="utf-8"
    )

    summary = summarize_frontier(tmp_path)

    conditions = {row["condition"]: row for row in summary["conditions"]}
    assert conditions["no_pra"]["physical_input_tokens"] == 300
    assert conditions["no_pra"]["wall_time_s"] is None
    assert conditions["truncation_50"]["token_saving_fraction"] == 0.5
    assert {row["outcome"] for row in summary["paired"]} == {"regressed", "recovered"}


def test_context_demand_buckets_are_contiguous_tertiles() -> None:
    rows = [
        {
            "baseline_trajectory_length": index,
            "baseline_resolved": index < 3,
            "treatment_resolved": index >= 3,
        }
        for index in range(6)
    ]
    assert _bucket_effects(rows, "baseline_trajectory_length") == [-1.0, 0.0, 1.0]


def test_selected_context_trace_has_stable_resource_digest() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "alpha beta gamma"},
            {"role": "user", "content": "find alpha"},
        ]
    }
    _, first = transform_chat_payload(
        payload, mode=ContextTreatment.PRA_SELECTED_CONTEXT, budget_fraction=0.75,
    )
    _, second = transform_chat_payload(
        payload, mode=ContextTreatment.PRA_SELECTED_CONTEXT, budget_fraction=0.75,
    )
    assert first.selected_resource_digest
    assert first.selected_resource_digest == second.selected_resource_digest


def test_full_budget_selection_preserves_causal_order_and_formatting() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task alpha\n\nkeep spacing"},
            {
                "role": "assistant",
                "content": "THOUGHT: inspect alpha\n\n```mswea_bash_command\nls -la\n```",
            },
            {"role": "user", "content": "latest observation"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload, mode=ContextTreatment.DIRECT_NATIVE_PRA, budget_fraction=1.0,
    )

    resources = transformed["pra"]["resources"]
    assert transformed["pra"]["metadata"]["selection_complete"] is True
    assert transformed["pra"]["metadata"]["history_projection"] == "live-agent-kv-v1"
    assert transformed["pra"]["metadata"]["mandatory_message_indices"] == [0, 2, 3]
    assert transformed["pra"]["metadata"]["logical_message_manifest"] == [
        {
            "message_index": index,
            "role": message["role"],
            "content_sha256": hashlib.sha256(
                message["content"].encode("utf-8")
            ).hexdigest(),
        }
        for index, message in enumerate(payload["messages"])
    ]
    assert [row["resource_id"] for row in resources] == ["m1-0-user"]
    assert resources[0]["text"] == payload["messages"][1]["content"]
    assert transformed["messages"] == [
        payload["messages"][0], payload["messages"][2], payload["messages"][3],
    ]
    assert resources[0]["metadata"] == {
        "selection_policy": "typed_bm25_embedding_rrf",
        "message_index": 1,
        "segment_index": 0,
        "role": "user",
        "parent_record_id": "m1",
        "causal_group_id": "record:m1",
    }


def test_record_aligned_children_keep_causal_and_result_metadata() -> None:
    observation = (
        "return code: 1\n"
        "Traceback (most recent call last):\n"
        "  File \"a.py\", line 1\n"
        "ValueError: bad\n"
        "diff --git a/a.py b/a.py\n"
        "@@ -1 +1 @@\n"
        "-bad\n+good\n"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "run failing test"},
            {
                "role": "tool",
                "content": observation,
                "tool_call_id": "call-1",
                "return_code": 1,
                "mutation_status": "changed",
                "result_metadata": {"test": "test_a"},
            },
            {"role": "assistant", "content": "inspect latest"},
            {"role": "user", "content": "current result"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=1.0,
        segment_tokens=4,
    )
    children = [
        row for row in transformed["pra"]["resources"]
        if row["metadata"]["message_index"] == 3
    ]

    assert "".join(row["text"] for row in children) == observation
    assert len(children) > 1
    assert {row["metadata"]["parent_record_id"] for row in children} == {"m3"}
    assert {row["metadata"]["causal_group_id"] for row in children} == {"turn:m2"}
    assert all(row["metadata"]["tool_call_id"] == "call-1" for row in children)
    assert all(row["metadata"]["return_code"] == 1 for row in children)
    assert all(row["metadata"]["mutation_status"] == "changed" for row in children)
    assert all(
        row["metadata"]["result_metadata"] == {"test": "test_a"}
        for row in children
    )


def test_partial_selection_keeps_assistant_observation_turns_atomic() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "old action short"},
            {"role": "user", "content": "old observation short"},
            {"role": "assistant", "content": "needle action"},
            {"role": "user", "content": "needle observation"},
            # mini-swe-agent can append this without retaining the malformed
            # assistant response that caused it.
            {"role": "user", "content": "needle format error"},
            {"role": "assistant", "content": "latest action"},
            {"role": "user", "content": "find needle"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload, mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.75, segment_tokens=2,
    )
    resources = transformed["pra"]["resources"]
    selected_indices = {
        row["metadata"]["message_index"] for row in resources
    }

    # Any selected historical action brings its causal observation(s).  The
    # active action and current observation are both mandatory rather than
    # splitting the causal tail between resources and visible messages.
    assert 4 in selected_indices
    assert {5, 6}.issubset(selected_indices)
    assert (2 in selected_indices) == (3 in selected_indices)
    assert transformed["messages"][-2:] == payload["messages"][-2:]


def test_causal_bundle_budget_rounds_retention_up_instead_of_underfilling() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "investigate old source"},
            {"role": "user", "content": " ".join(["large-observation"] * 20)},
            {"role": "assistant", "content": "active action"},
            {"role": "user", "content": "active result"},
        ]
    }

    transformed, trace = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.5,
        recent_completed_turns=0,
        recent_source_turns=0,
        recent_progress_turns=0,
        recent_mutation_turns=0,
        recent_verification_turns=0,
    )

    metadata = transformed["pra"]["metadata"]
    assert trace.physical_input_tokens_estimate >= math.ceil(
        trace.logical_input_tokens_estimate * 0.5
    )
    assert metadata["selection_budget_policy"] == "causal_bundle_round_up_v1"
    assert metadata["causal_round_up_tokens_estimate"] > 0
    assert metadata["target_retention_fraction"] == 0.5
    assert metadata["realized_retention_fraction"] >= 0.5
    assert metadata["retention_rounded_up"] is True
    assert {2, 3}.issubset({
        row["metadata"]["message_index"]
        for row in transformed["pra"]["resources"]
    })


def test_active_assistant_tool_tail_is_mandatory() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "run command"},
            {"role": "tool", "content": "command output"},
        ]
    }

    transformed, trace = transform_chat_payload(
        payload, mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.25,
    )

    assert transformed["messages"] == [
        payload["messages"][0], payload["messages"][2], payload["messages"][3],
    ]
    assert [row["resource_id"] for row in transformed["pra"]["resources"]] == [
        "m1-0-user"
    ]
    assert trace.token_saving_fraction_estimate == 0


def test_reduced_agent_history_keeps_two_completed_progress_turns() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "old action"},
            {"role": "user", "content": "old result"},
            {"role": "assistant", "content": "recent action one"},
            {"role": "user", "content": "recent result one"},
            {"role": "assistant", "content": "recent action two"},
            {"role": "user", "content": "recent result two"},
            {"role": "assistant", "content": "active action"},
            {"role": "user", "content": "active result"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload, mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.25, segment_tokens=16,
    )

    resource_indices = [
        row["metadata"]["message_index"] for row in transformed["pra"]["resources"]
    ]
    assert resource_indices == [1, 4, 5, 6, 7]
    assert transformed["messages"] == [
        payload["messages"][0], payload["messages"][8], payload["messages"][9],
    ]
    assert transformed["pra"]["metadata"]["pinned_progress_segments"] == [
        "m4-0-assistant", "m5-0-user", "m6-0-assistant", "m7-0-user",
    ]


def test_progress_spine_keeps_latest_mutation_outside_recency_window() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "sed -i 's/a/b/' source.py"},
            {"role": "user", "content": "mutation completed"},
            {"role": "assistant", "content": "inspect one"},
            {"role": "user", "content": "inspection one"},
            {"role": "assistant", "content": "inspect two"},
            {"role": "user", "content": "inspection two"},
            {"role": "assistant", "content": "inspect three"},
            {"role": "user", "content": "inspection three"},
            {"role": "assistant", "content": "active action"},
            {"role": "user", "content": "active result"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload, mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.25, segment_tokens=16,
    )

    resource_indices = {
        row["metadata"]["message_index"] for row in transformed["pra"]["resources"]
    }
    assert {2, 3, 6, 7, 8, 9}.issubset(resource_indices)


def test_progress_spine_retention_counts_are_explicit_and_auditable() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "sed -i 's/a/b/' one.py"},
            {"role": "user", "content": "mutation one"},
            {"role": "assistant", "content": "sed -i 's/b/c/' two.py"},
            {"role": "user", "content": "mutation two"},
            {"role": "assistant", "content": "pytest test_one.py"},
            {"role": "user", "content": "verification one"},
            {"role": "assistant", "content": "pytest test_two.py"},
            {"role": "user", "content": "verification two"},
            {"role": "assistant", "content": "inspect latest"},
            {"role": "user", "content": "latest inspection"},
            {"role": "assistant", "content": "active action"},
            {"role": "user", "content": "active result"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.01,
        segment_tokens=32,
        recent_completed_turns=1,
        recent_mutation_turns=2,
        recent_verification_turns=2,
    )

    resource_indices = {
        row["metadata"]["message_index"]
        for row in transformed["pra"]["resources"]
    }
    assert resource_indices == set(range(1, 12))
    policy = transformed["pra"]["metadata"]["retention_policy"]
    assert policy["recent_completed_turns"] == 1
    assert policy["recent_records_per_turn"] == 2
    assert policy["recent_source_turns"] == 1
    assert policy["recent_progress_turns"] == 1
    assert policy["recent_mutation_turns"] == 2
    assert policy["recent_verification_turns"] == 2
    assert policy["large_record_chunk_tokens"] == 32
    assert policy["preserve_action_observation_pairs"] is True


def test_request_metadata_controls_dense_turn_record_floor_and_chunk_size() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "call several tools"},
            {"role": "tool", "content": "old observation one"},
            {"role": "tool", "content": "old observation two"},
            {"role": "tool", "content": "new observation three"},
            {"role": "tool", "content": "new observation four"},
            {"role": "assistant", "content": "active action"},
            {"role": "tool", "content": "active result"},
        ],
        "pra": {
            "metadata": {
                "caller_trace_id": "trace-17",
                "retention_policy": {
                    "recent_completed_turns": 1,
                    "recent_records_per_turn": 2,
                    "recent_source_turns": 0,
                    "recent_mutation_turns": 0,
                    "recent_verification_turns": 0,
                    "large_record_chunk_tokens": 2,
                    "max_records_per_turn_before_chunking": 3,
                }
            }
        },
    }

    transformed, _ = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.01,
    )

    resource_indices = {
        row["metadata"]["message_index"]
        for row in transformed["pra"]["resources"]
    }
    # The action is retained to preserve causality and the newest two records
    # satisfy the per-turn floor; earlier observations remain selectable.
    assert {2, 5, 6}.issubset(resource_indices)
    assert transformed["pra"]["metadata"]["retention_policy"] == {
        "recent_completed_turns": 1,
        "recent_records_per_turn": 2,
        "recent_source_turns": 0,
        "recent_progress_turns": 1,
        "recent_mutation_turns": 0,
        "recent_verification_turns": 0,
        "large_record_chunk_tokens": 2,
        "max_records_per_turn_before_chunking": 3,
        "preserve_action_observation_pairs": True,
        "causal_bundle_round_up": True,
    }
    assert transformed["pra"]["metadata"]["caller_trace_id"] == "trace-17"
    assert all(
        len(row["text"].split()) <= 2
        for row in transformed["pra"]["resources"]
    )


def test_hard_cap_policy_does_not_round_up_an_oversized_causal_bundle() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "alpha beta gamma delta"},
            {"role": "tool", "content": "one two three four"},
            {"role": "assistant", "content": "active"},
            {"role": "tool", "content": "result"},
        ],
        "pra": {
            "metadata": {
                "retention_policy": {
                    "recent_completed_turns": 0,
                    "recent_source_turns": 0,
                    "recent_mutation_turns": 0,
                    "recent_verification_turns": 0,
                    "causal_bundle_round_up": False,
                }
            }
        },
    }

    transformed, _ = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.5,
    )

    assert transformed["pra"]["metadata"]["selection_budget_policy"] == (
        "causal_bundle_hard_cap_v1"
    )


def test_progress_spine_rejects_negative_retention_counts() -> None:
    payload = {"messages": [{"role": "user", "content": "task"}]}

    with pytest.raises(ValueError, match="non-negative"):
        transform_chat_payload(
            payload,
            mode=ContextTreatment.DIRECT_NATIVE_PRA,
            budget_fraction=1,
            recent_completed_turns=-1,
        )


def test_zero_progress_retention_does_not_accidentally_pin_all_history() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "old inspection"},
            {"role": "user", "content": "old result"},
            {"role": "assistant", "content": "active action"},
            {"role": "user", "content": "active result"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.01,
        recent_completed_turns=0,
        recent_source_turns=0,
        recent_progress_turns=0,
        recent_mutation_turns=0,
        recent_verification_turns=0,
    )

    assert [
        row["metadata"]["message_index"]
        for row in transformed["pra"]["resources"]
    ] == [1]


def test_latest_source_evidence_turn_is_pinned_beyond_recency_window() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "```mswea_bash_command\ncat source.py\n```"},
            {"role": "user", "content": "def target(): return old"},
            {"role": "assistant", "content": "python -c 'import missing_dependency'"},
            {"role": "user", "content": "ModuleNotFoundError"},
            {"role": "assistant", "content": "inspect environment"},
            {"role": "user", "content": "environment result"},
            {"role": "assistant", "content": "active action"},
            {"role": "user", "content": "active result"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.01,
        recent_completed_turns=2,
        recent_source_turns=1,
        recent_mutation_turns=0,
        recent_verification_turns=0,
    )

    resource_indices = {
        row["metadata"]["message_index"]
        for row in transformed["pra"]["resources"]
    }
    assert {2, 3}.issubset(resource_indices)


def test_task04_request5_progress_state_and_source_survive_at_90_percent() -> None:
    task04_request5 = (
        "THOUGHT: Let me look at how lazy objects are typically handled in Django. "
        "The issue is that `get_format` function is called with a lazy object (like "
        "from `_()` function) as the `format_type` parameter, but `getattr()` expects "
        "a string. \n\nLooking at the `get_format_lazy` definition, it seems like it's "
        "meant to handle lazy objects, but the problem is that `get_format_lazy` is a "
        "lazy wrapper around `get_format`, not the `get_format` function itself.\n\n"
        "The issue occurs when someone calls `get_format(_('Y-m-d'))` directly, not "
        "when using `get_format_lazy`. The fix should be to ensure that `format_type` "
        "is converted to a string before being used in `getattr()` calls.\n\nLet me "
        "create a simple test to reproduce the issue first:\n\n"
        "```mswea_bash_command\ncd /testbed && python -c \"\n"
        "from django.utils.formats import get_format\n"
        "from django.utils.translation import gettext_lazy as _\n"
        "try:\n    result = get_format(_('Y-m-d'))\n"
        "    print('Success:', result)\nexcept Exception as e:\n"
        "    print('Error:', type(e).__name__, str(e))\n\"\n```\n"
    )
    assert hashlib.sha256(task04_request5.encode()).hexdigest() == (
        "313c708e78de9ca9d768011bb2fecc6fa1658fd92761bef98733ac589f5c56ad"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix lazy get_format " + "task " * 80},
            {
                "role": "assistant",
                "content": "THOUGHT: inspect source\n```mswea_bash_command\n"
                "cat /testbed/django/utils/formats.py\n```",
            },
            {"role": "user", "content": "source evidence " * 120},
            {
                "role": "assistant",
                "content": "THOUGHT: I will inspect another file in detail.\n"
                "```mswea_bash_command\nfind /testbed -name '*.py'\n```",
            },
            {"role": "user", "content": "unrelated paths " * 120},
            {"role": "assistant", "content": task04_request5},
            {"role": "user", "content": "ModuleNotFoundError: asgiref"},
            {
                "role": "assistant",
                "content": "THOUGHT: inspect environment\n"
                "```mswea_bash_command\nenv\n```",
            },
            {"role": "user", "content": "environment output " * 120},
            {"role": "assistant", "content": "THOUGHT: current action"},
            {"role": "user", "content": "current observation"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.90,
        recent_completed_turns=0,
        recent_records_per_turn=2,
        recent_source_turns=1,
        recent_progress_turns=1,
        recent_mutation_turns=0,
        recent_verification_turns=0,
        causal_bundle_round_up=False,
    )

    by_index = {
        row["metadata"]["message_index"]: row
        for row in transformed["pra"]["resources"]
    }
    assert {2, 3, 6, 7}.issubset(by_index)
    assert by_index[6]["text"] == task04_request5
    assert transformed["pra"]["metadata"]["pinned_progress_state_segments"] == [
        "m6-0-assistant", "m7-0-user",
    ]


def test_verbose_reasoning_without_explicit_progress_state_is_not_pinned() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "THOUGHT: I will carefully explore the repository and read "
                "several files before deciding what to do next.\n"
                "```mswea_bash_command\nfind /testbed -type f\n```",
            },
            {"role": "user", "content": "paths"},
            {"role": "assistant", "content": "active"},
            {"role": "user", "content": "result"},
        ]
    }

    transformed, _ = transform_chat_payload(
        payload,
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.01,
        recent_completed_turns=0,
        recent_source_turns=0,
        recent_progress_turns=1,
        recent_mutation_turns=0,
        recent_verification_turns=0,
        causal_bundle_round_up=False,
    )

    assert transformed["pra"]["metadata"]["pinned_progress_state_segments"] == []


def test_causal_chat_validation_rejects_adjacent_assistant_messages() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "observation"},
    ]

    with pytest.raises(ValueError, match="adjacent assistant"):
        CausalChatNativePromptMixin._validate_causal_messages(messages)


def test_turn_bundle_selection_has_exact_frozen_replay() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task alpha"},
            {"role": "assistant", "content": "inspect alpha"},
            {"role": "user", "content": "alpha output"},
            {"role": "assistant", "content": "inspect beta"},
            {"role": "user", "content": "beta output"},
            {"role": "user", "content": "format error guidance"},
            {"role": "assistant", "content": "retry beta"},
            {"role": "user", "content": "find beta"},
        ]
    }
    selected, first_trace = transform_chat_payload(
        payload, mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.75, segment_tokens=2,
    )
    frozen = [
        (row["resource_id"], row["text"])
        for row in selected["pra"]["resources"]
    ]

    replayed, replay_trace = transform_chat_payload(
        payload, mode=ContextTreatment.GATEWAY_NATIVE_PRA,
        budget_fraction=0.75, segment_tokens=2,
        frozen_selection=frozen,
    )

    assert [
        (row["resource_id"], row["text"])
        for row in replayed["pra"]["resources"]
    ] == frozen
    assert replay_trace.selected_resource_digest == first_trace.selected_resource_digest


def test_frozen_replay_rejects_stale_duplicate_or_reordered_resources() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task alpha"},
            {"role": "assistant", "content": "inspect alpha"},
            {"role": "user", "content": "alpha output"},
            {"role": "assistant", "content": "inspect beta"},
            {"role": "user", "content": "beta output"},
            {"role": "user", "content": "current observation"},
        ]
    }
    selected, _ = transform_chat_payload(
        payload, mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=1.0, segment_tokens=8,
    )
    frozen = [
        (row["resource_id"], row["text"])
        for row in selected["pra"]["resources"]
    ]

    with pytest.raises(ValueError, match="duplicate resource IDs"):
        transform_chat_payload(
            payload, mode=ContextTreatment.GATEWAY_NATIVE_PRA,
            budget_fraction=1.0, segment_tokens=8,
            frozen_selection=[*frozen, frozen[-1]],
        )
    with pytest.raises(ValueError, match="exact subset"):
        transform_chat_payload(
            payload, mode=ContextTreatment.GATEWAY_NATIVE_PRA,
            budget_fraction=1.0, segment_tokens=8,
            frozen_selection=[*frozen[:-1], ("m999-0-user", "stale")],
        )
    with pytest.raises(ValueError, match="causal resource order"):
        transform_chat_payload(
            payload, mode=ContextTreatment.GATEWAY_NATIVE_PRA,
            budget_fraction=1.0, segment_tokens=8,
            frozen_selection=[frozen[0], *reversed(frozen[1:])],
        )


def test_native_agent_split_is_an_exact_causal_chat_template_prefix() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "alpha  beta\n\ngamma delta"},
            {"role": "assistant", "content": "inspect\n```tool\nls\n```"},
            {"role": "user", "content": "latest observation"},
        ]
    }
    transformed, _ = transform_chat_payload(
        payload, mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=1.0, segment_tokens=2,
    )

    render_calls = []

    class FakeNativeBase:
        def _request_json(self, path, body):
            assert path == "/apply-template"
            render_calls.append((len(body["messages"]), body["add_generation_prompt"]))
            prompt = "<chat>" + "".join(
                f"<{row['role']}>{row['content']}</{row['role']}>"
                for row in body["messages"]
            )
            if body["add_generation_prompt"]:
                prompt += "<assistant>"
            return {"prompt": prompt}

    class Executor(CausalChatNativePromptMixin, FakeNativeBase):
        pass

    resources = tuple(
        SimpleNamespace(
            resource_id=row["resource_id"], text=row["text"],
            metadata=row["metadata"],
        )
        for row in transformed["pra"]["resources"]
    )
    request = SimpleNamespace(
        request_id="causal-prefix-test",
        resources=resources,
        messages=tuple(transformed["messages"]),
        tools=(),
    )
    executor = Executor()
    prefix, suffix = executor._causal_prompt_pair(request)
    assert executor._causal_prompt_pair(request) == (prefix, suffix)
    expected = Executor()._render_chat(payload["messages"], request, generate=True)

    assert prefix + suffix == expected
    assert payload["messages"][1]["content"] in prefix
    assert payload["messages"][2]["content"] not in prefix
    assert payload["messages"][2]["content"] in suffix
    assert "latest observation" not in prefix
    assert "latest observation" in suffix
    # The physical model validates the complete request before the resource
    # prefix is rendered for materialization, and repeated adapter accessors
    # reuse that validated pair instead of issuing duplicate template calls.
    assert render_calls[:2] == [(4, True), (2, False)]
    assert len(render_calls) == 3


def test_changed_resource_uses_in_place_prefix_delta_without_delete() -> None:
    calls = []

    class FakeNativeBase:
        def __init__(self):
            self._slot_identities = {0: SimpleNamespace(resource_digest="old")}

        @staticmethod
        def _resource_text(request):
            return request.text

        @staticmethod
        def _resource_digest(request, text):
            return text

        def _request_json(self, path, body):
            calls.append((path, body))
            return {"timings": {"cache_n": 11, "prompt_n": 17}}

        def _delete_resource(self, slot):
            raise AssertionError("prefix-delta update must not delete the resource slot")

    class Executor(CausalChatNativePromptMixin, FakeNativeBase):
        pass

    executor = Executor()
    identity = SimpleNamespace(resource_digest="new")
    lease = SimpleNamespace(resource_slot=0, identity=identity)
    request = SimpleNamespace(text="new", resources=())

    assert executor._ensure_resource(request, lease) == ("new", True)
    assert calls == [(
        "/completion",
        {
            "prompt": "new",
            "id_slot": 0,
            "n_predict": 0,
            "cache_prompt": True,
            "temperature": 0,
            "pra_pin_resource": True,
        },
    )]
    assert executor._resource_update_metrics["new"] == {
        "resource_update_mode": "prefix_delta",
        "resource_prefix_cached_tokens": 11,
        "resource_evaluated_tokens": 17,
        "resource_total_tokens": 28,
    }
    assert executor._ensure_resource(request, lease) == ("new", False)
    assert len(calls) == 1
    assert executor._resource_update_metrics["new"]["resource_update_mode"] == "identity_hit"
    assert executor._resource_update_metrics["new"]["resource_evaluated_tokens"] == 0
    assert executor._resource_update_metrics["new"]["resource_total_tokens"] == 28


def test_hybrid_llamacpp_advertises_implemented_resource_delta() -> None:
    native = SimpleNamespace(
        capabilities=lambda: PRAEngineCapabilities(
            adapter="llama_cpp_pra",
            integration_level="E2",
            native_kv=True,
            session_state=True,
            live_prefix_kv_capture=True,
            live_prefix_kv_subset=True,
            zero_selected_text_reencoding=True,
            stable_record_kv_identity=True,
            multiple_selected_records=True,
            source_positions_preserved=True,
            request_membership_attach=True,
        ),
    )
    adapter = HybridLlamaCppAdapter(
        native,
        SimpleNamespace(),
        prefix_caching=True,
        slot_save_path=Path("agent-slot-checkpoints"),
    )

    capabilities = adapter.capabilities()

    assert capabilities.native_kv is True
    assert capabilities.resource_delta is True
    assert capabilities.cache_affinity is True
    assert capabilities.pinned_kv is True
    assert capabilities.agent_history_kv_qualified is True


def test_hybrid_llamacpp_qualification_is_configuration_specific() -> None:
    native = SimpleNamespace(
        capabilities=lambda: PRAEngineCapabilities(
            adapter="llama_cpp_pra",
            integration_level="E2",
            native_kv=True,
            session_state=True,
            live_prefix_kv_capture=True,
            live_prefix_kv_subset=True,
            zero_selected_text_reencoding=True,
            stable_record_kv_identity=True,
            multiple_selected_records=True,
            source_positions_preserved=True,
            request_membership_attach=True,
        ),
    )

    cache_off = HybridLlamaCppAdapter(
        native, SimpleNamespace(), prefix_caching=False, slot_save_path=Path("slots"),
    ).capabilities()
    no_offload = HybridLlamaCppAdapter(
        native, SimpleNamespace(), prefix_caching=True,
    ).capabilities()

    assert cache_off.agent_history_kv_qualified is False
    assert cache_off.pinned_kv is False
    assert no_offload.agent_history_kv_qualified is False
    assert no_offload.pinned_kv is True


def test_hybrid_llamacpp_reconstructs_complete_ordered_g11_resource_delta() -> None:
    task = PRAWireResource(
        resource_id="m1-0-user", uri="pra://m1", text="pinned task",
    )
    action = PRAWireResource(
        resource_id="m2-0-assistant", uri="pra://m2", text="first command",
    )
    observation = PRAWireResource(
        resource_id="m3-0-user", uri="pra://m3", text="first result",
    )
    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(capabilities=lambda: PRAEngineCapabilities(
            adapter="llama_cpp_pra", integration_level="E2", native_kv=True,
        )),
        SimpleNamespace(),
        prefix_caching=True,
    )
    first = PRAWireRequest(
        model="model", messages=({"role": "user", "content": "next"},),
        tenant_id="tenant", session_id="session", resources=(task,),
    )
    adapter._hydrate_resource_delta(first)
    delta = PRAWireRequest(
        model="model", messages=({"role": "user", "content": "next"},),
        tenant_id="tenant", session_id="session",
        resources=(action, observation),
        resource_ops=(
            ResourceDelta(ResourceOperation.UNCHANGED, task.resource_id, task.uri, "1"),
            ResourceDelta(ResourceOperation.ADD, action.resource_id, action.uri, "1", resource=action),
            ResourceDelta(
                ResourceOperation.ADD, observation.resource_id, observation.uri, "1",
                resource=observation,
            ),
        ),
    )

    hydrated = adapter._hydrate_resource_delta(delta)

    assert [row.resource_id for row in hydrated.resources] == [
        task.resource_id, action.resource_id, observation.resource_id,
    ]
    assert [row.text for row in hydrated.resources] == [
        task.text, action.text, observation.text,
    ]
    assert hydrated.resource_ops == ()


def test_hybrid_llamacpp_resource_delta_fails_closed_without_prior_body() -> None:
    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(), SimpleNamespace(), prefix_caching=True,
    )
    request = PRAWireRequest(
        model="model", messages=({"role": "user", "content": "next"},),
        tenant_id="tenant", session_id="session",
        resource_ops=(ResourceDelta(
            ResourceOperation.UNCHANGED, "missing", "pra://missing", "1",
        ),),
    )

    with pytest.raises(ValueError, match="cannot reconstruct"):
        adapter._hydrate_resource_delta(request)


def test_g11_keeps_session_for_declared_detached_history_projection() -> None:
    gateway = object.__new__(PRAGateway)
    gateway.mode = PRAGatewayMode.G11_MEDIATION
    state = SimpleNamespace(
        turns=2,
        model_revision=None,
        chat_template_digest=None,
        visible_prefix_profile=None,
    )
    turn = SimpleNamespace(
        state=state, prefix_changed_reason="history_rewrite",
    )
    request = SimpleNamespace(metadata={
        "history_projection": "detached-agent-trajectory-v1",
    })

    assert gateway._invalidation_reason(turn, request) is None

    request.metadata = {}
    assert gateway._invalidation_reason(
        turn, request,
    ) == "system_prefix_or_history_rewrite"


def test_llamacpp_wrapper_exposes_same_engine_g00_and_g11_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mode in ("g00", "g11"):
        monkeypatch.setattr(sys, "argv", [
            "serve_llamacpp_pra", "--port", "18101", "--mode", mode,
            "--model", "model", "--model-fingerprint", "fingerprint",
        ])
        assert parse_llamacpp_server_args().mode == mode


def test_prefix_cache_off_complete_selection_uses_full_plain_prompt() -> None:
    calls = []

    class Native:
        request_slot = 1

        @staticmethod
        def _erase_request_slot(slot):
            assert slot == 1

        @staticmethod
        def _causal_prompt_pair(request):
            return "selected-prefix", "current-suffix"

        @staticmethod
        def _request_json(path, body):
            calls.append((path, body))
            return {"content": "cold", "tokens": [], "timings": {"cache_n": 0}}

    native_adapter = SimpleNamespace(native_executor=Native())
    plain = PlainSlotExecutor(native_adapter.native_executor, prefix_caching=False)
    adapter = HybridLlamaCppAdapter(
        native_adapter, plain, prefix_caching=False,
    )
    adapter._live_session_slots["session"] = 1
    adapter._generate_from_live_slot = lambda request, source: (_ for _ in ()).throw(
        AssertionError("cache-off endpoint must not continue a live request slot")
    )
    adapter._remember_resident_tokens = lambda request, result: None
    request = SimpleNamespace(
        openai_fields={},
        to_dict=lambda: {
            "model": "model",
            "messages": [{"role": "user", "content": "current"}],
            "session_id": "session",
            "resources": [{
                "resource_id": "m1-0-user",
                "uri": "pra://agent-trajectory/m1-0-user",
                "text": "task",
                "metadata": {"message_index": 1, "segment_index": 0, "role": "user"},
            }],
            "metadata": {"selection_complete": True},
        },
    )

    result = adapter.generate(request)

    assert result.text == "cold"
    assert calls == [(
        "/completion",
        {
            "prompt": "selected-prefixcurrent-suffix",
            "id_slot": 1,
            "n_predict": 64,
            "cache_prompt": False,
            "temperature": 0.0,
            "seed": 0,
            "return_tokens": True,
        },
    )]


def test_live_llamacpp_prefix_uses_resident_tokens_not_last_prompt_length() -> None:
    class Native:
        request_slot = 1

        @staticmethod
        def _causal_prompt_pair(request):
            return "resident prefix", "suffix"

        @staticmethod
        def _request_json(path, body=None):
            if path == "/tokenize":
                return {"tokens": list(range(23))}
            assert path == "/slots"
            # This is intentionally the most recent wire-prompt length, which
            # is unrelated to the full sequence length after native attach.
            return [{"id": 0, "n_prompt_tokens": 4, "is_processing": False}]

    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=Native()), SimpleNamespace(),
        prefix_caching=True,
    )
    adapter._live_session_tokens["session"] = tuple(range(23))

    assert adapter._live_prefix_matches(SimpleNamespace(session_id="session"), 0)


def test_complete_selection_continues_live_slot_without_detached_prefix_check() -> None:
    class Native:
        request_slot = 1

    class Adapter:
        native_executor = Native()

    adapter = HybridLlamaCppAdapter(
        Adapter(), SimpleNamespace(), prefix_caching=True,
    )
    adapter._live_session_slots["session"] = 1
    request = SimpleNamespace(
        session_id="session",
        resources=(SimpleNamespace(resource_id="history"),),
        metadata={"selection_complete": True},
        openai_fields={"prefix_caching": True},
        to_dict=lambda: {
            "model": "model",
            "session_id": "session",
            "resources": [{
                "resource_id": "history", "uri": "memory://history",
                "text": "history",
            }],
            "messages": [{"role": "user", "content": "continue"}],
            "tools": [],
            "metadata": {"selection_complete": True},
            "openai_fields": {"prefix_caching": True},
        },
    )
    continued = SimpleNamespace(raw={"tokens": []})
    adapter._live_prefix_matches = lambda *_: (_ for _ in ()).throw(
        AssertionError("complete selection must not enter detached-prefix qualification")
    )
    adapter._generate_from_live_slot = lambda req, slot: (continued, slot)
    adapter._remember_resident_tokens = lambda *_: None

    assert adapter.generate(request) is continued


def test_live_record_plan_selects_resident_kv_without_omitted_text_prefill() -> None:
    calls = []

    class Native:
        request_slot = 1
        resource_slot = 0

        @staticmethod
        def _render_chat(messages, request, *, generate):
            del request, generate
            return "".join(str(message["content"]) for message in messages)

        @staticmethod
        def _request_json(path, body=None):
            assert path == "/tokenize"
            return {"tokens": [ord(char) for char in body["content"]]}

        @staticmethod
        def generate_live_prefix(request, *, prompt_suffix, plan, request_slot):
            calls.append((tuple(prompt_suffix), plan, request_slot))
            return PRAEngineResult(
                "answer",
                {
                    "tokens": [ord("Z")],
                    "pra": {
                        "kv_source": "live_prefix_capture",
                        "selected_kv_tokens": plan.selected_tokens,
                        "selected_text_reencoded_tokens": 0,
                        "commit_succeeded": True,
                    },
                },
                (),
            )

    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=Native()), SimpleNamespace(),
        prefix_caching=True,
    )
    # The canonical source contains S(system), T(task), an old A/O turn, and
    # the active B action. The policy selects only S, T, and active B. C is the
    # new observation and is the only prompt token evaluated this turn.
    adapter._live_session_tokens["session"] = tuple(map(ord, "STAOB"))
    logical = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "T"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "O"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]
    resource = PRAWireResource(
        resource_id="m1-0-user",
        uri="pra://agent-trajectory/m1-0-user",
        text="T",
        metadata={
            "message_index": 1,
            "segment_index": 0,
            "role": "user",
            "parent_record_id": "m1",
            "causal_group_id": "record:m1",
        },
    )
    request = PRAWireRequest(
        model="model",
        messages=(logical[0], logical[4], logical[5]),
        resources=(resource,),
        session_id="session",
        metadata={"mandatory_message_indices": [0, 4, 5]},
    )

    result = adapter._generate_from_live_records(request, 1, logical)

    assert result.text == "answer"
    suffix, plan, destination = calls[0]
    assert suffix == (ord("C"),)
    assert destination == 0
    assert plan.source_tokens == 5
    assert plan.selected_tokens == 3
    assert [(row.record_id, row.start, row.end) for row in plan.ranges] == [
        ("system-prefix", 0, 1),
        ("m1", 1, 2),
        ("active-tail:m4", 4, 5),
    ]
    assert result.raw["pra"]["selected_text_reencoded_tokens"] == 0


def test_full_live_record_selection_continues_canonical_prefix_without_slot_handoff() -> None:
    continued = []

    class Native:
        request_slot = 1
        resource_slot = 0

        @staticmethod
        def _render_chat(messages, request, *, generate):
            del request, generate
            return "".join(str(message["content"]) for message in messages)

        @staticmethod
        def _request_json(path, body=None):
            assert path == "/tokenize"
            return {"tokens": [ord(char) for char in body["content"]]}

        @staticmethod
        def generate_live_prefix(*args, **kwargs):
            raise AssertionError("full retention must not hand K/V to another slot")

    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=Native()), SimpleNamespace(),
        prefix_caching=True,
    )
    adapter._live_session_tokens["session"] = tuple(map(ord, "STAOB"))
    adapter._generate_from_live_slot = lambda request, source: (
        continued.append((request, source)) or PRAEngineResult(
            "answer",
            {"pra": {"native_tokens": 5, "wire_tokens": 1}},
            ({"stage": "llama_cpp_live_prefix_continue"},),
        ),
        source,
    )
    logical = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "T"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "O"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]

    def resource(index, role):
        return PRAWireResource(
            resource_id=f"m{index}-0-{role}",
            uri=f"pra://agent-trajectory/m{index}-0-{role}",
            text=logical[index]["content"],
            metadata={
                "message_index": index,
                "segment_index": 0,
                "role": role,
                "parent_record_id": f"m{index}",
                "causal_group_id": f"record:m{index}",
            },
        )

    request = PRAWireRequest(
        model="model",
        messages=(logical[0], logical[4], logical[5]),
        resources=(resource(1, "user"), resource(2, "assistant"), resource(3, "user")),
        session_id="session",
        metadata={"mandatory_message_indices": [0, 4, 5]},
    )

    result = adapter._generate_from_live_records(request, 1, logical)

    assert continued == [(request, 1)]
    assert result.text == "answer"
    assert result.raw["pra"]["selected_kv_tokens"] == 5
    assert result.raw["pra"]["selected_text_reencoded_tokens"] == 0
    assert result.raw["pra"]["exact_live_prefix_continuation"] is True
    assert result.trace[-1]["stage"] == "llama_cpp_live_prefix_full_continue"


def test_fresh_prefill_control_consumes_the_same_selected_logical_records() -> None:
    consumed = []

    class Plain:
        def generate(self, request):
            consumed.append(request)
            return PRAEngineResult(
                "answer",
                {"timings": {"prompt_n": 17}},
                ({"stage": "llama_cpp_plain", "prefix_cache_enabled": False},),
            )

    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=SimpleNamespace()), Plain(),
        prefix_caching=False,
    )
    logical = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "T"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "O"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]

    def resource(index, role):
        return PRAWireResource(
            resource_id=f"m{index}-0-{role}",
            uri=f"pra://agent-trajectory/m{index}-0-{role}",
            text=logical[index]["content"],
            metadata={
                "message_index": index,
                "segment_index": 0,
                "role": role,
                "parent_record_id": f"m{index}",
                "causal_group_id": f"record:m{index}",
            },
        )

    request = PRAWireRequest(
        model="model",
        messages=(logical[0], logical[4], logical[5]),
        resources=(
            resource(1, "user"), resource(2, "assistant"), resource(3, "user"),
        ),
        session_id="session",
        metadata={"mandatory_message_indices": [0, 4, 5]},
    )

    result = adapter._generate_fresh_selected_records(request, logical)

    assert [dict(message) for message in consumed[0].messages] == logical
    assert consumed[0].resources == ()
    assert result.raw["pra"]["native_kv"] is False
    assert result.raw["pra"]["kv_source"] == "fresh_selected_prefill"
    assert result.raw["pra"]["selected_text_reencoded_tokens"] == 17
    assert result.trace[-1]["stage"] == "llama_cpp_fresh_selected_prefill_control"


def test_closing_live_agent_session_erases_request_and_resource_membership() -> None:
    erased = []
    closed = []
    native = SimpleNamespace(_erase_request_slot=erased.append)
    native_adapter = SimpleNamespace(
        native_executor=native,
        close_session=closed.append,
    )
    adapter = HybridLlamaCppAdapter(
        native_adapter, SimpleNamespace(), prefix_caching=True,
    )
    adapter._live_session_slots["session"] = 1
    adapter._live_session_tokens["session"] = (1, 2)
    adapter._logical_session_messages["session"] = [
        {"role": "user", "content": "task"},
    ]

    adapter.close_session("session")

    assert erased == [1]
    assert closed == ["session"]
    assert "session" not in adapter._live_session_slots
    assert "session" not in adapter._live_session_tokens
    assert "session" not in adapter._logical_session_messages


def test_direct_llamacpp_endpoint_can_close_a_live_session() -> None:
    closed = []
    adapter = SimpleNamespace(close_session=closed.append)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), _direct_handler(adapter, "model"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/pra/sessions/seed%201",
            method="DELETE",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
        assert payload == {"closed": True, "session_id": "seed 1"}
        assert closed == ["seed 1"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_live_agent_template_is_qualified_before_inference() -> None:
    class Executor(CausalChatNativePromptMixin):
        inject_final_assistant_wrapper = False

        def _request_json(self, path, body):
            if path == "/apply-template":
                rows = body["messages"]
                rendered = "".join(
                    f"<{row['role']}>{row['content']}" for row in rows
                )
                if (
                    self.inject_final_assistant_wrapper
                    and rows[-1]["role"] == "assistant"
                ):
                    rendered = rendered[:-len(rows[-1]["content"])] + (
                        "<think></think>" + rows[-1]["content"]
                    )
                if body["add_generation_prompt"]:
                    rendered += "<assistant>"
                return {"prompt": rendered}
            assert path == "/tokenize"
            return {"tokens": [ord(char) for char in body["content"]]}

    result = Executor().validate_record_prefix_template()
    assert result["record_prefix_separable"] is True
    assert result["probe_messages"] == 4

    Executor.inject_final_assistant_wrapper = True
    with pytest.raises(RuntimeError, match="not record-prefix-separable"):
        Executor().validate_record_prefix_template()


def test_live_agent_sessions_use_disjoint_pairs_concurrently() -> None:
    completion_slots = []
    erased = []
    closed = []
    overlap = threading.Barrier(2, timeout=5)

    class Native:
        request_slot = 1
        resource_slot = 0
        slot_allocator = SimpleNamespace(
            request_slots=(1, 3), resource_slots=(0, 2),
        )

        @staticmethod
        def _query_text(request):
            return str(request.messages[-1]["content"])

        @staticmethod
        def _request_json(path, body=None):
            if path == "/tokenize":
                return {"tokens": [ord(char) for char in body["content"]]}
            assert path == "/completion"
            completion_slots.append(body["id_slot"])
            overlap.wait()
            return {
                "content": f"answer-{body['id_slot']}",
                "tokens": [100 + body["id_slot"]],
                "timings": {"cache_n": 0},
            }

        @staticmethod
        def _erase_request_slot(slot):
            erased.append(slot)

    native = Native()
    native_adapter = SimpleNamespace(
        native_executor=native, close_session=closed.append,
    )
    adapter = HybridLlamaCppAdapter(
        native_adapter,
        PlainSlotExecutor(native, prefix_caching=True),
        prefix_caching=True,
    )
    requests = [
        PRAWireRequest(
            model="model",
            messages=({"role": "user", "content": f"task-{session}"},),
            session_id=session,
            metadata={"history_projection": "live-agent-kv-v1"},
        )
        for session in ("a", "b")
    ]
    errors = []

    def generate(request):
        try:
            adapter.generate(request)
        except Exception as error:  # pragma: no cover - assertion reports value
            errors.append(error)

    threads = [threading.Thread(target=generate, args=(request,)) for request in requests]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert set(completion_slots) == {1, 3}
    assert adapter._live_session_slots == {"a": 1, "b": 3}
    assert adapter._live_session_request_slots == {"a": 0, "b": 2}

    adapter.close_session("a")
    adapter.close_session("b")
    assert set(erased) == {0, 1, 2, 3}
    assert closed == ["a", "b"]


def test_live_agent_generation_head_is_idempotent_and_rejects_stale_forks() -> None:
    native = SimpleNamespace(request_slot=1, resource_slot=0)
    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=native), SimpleNamespace(),
        prefix_caching=True,
    )
    logical = [{"role": "user", "content": "task"}]
    request = PRAWireRequest(
        model="model", messages=tuple(logical), session_id="session",
        request_id="first",
    )
    result = PRAEngineResult("answer", {"tokens": [1]}, ({"stage": "live"},))
    adapter._record_generation_head(request, logical, result)

    retry = PRAWireRequest(
        model="model", messages=tuple(logical), session_id="session",
        request_id="retry-with-a-new-transport-id",
    )
    replay = adapter._validate_generation_parent(retry, logical)
    assert replay is not None
    assert replay.text == "answer"
    assert replay.trace[-1]["stage"] == "llama_cpp_idempotent_generation_replay"

    stale_fork = PRAWireRequest(
        model="model", messages=tuple(logical), session_id="session",
        request_id="different", openai_fields={"seed": 99},
    )
    with pytest.raises(SessionCommitConflict, match="different generation"):
        adapter._validate_generation_parent(stale_fork, logical)

    accepted = [
        *logical,
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "tool output"},
    ]
    assert adapter._validate_generation_parent(retry, accepted) is None
    rejected_recovery = [
        *logical,
        {"role": "user", "content": "format error: retry the action"},
    ]
    assert adapter._validate_generation_parent(retry, rejected_recovery) is None


def test_closed_live_agent_session_cannot_be_reopened() -> None:
    erased = []
    closed = []
    native = SimpleNamespace(
        request_slot=1,
        resource_slot=0,
        _erase_request_slot=erased.append,
    )
    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=native, close_session=closed.append),
        SimpleNamespace(),
        prefix_caching=True,
    )
    adapter._live_session_slots["session"] = 1
    adapter._live_session_request_slots["session"] = 0
    adapter.close_session("session")
    with pytest.raises(SessionClosedError, match="is closed"):
        adapter.generate(PRAWireRequest(
            model="model",
            messages=({"role": "user", "content": "late request"},),
            session_id="session",
        ))
    assert erased == [1, 0]
    assert closed == ["session"]


def test_live_agent_offload_reclaims_pair_and_restore_repins_without_prefill(
    tmp_path: Path,
) -> None:
    calls = []
    erased = []

    class Native:
        request_slot = 1
        resource_slot = 0
        slot_allocator = SimpleNamespace(request_slots=(1,), resource_slots=(0,))

        @staticmethod
        def _request_json(path, body=None):
            calls.append((path, body))
            if "action=save" in path:
                return {"n_tokens": 42, "n_bytes": 4096}
            if "action=restore" in path:
                return {"n_tokens": 42, "n_bytes": 4096}
            raise AssertionError(path)

        @staticmethod
        def _erase_request_slot(slot):
            erased.append(slot)

    native = Native()
    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=native, close_session=lambda session: None),
        SimpleNamespace(),
        prefix_caching=True,
        slot_save_path=tmp_path,
    )
    adapter._live_session_slots["session"] = 1
    adapter._live_session_request_slots["session"] = 0
    adapter._live_session_tokens["session"] = (1, 2, 3)

    receipt = adapter.offload_session("session")
    assert receipt["saved_tokens"] == 42
    assert receipt["saved_bytes"] == 4096
    assert erased == [1, 0]
    assert "session" not in adapter._live_session_slots
    assert adapter.session_status("session")["offloaded"] is True

    adapter._restore_offloaded_session("session")
    assert adapter._live_session_slots["session"] == 1
    assert adapter._live_session_request_slots["session"] == 0
    assert adapter.session_status("session")["offloaded"] is False
    assert calls[-1][0] == "/slots/1?action=restore"
    assert calls[-1][1]["pra_pin_resource"] is True
    assert all(path != "/completion" for path, _ in calls)


def test_frozen_prefix_probe_compares_recorded_responses(tmp_path: Path) -> None:
    fixture = tmp_path / "interaction.jsonl"
    fixture.write_text("\n".join(json.dumps(row) for row in (
        {
            "event": "request",
            "request_input_sha256": "request-hash",
            "logical_payload": {"model": "model", "messages": []},
        },
        {
            "event": "response",
            "payload": {"choices": [{"message": {"content": "expected"}}]},
        },
    )) + "\n", encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        response_text = "expected"

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            encoded = json.dumps({
                "choices": [{"message": {"content": self.response_text}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        output = tmp_path / "result.json"
        rows = run_frozen_prefix_probe(
            fixture,
            output,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            request_count=1,
            prefix_caching=True,
            require_exact=True,
        )
        assert rows[0]["exact_response"] is True
        assert rows[0]["expected_response_text_sha256"] == rows[0]["response_text_sha256"]

        Handler.response_text = "different"
        with pytest.raises(RuntimeError, match="diverged at request 1"):
            run_frozen_prefix_probe(
                fixture,
                output,
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                request_count=1,
                prefix_caching=True,
                require_exact=True,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_live_llamacpp_prefix_backtracks_after_rejected_agent_output() -> None:
    calls = []

    class Native:
        request_slot = 1

        @staticmethod
        def _request_json(path, body=None):
            calls.append((path, body))
            return {
                "content": "recovered",
                "tokens": [30, 31],
                "timings": {"cache_n": 2},
            }

    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=Native()), SimpleNamespace(),
        prefix_caching=True,
    )
    # The live slot contains a sampled assistant response (3, 4) that the
    # agent parser rejected. Its next logical prompt backtracks after token 2
    # and replaces that response with a format-error observation (9, 10).
    adapter._live_session_tokens["session"] = (1, 2, 3, 4)
    adapter._logical_prompt_tokens = lambda request: (1, 2, 9, 10)
    request = SimpleNamespace(
        session_id="session",
        resources=(SimpleNamespace(resource_id="history"),),
        resolved_max_new_tokens=64,
        openai_fields={"temperature": 0, "seed": 0},
    )

    result, slot = adapter._generate_from_live_slot(request, 1)

    assert slot == 1
    assert result.text == "recovered"
    assert calls == [(
        "/completion",
        {
            "prompt": [1, 2, 9, 10],
            "id_slot": 1,
            "n_predict": 64,
            "cache_prompt": True,
            "temperature": 0.0,
            "seed": 0,
            "return_tokens": True,
            "pra_pin_resource": True,
        },
    )]
    assert result.raw["pra"]["wire_tokens"] == 2


def test_llamacpp_resident_token_count_includes_native_prefix_and_excludes_unevaluated_last_token() -> None:
    raw = {
        "tokens_evaluated": 78,
        "tokens_predicted": 2384,
        "pra": {"native_tokens": 2263},
    }

    assert HybridLlamaCppAdapter._resident_token_count(raw) == 4724


def test_llamacpp_live_state_retains_exact_ids_and_leaves_last_sample_as_bridge() -> None:
    class Native:
        request_slot = 1

        @staticmethod
        def _query_text(request):
            return "prompt"

        @staticmethod
        def _request_json(path, body=None):
            assert path == "/tokenize"
            return {"tokens": [10, 11, 12]}

    adapter = HybridLlamaCppAdapter(
        SimpleNamespace(native_executor=Native()), SimpleNamespace(),
        prefix_caching=True,
    )
    request = SimpleNamespace(resources=(), session_id="session")
    result = SimpleNamespace(raw={"tokens": [20, 21, 22]})

    adapter._remember_resident_tokens(request, result)

    assert adapter._live_session_tokens["session"] == (10, 11, 12, 20, 21)


def test_frozen_selection_replays_exact_order_and_content() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "alpha beta gamma"},
            {"role": "user", "content": "alpha output"},
            {"role": "assistant", "content": "beta"},
            {"role": "user", "content": "find alpha"},
        ]
    }
    frozen = [
        ("m1-0-user", "task"),
        ("m2-0-assistant", "alpha beta gamma"),
        ("m3-0-user", "alpha output"),
    ]

    transformed, trace = transform_chat_payload(
        payload, mode=ContextTreatment.GATEWAY_NATIVE_PRA,
        budget_fraction=1.0, frozen_selection=frozen,
    )

    assert [
        (row["resource_id"], row["text"])
        for row in transformed["pra"]["resources"]
    ] == frozen
    assert trace.selected_segments == 3
    assert trace.selected_resource_digest


def test_selection_fixture_rejects_modified_content(tmp_path: Path) -> None:
    fixture = tmp_path / "selection.jsonl"
    fixture.write_text(json.dumps({
        "request_input_sha256": "request",
        "selected_resource_digest": "wrong",
        "resources": [{"resource_id": "record", "text": "content"}],
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content digest"):
        _load_selection_fixture(fixture)


def test_frontier_reports_matched_pra_vs_truncation_delta(tmp_path: Path) -> None:
    for condition, mode, resolved, tokens in (
        ("truncation_50", "truncation", False, 50),
        ("pra_selected_50", "gateway-pra", True, 45),
    ):
        directory = tmp_path / condition
        directory.mkdir()
        (directory / "results.jsonl").write_text(json.dumps({
            "instance_id": "task", "resolved": resolved, "mode": mode,
            "context_budget_fraction": 0.5, "physical_input_tokens": tokens,
        }) + "\n", encoding="utf-8")

    summary = summarize_frontier(tmp_path)

    assert summary["matched_budget"][0]["success_delta"] == 1.0
    assert summary["matched_budget"][0]["physical_input_token_delta"] == -5


def test_easy20_campaign_is_local_no_pra_calibration() -> None:
    campaign = CampaignConfig.load(EASY20_CONFIG)
    baseline = campaign.baselines[0]
    assert baseline.admission_kind == "local_calibration"
    assert baseline.published_score is None
    assert baseline.minimum_admission_score == 0.2
    assert len(baseline.task_ids) == 20
    assert len(campaign.cells) == 1
    assert campaign.cells[0].mode.value == "native"
    assert baseline.context_limit == 32768
    assert baseline.max_steps == 50
    assert campaign.cells[0].command[
        campaign.cells[0].command.index("--max-steps") + 1
    ] == "50"
    assert "--local-calibration" in campaign.cells[0].command


def test_easy20_official_receipt_records_admitted_nontrivial_baseline() -> None:
    result = OfficialResult.load(EASY20_RESULT)
    assert result.official_grader is True
    assert (result.resolved, result.total, result.score) == (9, 20, 0.45)
    review = review_result(
        CampaignConfig.load(EASY20_CONFIG).baselines[0], result,
        absolute_tolerance=0, require_exact_cohort=True,
    )
    assert review.status == ReproductionStatus.BASELINE_REPRODUCED


@pytest.mark.parametrize(
    ("score", "band"),
    [(0.09, "FLOOR"), (0.15, "MARGINAL"), (0.25, "USEFUL"),
     (0.50, "PREFERRED"), (0.75, "USEFUL"), (0.81, "SATURATED")],
)
def test_calibration_success_bands(score: float, band: str) -> None:
    assert _success_band(score) == band


def test_calibration_next_tier_follows_admission_policy() -> None:
    assert "correction" in _next_tier(0.15)
    assert "Easy-50" in _next_tier(0.50)
    assert "harder" in _next_tier(0.90)


def test_easy50_frontier_is_nested_gated_and_budget_matched() -> None:
    campaign = CampaignConfig.load(EASY50_CONFIG)
    assert len(campaign.baselines[0].task_ids) == 50
    assert len(campaign.cells) == 16
    baseline, *treatments = campaign.cells
    assert baseline.cell_id == "easy50-no-pra"
    assert baseline.evidence_role == "baseline_admission"
    assert all(cell.baseline_cell == baseline.cell_id for cell in treatments)
    assert all(cell.minimum_baseline_score == 0.2 for cell in treatments)
    gateway_passthrough = next(
        cell for cell in treatments if cell.mode.value == "gateway_passthrough"
    )
    gateway_treatments = [
        cell for cell in treatments
        if cell.connection == "gateway" and cell is not gateway_passthrough
    ]
    assert gateway_treatments
    assert all(
        gateway_passthrough.cell_id in cell.prerequisite_cells
        for cell in gateway_treatments
    )
    truncation = {
        cell.cell_id.rsplit("-", 1)[-1]: cell
        for cell in treatments if cell.mode.value == "truncation"
    }
    selected = {
        cell.cell_id.rsplit("-", 1)[-1]: cell
        for cell in treatments if cell.mode.value == "gateway_pra"
    }
    assert truncation.keys() == selected.keys() == {"50", "25", "5"}
    for key in truncation:
        truncation_budget = truncation[key].command[
            truncation[key].command.index("--budget-fraction") + 1
        ]
        selected_budget = selected[key].command[
            selected[key].command.index("--budget-fraction") + 1
        ]
        assert truncation_budget == selected_budget
    direct = next(
        cell for cell in treatments
        if cell.mode.value == "native_pra" and not cell.prefix_caching
    )
    direct_cached = next(
        cell for cell in treatments
        if cell.mode.value == "native_pra" and cell.prefix_caching
    )
    gateway = next(
        cell for cell in treatments
        if cell.mode.value == "gateway_native_pra" and cell.selection_contract == "route_owned"
    )
    equivalence = next(
        cell for cell in treatments
        if cell.mode.value == "gateway_native_pra"
        and cell.selection_contract == "frozen_replay"
        and not cell.prefix_caching
    )
    equivalence_cached = next(
        cell for cell in treatments
        if cell.mode.value == "gateway_native_pra"
        and cell.selection_contract == "frozen_replay"
        and cell.prefix_caching
    )
    assert direct.connection == "direct"
    assert gateway.connection == "gateway"
    assert direct.engine_pra_enabled is gateway.engine_pra_enabled is True
    assert direct.gateway_pra_enabled is False
    assert gateway.gateway_pra_enabled is True
    assert gateway.paired_cell == direct.cell_id
    assert gateway.comparison_group == direct.comparison_group == "native-pra-prefix-50"
    assert direct.evidence_role == "efficacy"
    assert gateway.evidence_role == "product_end_to_end"
    assert direct.selection_contract == gateway.selection_contract == "route_owned"
    assert equivalence.evidence_role == "transport_equivalence"
    assert equivalence.paired_cell == direct.cell_id
    assert "--selection-record" in direct.command
    assert "--selection-replay" in equivalence.command
    assert direct_cached.evidence_role == "prefix_cache_effect"
    assert direct_cached.paired_cell == direct.cell_id
    assert direct_cached.factorial_group == direct.factorial_group
    assert "--prefix-caching" in direct_cached.command
    assert equivalence_cached.paired_cell == direct_cached.cell_id
    assert equivalence_cached.factorial_group == direct.factorial_group
    assert "--prefix-caching" in equivalence_cached.command
    engine_controls = [
        cell for cell in treatments if cell.mode.value == "engine_control"
    ]
    assert len(engine_controls) == 2
    assert {cell.prefix_caching for cell in engine_controls} == {False, True}
    assert {cell.engine_target_id for cell in engine_controls + [direct, direct_cached]} == {
        "qwen3-coder-30b-q4-k-m-llamacpp-pra-v1"
    }
    headroom = next(cell for cell in treatments if cell.mode.value == "headroom")
    assert headroom.gateway_mode == "HEADROOM"
    assert headroom.engine_pra_enabled is headroom.gateway_pra_enabled is False
    assert "${PRA_AGENT_HEADROOM_URL}" in headroom.command
    assert direct_cached.cell_id in headroom.prerequisite_cells
    stage_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    declared_order = {cell.cell_id: index for index, cell in enumerate(campaign.cells)}

    def schedule_key(cell_id: str) -> tuple[int, int, int]:
        cell = next(row for row in campaign.cells if row.cell_id == cell_id)
        return stage_order[cell.stage], cell.priority, declared_order[cell_id]

    text_frontier = [
        "easy50-gateway-passthrough",
        "easy50-truncation-50",
        "easy50-pra-selected-50",
        "easy50-truncation-25",
        "easy50-pra-selected-25",
    ]
    assert sorted(text_frontier, key=schedule_key) == text_frontier
    assert all(
        schedule_key("easy50-pra-selected-25") < schedule_key(cell.cell_id)
        for cell in (direct, equivalence, gateway)
    )
    for cell_id in (
        "easy50-gateway-passthrough",
        "easy50-pra-selected-50",
        "easy50-pra-selected-25",
    ):
        command = next(cell.command for cell in campaign.cells if cell.cell_id == cell_id)
        assert "http://127.0.0.1:8080/v1" not in command
        assert "http://127.0.0.1:8081/v1" not in command
    assert "${PRA_AGENT_G00_URL}" in gateway_passthrough.command
    assert all(
        "${PRA_AGENT_G10_URL}" in cell.command for cell in selected.values()
    )
    assert not truncation["50"].enabled and selected["50"].enabled
    assert all(not truncation[key].enabled and not selected[key].enabled
               for key in ("25", "5"))


def test_fixed50_campaign_hydrates_ids_and_keeps_treatments_locked() -> None:
    campaign = CampaignConfig.load(SWEBENCH_CONFIG)
    assert [baseline.published_resolved for baseline in campaign.baselines] == [7, 19]
    assert all(len(baseline.task_ids) == 50 for baseline in campaign.baselines)
    assert all(not cell.enabled for cell in campaign.cells)
    treatments = [cell for cell in campaign.cells if cell.baseline_cell]
    assert treatments
    assert all(cell.baseline_cell == "gemma4-31b-no-pra" for cell in treatments)
    assert all(cell.minimum_baseline_score == 0.20 for cell in treatments)
    assert all("raise SystemExit" not in " ".join(cell.command) for cell in treatments)


def test_fixed50_campaign_dry_run_cannot_admit_pra(tmp_path: Path) -> None:
    payload = yaml.safe_load(SWEBENCH_CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "swebench")
    for cell in payload["cells"]:
        cell["enabled"] = True
    config = ROOT / "experiments/paper4_5_agent/configs/campaigns/.tmp_swebench_test.yaml"
    try:
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        state = run_campaign(config, max_hours=1, resume=False, dry_run=True)
    finally:
        config.unlink(missing_ok=True)
    assert state["cells"]["qwen3-coder-30b-no-pra"]["state"] == "PENDING"
    assert state["cells"]["gemma4-31b-no-pra"]["state"] == "PENDING"
    assert state["cells"]["gemma4-pra-50"]["state"] == "BLOCKED"


def _fixed50_execution_identity(baseline: object) -> dict[str, object]:
    return {
        "cohort_sha256": baseline.task_ids_sha256,
        "benchmark_revision": baseline.benchmark_revision,
        "harness": baseline.harness,
        "harness_version": baseline.harness_version,
        "model": baseline.model,
        "engine": baseline.engine,
        "engine_version": baseline.engine_version,
        "dtype": baseline.dtype,
        "quantization": baseline.quantization,
        "kv_cache_dtype": baseline.kv_cache_dtype,
        "scaffold": baseline.scaffold,
        "context_limit": baseline.context_limit,
        "max_steps": baseline.max_steps,
        "temperature": baseline.temperature,
        "function_calling": baseline.function_calling,
        "prefix_caching": baseline.prefix_caching,
        "grading": baseline.grading,
    }


def test_baseline_score_floor_blocks_weak_but_reproduced_cell(tmp_path: Path) -> None:
    payload = yaml.safe_load(SWEBENCH_CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "swebench")
    treatment = payload["cells"][2]
    treatment["baseline_id"] = "qwen3-coder-30b-fixed50"
    treatment["baseline_cell"] = "qwen3-coder-30b-no-pra"
    payload["cells"] = [payload["cells"][0], treatment]
    config = ROOT / "experiments/paper4_5_agent/configs/campaigns/.tmp_swebench_gate.yaml"
    try:
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        loaded = CampaignConfig.load(config)
        baseline_ids = list(loaded.baselines[0].task_ids)
        result = tmp_path / "weak.json"
        result.write_text(json.dumps({
            "official_grader": True, "score": 0.14, "resolved": 7, "total": 50,
            "task_ids": baseline_ids, "configuration_differences": [],
            "execution_identity": _fixed50_execution_identity(loaded.baselines[0]),
        }), encoding="utf-8")
        record_result(config, cell_id="qwen3-coder-30b-no-pra", result_path=result)
        with pytest.raises(ValueError, match="baseline score >= 0.200"):
            record_result(config, cell_id="gemma4-gateway-passthrough", result_path=result)
    finally:
        config.unlink(missing_ok=True)


def test_fixed50_score_without_execution_identity_does_not_unlock() -> None:
    campaign = CampaignConfig.load(SWEBENCH_CONFIG)
    baseline = campaign.baselines[1]
    result = OfficialResult(
        official_grader=True, score=0.38, resolved=19, total=50,
        task_ids=baseline.task_ids,
    )
    review = review_result(
        baseline, result, absolute_tolerance=0.10, require_exact_cohort=True,
    )
    assert review.status == ReproductionStatus.BASELINE_ATTEMPTED
    assert any("structured execution identity" in reason for reason in review.reasons)


def test_local_calibration_admits_only_official_identity_matched_useful_score() -> None:
    payload = yaml.safe_load(SWEBENCH_CONFIG.read_text(encoding="utf-8"))["baselines"][0]
    payload.update(
        admission_kind="local_calibration",
        published_score=None,
        minimum_admission_score=0.2,
        maximum_admission_score=0.8,
    )
    baseline = PublishedBaseline.model_validate(payload)
    identity = _fixed50_execution_identity(baseline)
    result = OfficialResult(
        official_grader=True,
        score=0.4,
        resolved=20,
        total=50,
        task_ids=baseline.task_ids,
        execution_identity=identity,
    )
    review = review_result(
        baseline, result, absolute_tolerance=0.0, require_exact_cohort=True
    )
    assert review.status == ReproductionStatus.BASELINE_REPRODUCED
    assert review.published_score is None


def test_swebench_chunk_report_requires_exact_ids(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "submitted_ids": ["a", "b"], "resolved_ids": ["b"], "error_ids": [],
    }), encoding="utf-8")
    assert _normalize_report(report, ["a", "b"])["resolved_ids"] == ["b"]
    with pytest.raises(RuntimeError, match="frozen chunk"):
        _normalize_report(report, ["a", "c"])


def test_swebench_completed_chunks_reach_final_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index, instance_id in enumerate(("repo__project-1", "repo__project-2")):
        chunk = tmp_path / f"chunk_{index:02d}"
        chunk.mkdir()
        (chunk / "official_chunk_result.json").write_text(json.dumps({
            "submitted_ids": [instance_id],
            "resolved_ids": [instance_id] if index == 0 else [],
            "error_ids": [],
        }), encoding="utf-8")
    monkeypatch.setattr(
        "experiments.paper4_5_agent.runners.swebench_verified._write_task_rows",
        lambda *args, **kwargs: None,
    )
    args = SimpleNamespace(
        chunk_size=1, model="model", model_revision="model-revision",
        tokenizer_revision="tokenizer-revision", engine="ollama",
        engine_version="1", dtype="mixed", quantization="Q4_K_M",
        kv_cache_dtype="f16", scaffold="swebench_backticks.yaml",
        context_limit=32768, max_steps=50, grading="official",
        benchmark_revision="revision", harness_version="2.4.6",
    )
    card = {
        "instance_ids": ["repo__project-1", "repo__project-2"],
        "canonical_ids_sha256": "digest", "source_revision": "revision",
    }

    result_path = _execute_chunks(
        args, card, tmp_path, "http://unused", {"configuration_differences": []},
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["score"] == 0.5
    assert result["resolved"] == 1
    assert result["timeouts"] == 0
    assert result["configuration_differences"] == []


def test_treatment_resume_rejects_legacy_or_mismatched_chunk_receipts() -> None:
    treatment = SimpleNamespace(mode="gateway-passthrough")
    receipt = {"execution_fingerprint": "current"}
    assert not _chunk_receipt_reusable(treatment, {}, receipt)
    assert not _chunk_receipt_reusable(
        treatment, {"execution_fingerprint": "old"}, receipt,
    )
    assert _chunk_receipt_reusable(
        treatment, {"execution_fingerprint": "current"}, receipt,
    )
    assert _chunk_receipt_reusable(SimpleNamespace(mode="no-pra"), {}, receipt)


def test_timed_out_agent_becomes_an_empty_official_prediction(tmp_path: Path) -> None:
    destination = tmp_path / "preds.json"
    _write_empty_predictions(destination, ["repo__project-1"], "model")
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "repo__project-1": {
            "model_name_or_path": "openai/model",
            "instance_id": "repo__project-1",
            "model_patch": "",
        }
    }


def test_swebench_images_are_pulled_per_task_with_explicit_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], log: Path, timeout: int, **kwargs: object) -> float:
        commands.append(command)
        assert log.parent == tmp_path
        assert timeout == 900
        docker_config = Path(str(kwargs["extra_environment"]["DOCKER_CONFIG"]))
        assert docker_config == tmp_path / ".docker-anonymous"
        return 1.25

    monkeypatch.setattr(
        "experiments.paper4_5_agent.runners.swebench_verified._run", fake_run,
    )
    args = SimpleNamespace(
        docker_platform="linux/amd64", image_pull_timeout_seconds=900,
    )

    _prepull_swebench_images(args, ["django__django-13297"], tmp_path, 3)

    image = "docker.io/swebench/sweb.eval.x86_64.django_1776_django-13297:latest"
    assert commands == [["docker", "pull", "--platform", "linux/amd64", image]]
    receipt = json.loads(
        (tmp_path / "chunk_03.image_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt == {
        "schema_version": 1,
        "images": [{
            "instance_id": "django__django-13297",
            "image": image,
            "platform": "linux/amd64",
            "wall_time_s": 1.25,
        }],
        "registry_auth": "anonymous_public_pull",
        "excluded_from_model_and_grader_wall_time": True,
    }


def test_timeout_cleanup_targets_only_emitted_owned_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = (
        "Started container minisweagent-a1b2 with ID x\n"
        "container sweb.eval.repo__task.run-1 completed\n"
        "unrelated production-container must remain"
    )

    cleaned = _cleanup_owned_containers(output)

    assert cleaned == ["minisweagent-a1b2", "sweb.eval.repo__task.run-1"]
    assert commands == [
        ["docker", "rm", "-f", "minisweagent-a1b2"],
        ["docker", "rm", "-f", "sweb.eval.repo__task.run-1"],
    ]


def test_agent_execution_failure_is_not_admitted_as_empty_patch(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk_00"
    chunk.mkdir()
    (chunk / "minisweagent.log").write_text(
        "ERROR - Error processing instance django__django-15277: docker exit 125\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="refusing to grade"):
        _raise_on_agent_infrastructure_error(chunk, ["django__django-15277"])

    receipt = json.loads(
        (chunk / "infrastructure_failure.json").read_text(encoding="utf-8")
    )
    assert receipt["instance_ids"] == ["django__django-15277"]
    assert receipt["admitted_as_benchmark_result"] is False


def test_agent_execution_failure_gate_ignores_clean_log(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk_00"
    chunk.mkdir()
    (chunk / "minisweagent.log").write_text(
        "INFO - Instance django__django-15277 completed\n", encoding="utf-8"
    )

    _raise_on_agent_infrastructure_error(chunk, ["django__django-15277"])
    assert not (chunk / "infrastructure_failure.json").exists()


def test_swebench_patch_apply_failure_is_not_a_generic_grader_error(tmp_path: Path) -> None:
    log = tmp_path / "grader.log"
    log.write_text(
        "sympy__sympy-21847: >>>>> Patch Apply Failed: only garbage found",
        encoding="utf-8",
    )
    assert _grader_error_type(log, "sympy__sympy-21847") == "patch_apply_failed"
    assert _grader_error_type(log, "another__task-1") == "official_grader_error"


def test_swebench_agent_and_grader_share_locked_container_platform(
    tmp_path: Path,
) -> None:
    environment = _container_environment(
        SimpleNamespace(docker_platform="linux/amd64"), tmp_path,
    )

    assert environment == {
        "HF_DATASETS_CACHE": str(tmp_path / "hf_datasets_cache"),
        "DOCKER_DEFAULT_PLATFORM": "linux/amd64",
    }


def test_swebench_package_probe_uses_null_for_missing_distributions() -> None:
    versions = package_versions()
    assert set(versions) == {"mini-swe-agent", "swebench", "vllm"}
    assert all(value is None or isinstance(value, str) for value in versions.values())


def test_official_result_rejects_inconsistent_score_and_ids() -> None:
    with pytest.raises(ValueError, match="score must equal"):
        OfficialResult(official_grader=True, score=0.5, resolved=1, total=3)
    with pytest.raises(ValueError, match="task_ids length"):
        OfficialResult(
            official_grader=True, score=0.5, resolved=1, total=2,
            task_ids=("only-one",),
        )
    with pytest.raises(ValueError, match="timeout count"):
        OfficialResult(
            official_grader=True, score=0.0, resolved=0, total=2, timeouts=3,
        )


def test_h100_preflight_accepts_nvidia_smi_mib_format() -> None:
    assert _is_h100_80gb("NVIDIA H100 80GB HBM3, 81559 MiB")
    assert not _is_h100_80gb("NVIDIA H100 PCIe, 61440 MiB")
    assert not _is_h100_80gb("NVIDIA RTX 4090, 24564 MiB")


def test_gateway_preflight_requires_mode_and_pinned_model() -> None:
    class Handler(BaseHTTPRequestHandler):
        model_ids = ["qwen3-coder:30b"]
        gateway_mode = "G00"

        def do_GET(self) -> None:  # noqa: N802
            payload = (
                {"status": "ok", "gateway_mode": self.gateway_mode, "protocol_version": "1"}
                if self.path == "/health"
                else {"data": [{"id": model_id} for model_id in self.model_ids]}
            )
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            assert body["model"] == "qwen3-coder:30b"
            assert body["temperature"] == 0
            response = {
                "choices": [{"message": {"role": "assistant", "content": "OK"}}]
            }
            if self.gateway_mode == "G10":
                assert body["pra"]["resources"][0]["resource_id"] == "preflight-resource"
                response["pra"] = {"selected_resource_ids": ["preflight-resource"]}
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = SimpleNamespace(
        mode="gateway-passthrough",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        served_model="qwen3-coder:30b",
    )
    try:
        result = gateway_preflight(args)
        assert result["gateway_mode"] == "G00"
        assert result["generation_probe"] == "passed"
        Handler.model_ids = []
        with pytest.raises(RuntimeError, match="must pin and advertise"):
            gateway_preflight(args)
        Handler.model_ids = ["qwen3-coder:30b"]
        Handler.gateway_mode = "G10"
        args.mode = "gateway-pra"
        result = gateway_preflight(args)
        assert result["selected_context_probe"] == "passed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_native_preflight_requires_consumption_and_active_prefix_cache(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        prefix_enabled = True
        agent_qualified = True
        post_count = 0
        session_ids = []

        def do_GET(self) -> None:  # noqa: N802
            payload = (
                {
                    "status": "ok",
                    "prefix_cache_enabled": self.prefix_enabled,
                    "effective_capabilities": {
                        "native_kv": True,
                        "agent_history_kv_qualified": self.agent_qualified,
                        "session_state": True,
                        "live_prefix_kv_capture": True,
                        "zero_selected_text_reencoding": True,
                        "explicit_prefix_cache": True,
                        "prefix_cache_mode": "explicit_prefix_handle",
                    },
                }
                if self.path == "/health"
                else {"data": [{"id": "qwen3-coder:30b"}]}
            )
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802
            Handler.post_count += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["pra"]["required_capabilities"] == ["logical_refs", "native_kv"]
            assert body["pra"]["metadata"]["ephemeral_session"] is True
            Handler.session_ids.append(body["pra"]["session_id"])
            encoded = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "pra": {"native_kv": True},
                "pra_trace": [{"stage": "llama_cpp_native_attach", "native_tokens": 4}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = SimpleNamespace(
        mode="direct-native-pra",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        served_model="qwen3-coder:30b",
        prefix_caching=True,
    )
    try:
        result = gateway_preflight(args)
        repeated = gateway_preflight(args)
        assert result["native_consumption_probe"] == "passed"
        assert repeated["native_consumption_probe"] == "passed"
        assert len(set(Handler.session_ids)) == 2
        assert result["prefix_cache_enabled"] is True
        receipt = tmp_path / "preflight.json"
        receipt.write_text(json.dumps({"gateway_preflight": result}), encoding="utf-8")
        args.endpoint_preflight_receipt = receipt
        replayed = gateway_preflight(args)
        assert replayed["generation_probe_replayed"] is True
        assert Handler.post_count == 2
        args.endpoint_preflight_receipt = None
        Handler.prefix_enabled = False
        with pytest.raises(RuntimeError, match="prefix_cache_enabled=true"):
            gateway_preflight(args)
        args.prefix_caching = False
        result = gateway_preflight(args)
        assert result["prefix_cache_enabled"] is False
        Handler.prefix_enabled = True
        with pytest.raises(RuntimeError, match="prefix_cache_enabled=false"):
            gateway_preflight(args)
        Handler.prefix_enabled = False
        Handler.agent_qualified = False
        args.budget_fraction = 1.0
        result = gateway_preflight(args)
        assert result["native_consumption_probe"] == "passed"
        args.budget_fraction = 0.9
        with pytest.raises(RuntimeError, match="sparse native agent-history"):
            gateway_preflight(args)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_treatment_proxy_is_preflighted_after_start(tmp_path: Path) -> None:
    captured_payloads = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            encoded = json.dumps({"data": [{"id": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802
            captured_payloads.append(json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            ))
            encoded = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "OK"}}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    proxy = TreatmentProxy(
        f"http://127.0.0.1:{target.server_port}/v1",
        mode=ContextTreatment.PASSTHROUGH,
        budget_fraction=1.0,
        trace_path=tmp_path / "trace.jsonl",
    )
    proxy_url = proxy.start()
    args = SimpleNamespace(
        mode="gateway-passthrough",
        base_url="http://raw-engine.invalid/v1",
        served_model="model",
        chat_template_no_thinking=False,
        prefix_caching=False,
    )
    try:
        result = gateway_preflight(args, base_url=proxy_url)
        assert result["gateway_mode"] == "G00"
        assert result["generation_probe"] == "passed"
        assert "pra" not in captured_payloads[0]
    finally:
        proxy.close()
        target.shutdown()
        target.server_close()
        target_thread.join(timeout=5)


def test_campaign_command_expansion_can_fail_closed_on_missing_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRA_TEST_REQUIRED_URL", raising=False)
    command = ("runner", "--base-url", "${PRA_TEST_REQUIRED_URL}")
    assert _expand_command(command) == list(command)
    with pytest.raises(ValueError, match="PRA_TEST_REQUIRED_URL"):
        _expand_command(command, require_resolved=True)


def test_context_treatments_share_budget_and_keep_mandatory_messages() -> None:
    payload = {
        "model": "model",
        "messages": [
            {"role": "system", "content": "work carefully"},
            {"role": "user", "content": "repository task statement"},
            {"role": "assistant", "content": "I inspected alpha module ordinary details"},
            {"role": "user", "content": "tool output needle_value appears in alpha.py"},
            {"role": "assistant", "content": "more unrelated trajectory words here"},
            {"role": "user", "content": "where is needle_value defined"},
        ],
    }
    truncated, truncation = transform_chat_payload(
        payload, mode=ContextTreatment.TRUNCATION, budget_fraction=1.0,
    )
    selected, pra = transform_chat_payload(
        payload, mode=ContextTreatment.PRA_SELECTED_CONTEXT, budget_fraction=1.0,
    )
    assert truncated["messages"][0] == payload["messages"][0]
    assert truncated["messages"][-1] == payload["messages"][-1]
    assert selected["messages"] == [
        payload["messages"][0], payload["messages"][-2], payload["messages"][-1],
    ]
    assert selected["pra"]["resources"][0]["resource_id"] == "m1-0-user"
    assert selected["pra"]["resources"][0]["text"] == payload["messages"][1]["content"]
    assert selected["pra"]["metadata"]["pinned_task_segments"] == ["m1-0-user"]
    assert any("needle_value" in row["text"] for row in selected["pra"]["resources"])
    assert truncation.logical_input_tokens_estimate == pra.logical_input_tokens_estimate
    assert truncation.session_id == pra.session_id == selected["pra"]["session_id"]
    assert truncation.mandatory_tokens_estimate == pra.mandatory_tokens_estimate
    assert truncation.physical_input_tokens_estimate <= truncation.logical_input_tokens_estimate
    assert pra.physical_input_tokens_estimate <= pra.logical_input_tokens_estimate


def test_passthrough_does_not_rewrite_openai_payload() -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hello world"}]}
    transformed, trace = transform_chat_payload(
        payload, mode=ContextTreatment.PASSTHROUGH, budget_fraction=1.0,
    )
    assert transformed == payload
    assert trace.tokens_avoided_estimate == 0
    assert trace.selected_tokens_estimate == 0


def test_proxy_request_retention_metadata_overrides_process_defaults(
    tmp_path: Path,
) -> None:
    captured_payloads = []

    class Target(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            captured_payloads.append(json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            ))
            encoded = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "OK"}}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    thread = threading.Thread(target=target.serve_forever, daemon=True)
    thread.start()
    proxy = TreatmentProxy(
        f"http://127.0.0.1:{target.server_port}/v1",
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.5,
        trace_path=tmp_path / "trace.jsonl",
        recent_completed_turns=4,
        recent_records_per_turn=4,
        recent_progress_turns=3,
        max_records_per_turn_before_chunking=8,
    )
    proxy_url = proxy.start()
    request_policy = {
        "recent_completed_turns": 1,
        "recent_records_per_turn": 1,
        "recent_progress_turns": 0,
        "large_record_chunk_tokens": 3,
        "max_records_per_turn_before_chunking": 4,
    }
    payload = {
        "model": "model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "old action"},
            {"role": "tool", "content": "old result"},
            {"role": "assistant", "content": "active action"},
            {"role": "tool", "content": "active result"},
        ],
        "pra": {"metadata": {"retention_policy": request_policy}},
    }
    try:
        request = urllib.request.Request(
            f"{proxy_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
    finally:
        proxy.close()
        target.shutdown()
        target.server_close()
        thread.join(timeout=5)

    effective = captured_payloads[0]["pra"]["metadata"]["retention_policy"]
    assert effective["recent_completed_turns"] == 1
    assert effective["recent_records_per_turn"] == 1
    assert effective["recent_progress_turns"] == 0
    assert effective["large_record_chunk_tokens"] == 3
    assert effective["max_records_per_turn_before_chunking"] == 4
    # Request fields omitted by the caller inherit process policy defaults.
    assert effective["recent_source_turns"] == 1
    assert effective["preserve_action_observation_pairs"] is True


def test_headroom_mode_leaves_compression_to_the_pinned_external_proxy() -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hello world"}]}
    transformed, trace = transform_chat_payload(
        payload, mode=ContextTreatment.HEADROOM, budget_fraction=1.0,
    )
    assert transformed == payload
    assert trace.mode == "headroom"
    assert trace.tokens_avoided_estimate == 0
    assert trace.selected_tokens_estimate == 0


def test_stratified_outcomes_does_not_hide_regressions_behind_net_score() -> None:
    baseline = {"submitted_ids": ["a", "b", "c", "d"], "resolved_ids": ["a", "b"]}
    treatment = {"submitted_ids": ["a", "b", "c", "d"], "resolved_ids": ["a", "c", "d"]}
    result = stratified_outcomes(baseline, treatment)
    assert result["retained_count"] == 1
    assert result["regressed_count"] == 1
    assert result["acquired_count"] == 2
    assert result["net_solve_delta"] == 1


def test_minisweagent_trajectory_metrics_preserve_exact_usage(tmp_path: Path) -> None:
    path = tmp_path / "repo__project-1.traj.json"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task statement"},
        {
            "role": "assistant", "content": "inspect",
            "extra": {
                "timestamp": 10.0, "actions": [{"command": "cat file"}],
                "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
            },
        },
        {"role": "user", "content": "output", "extra": {"timestamp": 11.5}},
        {
            "role": "assistant", "content": "finish",
            "extra": {
                "timestamp": 15.0, "actions": [{"command": "patch"}],
                "response": {"usage": {"prompt_tokens": 180, "completion_tokens": 30}},
            },
        },
        {"role": "exit", "content": "done", "extra": {"timestamp": 16.0}},
    ]
    path.write_text(json.dumps({
        "messages": messages,
        "info": {"exit_status": "Submitted", "submission": "diff\n+line\n"},
    }), encoding="utf-8")

    metrics = _trajectory_metrics(path)

    assert metrics["cumulative_prompt_tokens"] == 280
    assert metrics["max_prompt_tokens"] == 180
    assert metrics["repeated_context_tokens_estimate"] == 100
    assert metrics["repeated_context_fraction_estimate"] == pytest.approx(100 / 280)
    assert metrics["output_tokens"] == 50
    assert metrics["model_call_count"] == metrics["tool_call_count"] == 2
    assert metrics["wall_time_s"] == 6.0
    assert metrics["tool_time_s"] == 1.5
    assert metrics["termination_reason"] == "Submitted"
    assert metrics["patch"] == "diff\n+line\n"


def test_treatment_trace_aggregation_keeps_estimates_disjoint() -> None:
    rows = [
        {
            "logical_input_tokens_estimate": 100,
            "physical_input_tokens_estimate": 60,
            "selected_tokens_estimate": 30,
            "route_time_s": 0.1,
            "token_estimator": "whitespace_v1",
            "prefix_cache_observed": True,
            "prefix_cached_tokens": 40,
            "native_tokens": 30,
            "wire_tokens": 12,
            "physical_kv_copy": False,
            "physical_kv_copy_bytes": 0,
            "total_kv_copy_bytes": 128,
            "canonical_suffix_graft_d2d_bytes": 128,
            "host_to_device_bytes": 0,
            "selected_kv_tokens": 30,
            "selected_text_reencoded_tokens": 0,
            "selected_history_reencoded_tokens": 0,
            "realized_retention_fraction": 0.6,
            "consumer_temporary_bytes": 1200,
            "consumer_temporary_peak_bytes": 700,
            "fused_attention_calls": 4,
            "full_retention": True,
            "resource_update_mode": "cold_prefill",
            "resource_prefix_cached_tokens": 0,
            "resource_evaluated_tokens": 30,
            "resource_total_tokens": 30,
        },
        {
            "logical_input_tokens_estimate": 200,
            "physical_input_tokens_estimate": 100,
            "selected_tokens_estimate": 50,
            "route_time_s": 0.2,
            "token_estimator": "whitespace_v1",
            "prefix_cache_observed": True,
            "prefix_cached_tokens": 80,
            "native_tokens": 50,
            "wire_tokens": 20,
            "physical_kv_copy": False,
            "physical_kv_copy_bytes": 64,
            "total_kv_copy_bytes": 320,
            "canonical_suffix_graft_d2d_bytes": 256,
            "host_to_device_bytes": 32,
            "selected_kv_tokens": 50,
            "selected_text_reencoded_tokens": 9,
            "selected_history_reencoded_tokens": 9,
            "realized_retention_fraction": 0.5,
            "consumer_temporary_bytes": 1800,
            "consumer_temporary_peak_bytes": 900,
            "fused_attention_calls": 6,
            "full_retention": False,
            "resource_update_mode": "prefix_delta",
            "resource_prefix_cached_tokens": 40,
            "resource_evaluated_tokens": 10,
            "resource_total_tokens": 50,
        },
    ]

    aggregate = _aggregate_traces(rows)

    assert aggregate["logical_input_tokens_estimate"] == 300
    assert aggregate["physical_input_tokens_estimate"] == 160
    assert aggregate["selected_tokens_estimate"] == 80
    assert aggregate["tokens_avoided_estimate"] == 140
    assert aggregate["token_saving_fraction_estimate"] == pytest.approx(140 / 300)
    assert aggregate["route_time_s"] == pytest.approx(0.3)
    assert aggregate["prefix_cached_tokens"] == 120
    assert aggregate["prefix_cache_observed_requests"] == 2
    assert aggregate["native_tokens"] == 80
    assert aggregate["wire_tokens"] == 32
    assert aggregate["physical_kv_copy_observed"] is False
    assert aggregate["physical_kv_copy_bytes"] == 64
    assert aggregate["total_kv_copy_bytes"] == 448
    assert aggregate["canonical_suffix_graft_d2d_bytes"] == 384
    assert aggregate["host_to_device_bytes"] == 32
    assert aggregate["selected_kv_tokens"] == 80
    assert aggregate["selected_text_reencoded_tokens"] == 9
    assert aggregate["selected_history_reencoded_tokens"] == 9
    assert aggregate["realized_retention_fraction"] == pytest.approx(160 / 300)
    assert aggregate["realized_retention_fraction_min"] == pytest.approx(0.5)
    assert aggregate["realized_retention_fraction_max"] == pytest.approx(0.6)
    assert aggregate["engine_reported_history_kv_retention_fraction_min"] == pytest.approx(0.5)
    assert aggregate["engine_reported_history_kv_retention_fraction_max"] == pytest.approx(0.6)
    assert aggregate["consumer_temporary_bytes"] == 3000
    assert aggregate["consumer_temporary_peak_bytes"] == 900
    assert aggregate["fused_attention_calls"] == 10
    assert aggregate["full_retention_requests"] == 1
    assert aggregate["sparse_kv_requests"] == 1
    assert aggregate["resource_update_counts"] == {
        "cold_prefill": 1, "prefix_delta": 1,
    }
    assert aggregate["resource_prefix_cached_tokens"] == 40
    assert aggregate["resource_evaluated_tokens"] == 40
    assert aggregate["resource_total_tokens"] == 80


def test_treatment_trace_aggregation_preserves_missing_physical_telemetry() -> None:
    aggregate = _aggregate_traces([{
        "logical_input_tokens_estimate": 100,
        "physical_input_tokens_estimate": 100,
    }])

    assert aggregate["selected_history_reencoded_tokens"] is None
    assert aggregate["physical_kv_copy_bytes"] is None
    assert aggregate["total_kv_copy_bytes"] is None
    assert aggregate["canonical_suffix_graft_d2d_bytes"] is None
    assert aggregate["host_to_device_bytes"] is None
    assert aggregate["consumer_temporary_bytes"] is None
    assert aggregate["consumer_temporary_peak_bytes"] is None


def test_treatment_proxy_forwards_selected_context_and_writes_trace(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class Target(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            observed.update(json.loads(self.rfile.read(length)))
            body = json.dumps({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens_details": {"cached_tokens": 7}},
                "pra": {
                    "native_kv": True,
                    "native_tokens": 11,
                    "wire_tokens": 3,
                    "physical_kv_copy": False,
                    "physical_kv_copy_bytes": 0,
                    "total_kv_copy_bytes": 4096,
                    "canonical_suffix_graft_d2d_bytes": 4096,
                    "host_to_device_bytes": 0,
                    "selected_kv_tokens": 11,
                    "selected_text_reencoded_tokens": 0,
                    "selected_history_reencoded_tokens": 0,
                    "realized_retention_fraction": 0.9,
                    "consumer_temporary_bytes": 2048,
                    "consumer_temporary_peak_bytes": 1024,
                    "fused_attention_calls": 28,
                    "full_retention": False,
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    trace_path = tmp_path / "request_telemetry.jsonl"
    selection_path = tmp_path / "selection_fixture.jsonl"
    interaction_path = tmp_path / "interaction_history.jsonl"
    proxy = TreatmentProxy(
        f"http://127.0.0.1:{target.server_port}/v1",
        mode=ContextTreatment.DIRECT_NATIVE_PRA,
        budget_fraction=0.5,
        trace_path=trace_path,
        selection_record_path=selection_path,
        interaction_trace_path=interaction_path,
        request_overrides={"prefix_caching": True},
    )
    proxy_url = proxy.start()
    try:
        payload = json.dumps({
            "model": "model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task alpha"},
                {"role": "assistant", "content": "alpha evidence details"},
                {"role": "user", "content": "find alpha"},
            ],
        }).encode()
        request = urllib.request.Request(
            f"{proxy_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read())["choices"][0]["message"]["content"] == "ok"
    finally:
        proxy.close()
        target.shutdown()
        target.server_close()
        target_thread.join(timeout=5)
    assert observed["pra"]["metadata"]["benchmark_fairness"] == "agent-visible-messages-only"
    assert observed["prefix_caching"] is True
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["mode"] == "direct-native-pra"
    assert trace["physical_input_tokens_estimate"] <= trace["logical_input_tokens_estimate"]
    assert trace["prefix_cache_observed"] is True
    assert trace["prefix_cached_tokens"] == 7
    assert trace["native_tokens"] == 11
    assert trace["wire_tokens"] == 3
    assert trace["physical_kv_copy"] is False
    assert trace["physical_kv_copy_bytes"] == 0
    assert trace["total_kv_copy_bytes"] == 4096
    assert trace["canonical_suffix_graft_d2d_bytes"] == 4096
    assert trace["host_to_device_bytes"] == 0
    assert trace["selected_kv_tokens"] == 11
    assert trace["selected_text_reencoded_tokens"] == 0
    assert trace["selected_history_reencoded_tokens"] == 0
    assert trace["realized_retention_fraction"] == pytest.approx(0.9)
    assert trace["consumer_temporary_bytes"] == 2048
    assert trace["consumer_temporary_peak_bytes"] == 1024
    assert trace["fused_attention_calls"] == 28
    assert trace["full_retention"] is False
    fixture = _load_selection_fixture(selection_path)
    assert len(fixture) == 1
    assert next(iter(fixture.values()))
    interactions = [
        json.loads(line) for line in interaction_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in interactions] == ["request", "response"]
    assert interactions[0]["logical_payload"]["messages"][1]["content"] == "task alpha"
    assert interactions[0]["physical_payload"]["pra"]["resources"]
    assert interactions[1]["payload"]["choices"][0]["message"]["content"] == "ok"


def test_treatment_proxy_enforces_and_traces_agent_boundary(tmp_path: Path) -> None:
    class Target(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": (
                        "THOUGHT: explore\n```mswea_bash_command\n"
                        "sed -n '1,20p' source.py\n```"
                    ),
                }}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    interaction_path = tmp_path / "interaction_history.jsonl"
    proxy = TreatmentProxy(
        f"http://127.0.0.1:{target.server_port}/v1",
        mode=ContextTreatment.PASSTHROUGH,
        budget_fraction=1.0,
        trace_path=tmp_path / "trace.jsonl",
        interaction_trace_path=interaction_path,
        consumption_policy="verification-enforced-v1",
    )
    proxy_url = proxy.start()
    try:
        payload = json.dumps({"messages": _history([_MUTATION_STEP])}).encode()
        request = urllib.request.Request(
            f"{proxy_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            delivered = json.loads(response.read())
    finally:
        proxy.close()
        target.shutdown()
        target.server_close()
        target_thread.join(timeout=5)

    assert "git diff" in delivered["choices"][0]["message"]["content"]
    assert delivered["pra"]["agent"]["action_enforced"] is True
    interactions = [
        json.loads(line) for line in interaction_path.read_text(encoding="utf-8").splitlines()
    ]
    response_event = interactions[-1]
    assert response_event["pra_agent_enforcement"]["enforcement_reason"] == (
        "diff_required_after_mutation"
    )
    assert "sed -n '1,20p' source.py" in (
        response_event["pra_agent_enforcement"]["original_content"]
    )
    assert "original_content" not in delivered["pra"]["agent"]
    assert response_event["payload"] == delivered


def test_treatment_proxy_returns_structured_frozen_replay_divergence(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "selection.jsonl"
    resources: list[dict[str, str]] = []
    fixture.write_text(json.dumps({
        "request_input_sha256": "not-the-live-request",
        "selected_resource_digest": _selection_digest([]),
        "resources": resources,
    }) + "\n", encoding="utf-8")
    interaction_path = tmp_path / "interaction_history.jsonl"
    proxy = TreatmentProxy(
        "http://unused.invalid/v1",
        mode=ContextTreatment.GATEWAY_NATIVE_PRA,
        budget_fraction=0.5,
        trace_path=tmp_path / "trace.jsonl",
        selection_replay_path=fixture,
        interaction_trace_path=interaction_path,
    )
    url = proxy.start()
    request = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps({
            "model": "model",
            "messages": [{"role": "user", "content": "live request"}],
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(request, timeout=5)
        assert captured.value.code == 409
        body = json.loads(captured.value.read().decode("utf-8"))
        assert body["error"] == "frozen_replay_diverged"
        assert "paired trajectories have diverged" in body["message"]
        interaction = json.loads(interaction_path.read_text(encoding="utf-8"))
        assert interaction["event"] == "frozen_replay_divergence"
        assert interaction["error"] == "frozen_replay_diverged"
        assert interaction["logical_payload"]["messages"][0]["content"] == "live request"
    finally:
        proxy.close()


def test_transport_history_audit_requires_requests_responses_and_engine_metrics(
    tmp_path: Path,
) -> None:
    request = {
        "event": "request",
        "request_index": 1,
        "request_input_sha256": "logical",
        "physical_payload": {
            "pra": {"resources": [{"resource_id": "task", "text": "body"}]},
        },
    }
    response = {
        "event": "response",
        "request_index": 1,
        "payload": {
            "choices": [{"message": {"content": "same"}}],
            "pra": {
                "prefix_cache_hit": True,
                "prefix_cached_tokens": 10,
                "engine_cached_tokens_total": 10,
                "native_tokens": 10,
                "wire_tokens": 2,
                "physical_kv_copy": False,
            },
        },
    }
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    content = "\n".join(json.dumps(row) for row in (request, response)) + "\n"
    reference.write_text(content, encoding="utf-8")
    candidate.write_text(content, encoding="utf-8")

    result = compare_transport_histories(reference, candidate)

    assert result["transport_equivalent"] is True
    assert result["exact_paired_responses"] == 1
    modified = dict(response)
    modified["payload"] = json.loads(json.dumps(response["payload"]))
    modified["payload"]["pra"]["wire_tokens"] = 3
    candidate.write_text(
        "\n".join(json.dumps(row) for row in (request, modified)) + "\n",
        encoding="utf-8",
    )
    result = compare_transport_histories(reference, candidate)
    assert result["transport_equivalent"] is False
    assert result["first_difference"] == 1


def test_fim14b_campaign_pins_published_identity_and_orders_treatments() -> None:
    campaign = CampaignConfig.load(CONFIG)
    baseline = campaign.baselines[0]
    assert baseline.model == "TIGER-Lab/FIM-14B"
    assert baseline.published_score == 0.292
    assert baseline.published_total == 500
    assert baseline.max_steps_absolute == 100
    assert baseline.function_calling is False
    assert baseline.prefix_caching is True
    assert campaign.cells[0].baseline_cell is None
    assert all(cell.baseline_cell == "fim14b-no-pra" for cell in campaign.cells[1:])


def test_changed_engine_is_attempted_not_reproduced() -> None:
    campaign = CampaignConfig.load(CONFIG)
    result = OfficialResult(
        official_grader=True, score=0.30, resolved=150, total=500,
        configuration_differences=("engine: MLX instead of vLLM",),
    )
    review = review_result(
        campaign.baselines[0], result, absolute_tolerance=0.05, require_exact_cohort=True,
    )
    assert review.status == ReproductionStatus.BASELINE_ATTEMPTED
    assert not review.compatible


def test_compatible_official_result_admits_baseline() -> None:
    campaign = CampaignConfig.load(CONFIG)
    result = OfficialResult(official_grader=True, score=0.29, resolved=145, total=500)
    review = review_result(
        campaign.baselines[0], result, absolute_tolerance=0.05, require_exact_cohort=True,
    )
    assert review.status == ReproductionStatus.BASELINE_REPRODUCED


def test_treatment_without_baseline_dependency_is_rejected() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["cells"][1]["baseline_cell"] = None
    with pytest.raises(ValueError, match="require baseline_cell"):
        CampaignConfig.model_validate(payload)


def test_dry_run_persists_reports_and_blocks_treatments(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "campaign")
    for cell in payload["cells"]:
        cell["enabled"] = True
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # The runner resolves output relative to the repository inferred from config;
    # an absolute path remains absolute on every supported host.
    state = run_campaign(config, max_hours=1, resume=False, dry_run=True)
    assert state["cells"]["fim14b-no-pra"]["state"] == "PENDING"
    assert state["cells"]["fim14b-gateway-passthrough"]["state"] == "BLOCKED"
    output = tmp_path / "campaign"
    assert (output / "campaign_state.json").is_file()
    assert (output / "reproduction_report.md").is_file()
    assert (output / "precision_report.md").is_file()
    assert (output / "engine_report.md").is_file()
    assert (output / "pra_frontier_report.md").is_file()
    assert json.loads((output / "summary.json").read_text())["pra_interpretation_allowed"] is False


def test_r2egym_converter_uses_only_visible_trajectory_patch(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.jsonl"
    source.write_text(json.dumps({
        "ds": {"instance_id": "repo__project-1", "patch": "hidden-gold"},
        "output_patch": "diff --git a/a.py b/a.py\n",
    }) + "\n", encoding="utf-8")
    destination = tmp_path / "predictions.jsonl"
    rows = trajectories_to_predictions(source, destination, "TIGER-Lab/FIM-7B")
    assert rows[0]["model_patch"].startswith("diff --git")
    assert "hidden-gold" not in destination.read_text(encoding="utf-8")


def test_official_swebench_report_is_normalized(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "submitted_ids": ["a", "b"], "resolved_ids": ["a"],
    }), encoding="utf-8")
    receipt = normalize_official_report(
        report, tmp_path / "official_result.json",
        configuration_differences=("engine: llama.cpp instead of vLLM",),
    )
    assert receipt["score"] == 0.5
    assert receipt["official_grader"] is True
    assert receipt["configuration_differences"]


def test_official_report_written_in_harness_root_is_retained(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    report = tmp_path / "model.smoke.json"
    report.write_text('{"submitted_ids": ["a"]}', encoding="utf-8")

    located = locate_official_report(output, tmp_path, "smoke")

    assert located == output / report.name
    assert located.read_bytes() == report.read_bytes()


def test_r2egym_task_telemetry_keeps_no_pra_costs_separate(tmp_path: Path) -> None:
    trajectories = tmp_path / "trajectory.jsonl"
    trajectories.write_text(json.dumps({
        "ds": {"instance_id": "repo__project-1"},
        "max_token_limit": 32768,
        "exit_reason": "agent",
        "output_patch": "diff\n+line\n",
        "trajectory_steps": [
            {
                "token_usage_prompt": 100, "token_usage_completion": 20,
                "llm_exec_time": 2.0, "env_exec_time": 0.5,
                "total_time_traj": 2.5, "action": "read file",
            },
            {
                "token_usage_prompt": 150, "token_usage_completion": 10,
                "llm_exec_time": 3.0, "env_exec_time": 0.25,
                "total_time_traj": 5.75, "tool_calls": [{"name": "patch"}],
            },
        ],
    }) + "\n", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "submitted_ids": ["repo__project-1"],
        "resolved_ids": ["repo__project-1"], "error_ids": [],
    }), encoding="utf-8")
    rows = write_task_results(
        trajectories, report, tmp_path / "results.jsonl",
        model="model", model_revision="revision", engine="llama.cpp",
        engine_version="version", quantization="Q4_K_M", harness_version="commit",
    )
    assert rows[0]["resolved"] is True
    assert rows[0]["physical_input_tokens"] == rows[0]["logical_input_tokens"] == 250
    assert rows[0]["unique_context_tokens_estimate"] == 150
    assert rows[0]["repeated_context_tokens_estimate"] == 100
    assert rows[0]["repeated_context_fraction_estimate"] == pytest.approx(0.4)
    assert rows[0]["model_call_count"] == rows[0]["trajectory_length"] == 2
    assert rows[0]["tool_call_count"] == 2
    assert rows[0]["p95_model_call_s"] == pytest.approx(2.95)
    assert rows[0]["patch_bytes"] == 11
    assert rows[0]["patch_lines"] == 2
    assert rows[0]["grader_outcome"] == "resolved"
    assert rows[0]["pra_route_time_s"] == rows[0]["pra_memory_bytes"] == 0
    assert rows[0]["prefill_time_s"] is None


def test_distributed_result_import_preserves_attempted_status(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "campaign")
    payload["cells"] = [payload["cells"][0]]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    result = tmp_path / "official_result.json"
    result.write_text(json.dumps({
        "official_grader": True, "score": 0.5, "resolved": 1, "total": 2,
        "configuration_differences": ["cohort=2-not-500"],
    }), encoding="utf-8")
    state = record_result(config, cell_id="fim14b-no-pra", result_path=result)
    cell = state["cells"]["fim14b-no-pra"]
    assert cell["reproduction_status"] == "BASELINE_ATTEMPTED"
    assert (tmp_path / "campaign/reproduction_report.md").is_file()


def test_agent_baseline_summary_keeps_small_cohort_locked() -> None:
    rows = [
        {
            "resolved": resolved,
            "cumulative_prompt_tokens": 100,
            "unique_context_tokens_estimate": 25,
            "repeated_context_tokens_estimate": 75,
            "output_tokens": 10,
            "model_call_count": 2,
            "tool_call_count": 2,
            "trajectory_length": 2,
            "patch_bytes": 20,
            "wall_time_s": 5.0,
            "decode_time_s": 3.0,
            "tool_time_s": 1.0,
            "repeated_context_fraction_estimate": 0.75,
            "ttft_ms": None,
            "prefill_time_s": None,
            "peak_memory_bytes": None,
            "kv_bytes": None,
        }
        for resolved in (True, False)
    ]
    result = summarize(rows, minimum_tasks=20)
    assert result["cohort_status"] == "INSUFFICIENT_COHORT"
    assert result["pra_treatment_unlocked"] is False
    assert result["token_totals"]["repeated_context_fraction_estimate"] == 0.75
    assert result["success_wilson_95_ci"] == pytest.approx(
        [0.09453120573423074, 0.9054687942657693]
    )


def test_stronger_model_matrix_crosses_10_tasks_and_three_harnesses() -> None:
    config = HarnessMatrixConfig.load(MATRIX_CONFIG)
    manifest = BenchmarkManifest.load(ROOT / config.manifest)
    cells = matrix_cells(config, manifest)
    assert len(cells) == 30
    assert {cell[2].harness_id for cell in cells[:3]} == {
        "mini-swe-agent-2.4.6", "qwen-code-0.23.0", "aider-0.86.2",
    }
    assert len({cell[3] for cell in cells[:15]}) == 5
    aider = next(row for row in config.harnesses if row.agent == "aider")
    assert aider.kwargs == {
        "stream": False, "auto_lint": False, "auto_test": False,
    }


def test_harness_matrix_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        HarnessMatrixConfig.load(path)


def test_completed_attempt_finds_terminal_interrupted_job(tmp_path: Path) -> None:
    attempt = tmp_path / "attempts/a002/job"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1, "n_cancelled_trials": 0,
            "n_running_trials": 0,
        },
    }), encoding="utf-8")
    assert _completed_attempt(tmp_path) == tmp_path / "attempts/a002"


def test_completed_attempt_ignores_running_job(tmp_path: Path) -> None:
    attempt = tmp_path / "attempts/a001/job"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({
        "n_total_trials": 1,
        "stats": {"n_completed_trials": 0, "n_running_trials": 1},
    }), encoding="utf-8")
    assert _completed_attempt(tmp_path) is None


def test_harbor_matrix_command_is_one_task_and_redactable() -> None:
    config = HarnessMatrixConfig.load(MATRIX_CONFIG)
    manifest = BenchmarkManifest.load(ROOT / config.manifest)
    _, model, harness, task_id, _, _ = matrix_cells(config, manifest)[0]
    command = harbor_command(
        harbor="harbor", manifest=manifest, model=model, harness=harness,
        task_id=task_id, job_directory=Path("jobs"),
        base_url="http://model-host:11435/v1", api_key="secret",
    )
    assert command.count("-i") == 1
    assert f"terminal-bench/{task_id}" in command
    assert "OPENAI_BASE_URL=http://model-host:11435/v1" in command
    assert "version=2.4.6" in command
    assert "max_tokens=8192" in command

    aider = next(row for row in config.harnesses if row.agent == "aider")
    aider_command = harbor_command(
        harbor="harbor", manifest=manifest, model=model, harness=aider,
        task_id=task_id, job_directory=Path("jobs"),
        base_url="http://model-host:11435/v1", api_key="secret",
    )
    assert aider_command[aider_command.index("-m") + 1] == (
        "openai/openai/qwen3-coder:30b"
    )


def test_harbor_matrix_retry_uses_isolated_attempt_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load(MATRIX_CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "matrix")
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.delenv("PRA_AGENT_QWEN3_CODER_30B_URL", raising=False)
    state = run_matrix(config_path, resume=False, dry_run=True, max_cells=1)
    record = next(iter(state["cells"].values()))
    assert record["active_attempt"] == "a001"
    assert "attempts" in record["command"][record["command"].index("--jobs-dir") + 1]


def test_harbor_matrix_dry_run_is_pra_locked_and_writes_all_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load(MATRIX_CONFIG.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "matrix")
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.delenv("PRA_AGENT_QWEN3_CODER_30B_URL", raising=False)
    state = run_matrix(config_path, resume=False, dry_run=True)
    assert state["pra_enabled"] is False
    assert len(state["cells"]) == 30
    assert {record["state"] for record in state["cells"].values()} == {"PENDING"}
    summary = json.loads((tmp_path / "matrix/summary.json").read_text(encoding="utf-8"))
    assert summary["expected_runs"] == 30
    assert summary["completed_runs"] == 0
    assert summary["summary"]["tasks_solved_any"] == 0
    assert summary["summary"]["token_reported_runs"] == 0
    assert summary["admission_gate"]["eligible"] is False
    report = (tmp_path / "matrix/report.md").read_text(encoding="utf-8")
    assert "official-Harbor **No-PRA baseline**" in report
    assert "Completed: `0/30`" in report


def test_pre_inference_harness_failure_is_not_quality_evidence() -> None:
    row = SimpleNamespace(
        outcome=SimpleNamespace(failure_kind="NonZeroAgentExitCodeError"),
        behavior=SimpleNamespace(model_calls=0),
        tokens=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
    assert "excluded" in (_invalid_trial_reason(row) or "")


def test_scored_model_failure_remains_admissible() -> None:
    row = SimpleNamespace(
        outcome=SimpleNamespace(failure_kind="official_score_below_success"),
        behavior=SimpleNamespace(model_calls=4),
        tokens=SimpleNamespace(input_tokens=1200, output_tokens=80),
    )
    assert _invalid_trial_reason(row) is None


def test_agent_timeout_without_adapter_telemetry_remains_admissible() -> None:
    row = SimpleNamespace(
        outcome=SimpleNamespace(failure_kind="AgentTimeoutError"),
        behavior=SimpleNamespace(model_calls=0),
        tokens=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
    assert _invalid_trial_reason(row) is None


def test_pra_transport_matrix_pairs_direct_and_gateway_to_one_engine() -> None:
    path = ROOT / "experiments/paper4_5_agent/configs/harness_matrices/qwen3_coder_30b_pra_transport.yaml"
    config = HarnessMatrixConfig.load(path)
    manifest = BenchmarkManifest.load(ROOT / config.manifest)
    cells = matrix_cells(config, manifest)

    assert config.matrix_kind == "pra_transport"
    assert config.evidence_role == "transport_qualification"
    assert len(cells) == 40
    first_direct, first_gateway = cells[:2]
    assert first_direct[5].connection == "direct"
    assert first_gateway[5].connection == "gateway"
    assert first_gateway[5].requires_route == first_direct[5].route_id
    assert first_direct[5].engine_target_id == first_gateway[5].engine_target_id


def test_pra_transport_admission_is_per_harness() -> None:
    path = ROOT / "experiments/paper4_5_agent/configs/harness_matrices/qwen3_coder_30b_pra_transport.yaml"
    config = HarnessMatrixConfig.load(path)
    admission = _load_baseline_admission(config, ROOT)

    assert admission["eligible"] is True
    assert admission["by_harness"]["mini-swe-agent"]["eligible"] is True
    assert admission["by_harness"]["qwen-coder"]["eligible"] is True
    assert admission["by_harness"]["aider"]["eligible"] is False


def test_native_route_rejects_an_ordinary_engine() -> None:
    with pytest.raises(ValueError, match="engine_pra_enabled"):
        MatrixRoute(
            route_id="invalid", connection="direct", base_url_env="URL",
            pra_mode="native-memory", pra_profile="balanced",
            engine_pra_enabled=False, engine_target_id="engine-1",
        )


def test_route_preflight_authenticates_and_verifies_native_gateway() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            assert self.headers["Authorization"] == "Bearer secret"
            payload = (
                {"data": [{"id": "served-model"}]}
                if self.path == "/v1/models"
                else {
                    "gateway_mode": "G11",
                    "effective_capabilities": {"native_kv": True},
                }
            )
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = MatrixModel(
        model_id="model", served_model="served-model", model_revision="revision",
        engine="engine", engine_version="version",
        routes=(MatrixRoute(
            route_id="gateway", connection="gateway", base_url_env="URL",
            pra_mode="native-memory", pra_profile="balanced",
            engine_pra_enabled=True, gateway_pra_enabled=True, gateway_mode="G11",
            engine_target_id="engine-1",
        ),),
    )
    try:
        _route_preflight(
            f"http://127.0.0.1:{server.server_port}/v1", api_key="secret",
            model=model, route=model.routes[0],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_native_treatment_placement_separates_gateway_and_engine() -> None:
    direct = treatment_placement("direct-native-pra")
    gateway = treatment_placement("gateway-native-pra")

    assert direct == {
        "connection": "direct", "engine_pra_enabled": True,
        "gateway_pra_enabled": False, "gateway_mode": None,
    }
    assert gateway == {
        "connection": "gateway", "engine_pra_enabled": True,
        "gateway_pra_enabled": True, "gateway_mode": "G11",
    }


def test_paired_route_summary_counts_regressions_and_recoveries() -> None:
    def row(connection: str, task: str, success: bool) -> SimpleNamespace:
        return SimpleNamespace(
            identity=SimpleNamespace(
                agent="agent", model="model", task_id=task, repeat=0,
                connection=connection,
            ),
            metadata={"comparison_group": "g", "engine_target_id": "engine-1"},
            outcome=SimpleNamespace(success=success),
            tokens=SimpleNamespace(input_tokens=100),
            timings=SimpleNamespace(task_wall_ms=10.0),
        )

    summary = _paired_route_comparisons([
        row("direct", "a", True), row("gateway", "a", False),
        row("direct", "b", False), row("gateway", "b", True),
        row("direct", "unpaired", True),
    ])[0]

    assert summary["pairs"] == 2
    assert summary["regressions"] == 1
    assert summary["recoveries"] == 1
    assert summary["outcome_matches"] == 0


def test_campaign_command_expands_endpoint_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRA_TEST_ENDPOINT", "http://engine:8000/v1")
    command = _expand_command(("runner", "--base-url", "${PRA_TEST_ENDPOINT}"))
    assert command == ["runner", "--base-url", "http://engine:8000/v1"]


def test_pra_transport_dry_run_orders_direct_before_gateway(
    tmp_path: Path,
) -> None:
    source = ROOT / "experiments/paper4_5_agent/configs/harness_matrices/qwen3_coder_30b_pra_transport.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["output_directory"] = str(tmp_path / "matrix")
    payload["baseline_admission_path"] = str(
        ROOT / payload["baseline_admission_path"]
    )
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    state = run_matrix(config, resume=False, dry_run=True)

    direct = [
        row for row in state["cells"].values()
        if row.get("route_id") == "pra-engine-direct"
    ]
    gateway = [
        row for row in state["cells"].values()
        if row.get("route_id") == "pra-gateway-engine"
    ]
    assert len(direct) == len(gateway) == 20
    assert {row["state"] for row in direct} == {"PENDING"}
    assert {row["state"] for row in gateway} == {"BLOCKED"}
    assert all("requires" in row["reason"] for row in gateway)
