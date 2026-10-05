import hashlib
import json
from types import SimpleNamespace

from experiments.paper8_5_agent_memory import run_persistent_first_divergence_oracle as oracle
from experiments.paper8_5_agent_memory.run_persistent_first_divergence_oracle import (
    _apply_diagnostic_generation_overrides,
    _causal_group_addback_batches,
    _completed_epoch_addback_batches,
    _counterfactual_config,
    _find_frozen_request,
    _full_control_config,
    _payload,
)
from experiments.paper8_5_agent_memory.autonomous_proxy import (
    AutonomousSelectionConfig,
)
from experiments.paper8_5_agent_memory.materialization import MaterializationMode
from experiments.paper8_5_agent_memory.multi_issue_session import BoundaryMode
from experiments.paper8_5_agent_memory.negative_receipts import NegativeRealizationMode


def test_generation_row_preserves_content_and_finish_reason(monkeypatch):
    monkeypatch.setattr(
        oracle,
        "_post",
        lambda endpoint, payload, timeout, upstream_dialect="openai": {
            "choices": [{
                "message": {"content": "analysis without a command"},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    )
    transformation = SimpleNamespace(
        payload={"messages": []},
        trace={"selected_messages_sha256": "selected", "oracle_addback": None},
        plan=SimpleNamespace(selected_tokens=10),
        materialized_tokens=10,
    )

    row = oracle._generation_row(
        arm="candidate",
        repeat=1,
        transformation=transformation,
        endpoint="http://unused",
        timeout=1,
    )

    assert row["response_content"] == "analysis without a command"
    assert row["finish_reason"] == "length"
    assert row["action_valid"] is False
    assert row["diagnostic_served_model"] is None
    assert row["diagnostic_max_tokens"] is None
    assert row["upstream_dialect"] == "openai"


def test_frozen_payload_preserves_effective_top_k():
    payload = _payload({
        "served_model": "locked-model",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 20,
        "seed": 0,
        "max_completion_tokens": 32,
    }, [{"role": "user", "content": "task"}])

    assert payload["top_k"] == 20


def test_oracle_native_transport_applies_top_k(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "model": "locked-model",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 1,
            }).encode()

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(oracle.urllib.request, "urlopen", urlopen)
    value = oracle._post(
        "http://engine/api/chat",
        {
            "model": "locked-model",
            "messages": [{"role": "user", "content": "task"}],
            "stream": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "seed": 0,
            "max_tokens": 32,
        },
        7.0,
        upstream_dialect="ollama_native",
    )

    assert captured["url"] == "http://engine/api/chat"
    assert captured["timeout"] == 7.0
    assert captured["payload"]["options"]["top_k"] == 20
    assert captured["payload"]["options"]["num_predict"] == 32
    assert value["choices"][0]["message"]["content"] == "ok"
    assert value["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "total_tokens": 11,
    }


def test_diagnostic_generation_overrides_do_not_change_selection_trace():
    transformation = SimpleNamespace(
        payload={"model": "original", "max_tokens": 1024, "messages": []},
        trace={"selected_messages_sha256": "frozen-selection"},
    )

    _apply_diagnostic_generation_overrides(
        transformation,
        served_model="smaller-screening-model",
        max_tokens=256,
    )

    assert transformation.payload["model"] == "smaller-screening-model"
    assert transformation.payload["max_tokens"] == 256
    assert transformation.trace["selected_messages_sha256"] == "frozen-selection"


def test_full_control_clone_disables_candidate_only_retirement_options():
    candidate = AutonomousSelectionConfig(
        policy="persistent_instruction_epoch_retirement",
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        completed_finalization_turns=1,
        compact_completed_finalizations=True,
        retire_closed_instructions=True,
        materialization_mode=MaterializationMode.TOOL_STRUCTURED_EVIDENCE,
        negative_realization=NegativeRealizationMode.DROP,
        negative_fallback="none",
        static_workflow_anchor="coding_search_inspect_edit_verify_v1",
    )

    control = _full_control_config(candidate)

    assert control.policy == "full"
    assert control.budget_fraction == 1.0
    assert control.materialization_mode is MaterializationMode.WHOLE_RECORD
    assert control.negative_realization is NegativeRealizationMode.DROP
    assert control.negative_fallback == "none"
    assert control.compact_completed_finalizations is False
    assert control.retire_closed_instructions is False
    assert control.static_workflow_anchor == "coding_search_inspect_edit_verify_v1"


def test_counterfactual_clone_changes_only_declared_frontier_values():
    candidate = AutonomousSelectionConfig(
        policy="frontier_dag_retirement",
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        frontier_recent_user_prompts=1,
        frontier_protocol_exemplars=1,
        frontier_workflow_exemplars=1,
        frontier_allow_superseded_unfinished_after=1,
    )

    counterfactual = _counterfactual_config(
        candidate,
        protocol_exemplars=0,
        workflow_exemplars=0,
    )

    assert counterfactual.policy == candidate.policy
    assert counterfactual.boundary_mode is candidate.boundary_mode
    assert counterfactual.frontier_recent_user_prompts == 1
    assert counterfactual.frontier_protocol_exemplars == 0
    assert counterfactual.frontier_workflow_exemplars == 0
    assert counterfactual.frontier_allow_superseded_unfinished_after == 1


def test_counterfactual_clone_rejects_negative_frontier_values():
    candidate = AutonomousSelectionConfig(
        policy="frontier_dag_retirement",
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )

    try:
        _counterfactual_config(candidate, workflow_exemplars=-1)
    except ValueError as exc:
        assert "frontier_workflow_exemplars" in str(exc)
    else:
        raise AssertionError("negative frontier override was accepted")


def test_frozen_request_reconstruction_joins_archived_receipts(tmp_path):
    command = "cat target.py"
    messages = [
        {"role": "system", "content": "Use one command."},
        {"role": "user", "content": "Inspect target.py."},
        {"role": "assistant", "content": (
            "THOUGHT: inspect\n```mswea_bash_command\n"
            + command + "\n```"
        )},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>x</output>"},
    ]
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    (receipt_dir / "execution_0000.json").write_text(json.dumps({
        "schema_version": 1,
        "step": 0,
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "observation_metadata": {
            "return_code": 0,
            "output_complete": True,
            "resource_version_fingerprints": {"target.py": "v1"},
        },
    }))
    manifest = {
        "served_model": "locked-model",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_completion_tokens": 32,
    }
    config = AutonomousSelectionConfig(
        policy="frontier_dag_retirement",
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        expected_model="locked-model",
        task_id="task-1",
        frontier_allow_heuristic=True,
    )
    frozen_payload = _payload(manifest, messages)
    expected = oracle.transform_autonomous_payload(
        frozen_payload,
        config,
        instrumentation_root=tmp_path,
    )
    # The immutable campaign archive later accumulates receipts from
    # subsequent decisions.  Replay of this earlier request must use only its
    # exact leading command/receipt prefix.
    later_command = "pytest -q"
    (receipt_dir / "execution_0001.json").write_text(json.dumps({
        "schema_version": 1,
        "step": 1,
        "command_sha256": hashlib.sha256(later_command.encode()).hexdigest(),
        "observation_metadata": {
            "return_code": 0,
            "output_complete": True,
            "resource_version_fingerprints": {"target.py": "v2"},
        },
    }))

    end, _, reconstructed = _find_frozen_request(
        trajectory={"messages": messages},
        target_request_digest=expected.trace["request_input_sha256"],
        target_selected_digest=expected.trace["selected_messages_sha256"],
        manifest=manifest,
        config=config,
        prior_episodes=(),
        count_tokens=lambda value: len(value.split()),
        instrumentation_root=tmp_path,
    )

    assert end == 4
    assert reconstructed.trace["selection_abstained_for_sidecar"] is False
    assert reconstructed.trace["instrumentation_sidecar_join"] == {
        "status": "exact",
        "commands": 1,
        "receipts": 1,
        "joined": 1,
        "archived_receipts_ignored": 1,
    }


def test_completed_epoch_batches_cover_each_exclusion_once():
    composed = {
        "episodes": [
            {"episode_index": 1, "model_visible_messages": 3},
            {"episode_index": 2, "model_visible_messages": 2},
        ]
    }
    history = SimpleNamespace(records=[
        SimpleNamespace(record_id=f"record-{index:06d}") for index in range(5)
    ])
    exclusions = [
        SimpleNamespace(
            causal_group_id="turn-000000",
            record_ids=("record-000001", "record-000002"),
            excluded_tokens=11,
        ),
        SimpleNamespace(
            causal_group_id="turn-000001",
            record_ids=("record-000003", "record-000004"),
            excluded_tokens=13,
        ),
    ]

    assert _completed_epoch_addback_batches(
        composed=composed, history=history, exclusions=exclusions,
    ) == [
        {
            "epoch_index": 1,
            "causal_group_ids": ("turn-000000",),
            "excluded_tokens": 11,
        },
        {
            "epoch_index": 2,
            "causal_group_ids": ("turn-000001",),
            "excluded_tokens": 13,
        },
    ]


def test_causal_group_batches_can_be_limited_to_one_source_epoch():
    composed = {
        "episodes": [
            {"episode_index": 1, "model_visible_messages": 3},
            {"episode_index": 2, "model_visible_messages": 2},
        ]
    }
    history = SimpleNamespace(records=[
        SimpleNamespace(record_id=f"record-{index:06d}") for index in range(5)
    ])
    exclusions = [
        SimpleNamespace(
            causal_group_id="turn-000000",
            record_ids=("record-000001", "record-000002"),
            excluded_tokens=11,
        ),
        SimpleNamespace(
            causal_group_id="turn-000001",
            record_ids=("record-000003", "record-000004"),
            excluded_tokens=13,
        ),
    ]

    assert _causal_group_addback_batches(
        composed=composed,
        history=history,
        exclusions=exclusions,
        addback_epoch=2,
    ) == [
        {
            "epoch_index": 2,
            "causal_group_ids": ("turn-000001",),
            "excluded_tokens": 13,
        }
    ]


def test_causal_group_batches_can_target_explicit_groups():
    composed = {
        "episodes": [{"episode_index": 1, "model_visible_messages": 4}]
    }
    history = SimpleNamespace(records=[
        SimpleNamespace(record_id=f"record-{index:06d}") for index in range(4)
    ])
    exclusions = [
        SimpleNamespace(
            causal_group_id="turn-000000",
            record_ids=("record-000000", "record-000001"),
            excluded_tokens=11,
        ),
        SimpleNamespace(
            causal_group_id="turn-000001",
            record_ids=("record-000002", "record-000003"),
            excluded_tokens=13,
        ),
    ]

    assert _causal_group_addback_batches(
        composed=composed,
        history=history,
        exclusions=exclusions,
        causal_group_ids=("turn-000001",),
    ) == [
        {
            "epoch_index": None,
            "causal_group_ids": ("turn-000001",),
            "excluded_tokens": 13,
        }
    ]
