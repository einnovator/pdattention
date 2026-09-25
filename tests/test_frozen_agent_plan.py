from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.paper4_5_agent.context_treatment import (
    _selection_digest,
    _selection_input_digest,
)
from experiments.paper4_5_agent.frozen_agent_plan import (
    _template_messages,
    frozen_live_kv_geometry,
    load_frozen_agent_decisions,
)


class CharacterTemplate:
    @staticmethod
    def _render(messages, add_generation_prompt: bool) -> str:
        text = "".join(
            f"<{row['role']}>{row['content']}</{row['role']}>"
            for row in messages
        )
        return text + ("<assistant>" if add_generation_prompt else "")

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = self._render(messages, add_generation_prompt)
        return [ord(char) for char in rendered] if tokenize else rendered

    @staticmethod
    def encode(rendered, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in rendered]


def test_openai_text_parts_are_projected_without_losing_tool_fields() -> None:
    tool_calls = [{
        "id": "call-1",
        "type": "function",
        "function": {"name": "bash", "arguments": "{}"},
    }]
    normalized = _template_messages([
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": " two"},
            ],
            "tool_calls": tool_calls,
        }
    ])
    assert normalized == [{
        "role": "assistant",
        "content": "one two",
        "tool_calls": tool_calls,
    }]
    with pytest.raises(ValueError, match="only text content parts"):
        _template_messages([{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "x"}}],
        }])


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old observation"},
        {"role": "user", "content": "current task"},
    ]
    request_digest = _selection_input_digest(messages)
    resources = [("m1-0-user", "old task")]
    plan = {
        "request_index": 1,
        "request_input_sha256": request_digest,
        "selected_message_indices": [0, 1, 4],
        "mandatory_message_indices": [0, 4],
        "selected_resource_digest": _selection_digest(resources),
        "source_policy": "frontier_dag_m2_heuristic_p1",
        "source_plan_digest": "plan",
        "resources": [
            {"resource_id": resource_id, "text": text}
            for resource_id, text in resources
        ],
    }
    response = "THOUGHT: act"
    replay = {
        "request_index": 1,
        "request_input_sha256": request_digest,
        "session_id": "session",
        "logical_payload": {"model": "model", "messages": messages},
        "expected_assistant_content": response,
        "expected_assistant_content_sha256": hashlib.sha256(
            response.encode()
        ).hexdigest(),
    }
    plan_path = tmp_path / "plan.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    replay_path.write_text(json.dumps(replay) + "\n", encoding="utf-8")
    return replay_path, plan_path


def test_frozen_plan_maps_exact_records_and_current_tail(tmp_path: Path) -> None:
    replay, plan = _write_pair(tmp_path)
    decision = load_frozen_agent_decisions(replay, plan)[0]
    geometry = frozen_live_kv_geometry(CharacterTemplate(), decision)

    assert geometry.selected_message_indices == (0, 1, 4)
    assert geometry.mandatory_message_indices == (0, 4)
    assert [interval.record_id for interval in geometry.plan.intervals] == [
        "message:0:system",
        "message:1:user",
    ]
    assert geometry.plan.has_holes
    assert "<user>current task</user><assistant>" == "".join(
        chr(token) for token in geometry.wire_tail_ids
    )
    assert geometry.realized_retention_fraction < 1.0

    full = frozen_live_kv_geometry(
        CharacterTemplate(), decision, full_retention=True
    )
    assert not full.plan.has_holes
    assert full.plan.selected_tokens == len(full.source_ids)


def test_frozen_plan_renders_native_tools_and_validates_full_action(
    tmp_path: Path,
) -> None:
    replay_path, plan_path = _write_pair(tmp_path)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    tools = [{
        "type": "function",
        "function": {"name": "bash", "parameters": {"type": "object"}},
    }]
    replay["logical_payload"]["tools"] = tools
    replay["logical_payload_sha256"] = hashlib.sha256(json.dumps(
        replay["logical_payload"],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()).hexdigest()
    expected_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"},
        }],
    }
    replay["expected_assistant_message"] = expected_message
    replay["expected_assistant_message_sha256"] = hashlib.sha256(json.dumps(
        expected_message,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()).hexdigest()
    replay_path.write_text(json.dumps(replay) + "\n", encoding="utf-8")
    calls = []

    class ToolTemplate(CharacterTemplate):
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, tools
        ):
            calls.append(tools)
            rendered = json.dumps(tools, sort_keys=True)
            rendered += self._render(messages, add_generation_prompt)
            return [ord(char) for char in rendered] if tokenize else rendered

    decision = load_frozen_agent_decisions(replay_path, plan_path)[0]
    assert decision.expected_assistant_message == expected_message
    frozen_live_kv_geometry(ToolTemplate(), decision)
    assert calls
    assert all(call == tools for call in calls)


def test_frozen_plan_rejects_excluded_message_inside_wire_tail(tmp_path: Path) -> None:
    replay, plan = _write_pair(tmp_path)
    row = json.loads(plan.read_text(encoding="utf-8"))
    row["selected_message_indices"] = [0, 1]
    plan.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mandatory selected state"):
        load_frozen_agent_decisions(replay, plan)


def test_strict_paper85_request_and_plan_bundles_join_exactly() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "docs/papers/shared/results/paper4_5_runtime_productization"
        / "agent_memory_plans/paper8_5_m2_p1_strict_v1"
    )
    for task, expected in ((3, 7), (4, 6), (5, 9)):
        decisions = load_frozen_agent_decisions(
            root / f"paper4_5_task{task}_request_replay.jsonl",
            root / f"paper4_5_task{task}_frozen_plan.jsonl",
        )
        assert len(decisions) == expected
        assert all(
            row.source_policy == "frontier_dag_m2_heuristic_p1"
            for row in decisions
        )


def test_heldout_m2p1_task8_handoff_is_hash_bound_and_complete() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "docs/papers/shared/results/paper4_5_runtime_productization"
        / "agent_memory_plans/paper8_5_m2_p1_heldout_task8_v1"
    )
    manifest = json.loads(
        (root / "frozen_plan_manifest.json").read_text(encoding="utf-8")
    )
    plan = root / "frozen_plan.jsonl"
    replay = root / "request_replay.jsonl"
    decisions = load_frozen_agent_decisions(replay, plan)

    assert len(decisions) == manifest["requests"] == 6
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == manifest[
        "fixture_sha256"
    ]
    assert hashlib.sha256(replay.read_bytes()).hexdigest() == manifest[
        "request_replay_sha256"
    ]
    assert all(
        row.source_policy == "frontier_dag_m2_heuristic_p1"
        for row in decisions
    )
    audit = json.loads(
        (root / "qwen3coder30b_tokenizer_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["all_geometry_valid"] is True
    assert len(audit["rows"]) == 6
    assert max(row["realized_retention_fraction"] for row in audit["rows"]) < 0.39


def test_materialized_receipt_fixture_exposes_mixed_kv_spans() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "docs/papers/shared/results/paper4_5_runtime_productization"
        / "agent_memory_plans/paper8_5_e2_f1c_scikit14496_v1"
    )
    decisions = load_frozen_agent_decisions(
        root / "request_replay.jsonl", root / "frozen_plan.jsonl"
    )
    assert len(decisions) == 9
    assert decisions[-1].materialized_message_replacements

    geometry = frozen_live_kv_geometry(CharacterTemplate(), decisions[-1])
    assert geometry.materialized_history_spans
    replacement_indices = {
        row.message_index for row in decisions[-1].materialized_message_replacements
    }
    selected_resident_indices = {
        int(str(interval.record_id).split(":", 2)[1])
        for interval in geometry.plan.intervals
    }
    assert replacement_indices.isdisjoint(selected_resident_indices)
    assert all(
        row.position_end <= geometry.plan.source_position_base
        for row in geometry.materialized_history_spans
    )
