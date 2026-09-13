from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from pra_hf.deployment import PRAWireRequest, PRAWireResource
from pra_vllm.agent_executor import (
    VLLMCudaAgentHistoryExecutor,
    VLLMGenerationReceipt,
    VLLMInProcessSchedulerDriver,
    _enforce_page_retention_floor,
)
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand
from pra_vllm.cuda_scheduler_alias import (
    SchedulerPageSelection,
    VLLMCudaSchedulerPageRegistry,
)


class _Tokenizer:
    chat_template = "stable"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_):
        assert not tokenize
        text = "".join(
            f"<{row['role']}>{row.get('content', '')}</{row['role']}>"
            for row in messages
        )
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in text]


class _Driver:
    block_size = 8

    def __init__(self) -> None:
        self.commands: list[SparseCudaConnectorCommand] = []
        self.prompts: list[list[int]] = []
        self.evictions: list[tuple[str, int]] = []
        self.bad_callbacks = False

    def generate(
        self,
        prompt_token_ids,
        command,
        *,
        max_tokens,
        commit_source=None,
        selected_page_indices=(),
        parent_source_key=None,
    ):
        self.commands.append(command)
        self.prompts.append(list(prompt_token_ids))
        assert SparseCudaConnectorCommand.parse(command.cache_salt()) == command
        if command.mode == "store":
            delta = {"source_pin_events": 1}
            committed = None
        else:
            delta = {
                "alias_hit_events": 1,
                "alias_prepare_events": 1,
                "alias_commit_events": 1,
                "alias_release_events": 1,
            }
            if self.bad_callbacks:
                delta["alias_commit_events"] = 0
            committed = (commit_source[0], commit_source[1], command.source_position_base)
            assert parent_source_key
            assert selected_page_indices
        return VLLMGenerationReceipt(
            f"request-{len(self.commands)}",
            "A",
            (ord("A"),),
            delta,
            committed,
            1234,
            0.01,
        )

    def generate_plain(self, prompt_token_ids, *, max_tokens):
        self.prompts.append(list(prompt_token_ids))
        return VLLMGenerationReceipt(
            f"plain-{len(self.prompts)}", "A", (ord("A"),), {}, None, 4321, 0.02
        )

    def evict_source(self, logical_key, generation):
        self.evictions.append((logical_key, generation))
        return (0,)

    def terminate_source(self, logical_key, generation):
        return True


def _manifest(messages):
    return [
        {
            "message_index": index,
            "role": row["role"],
            "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
        }
        for index, row in enumerate(messages)
    ]


def _request(
    messages,
    mandatory,
    *,
    session="s",
    retention=1.0,
    resources=(),
    selection_contract=None,
):
    metadata = {
        "history_projection": "live-agent-kv-v1",
        "logical_message_manifest": _manifest(messages),
        "mandatory_message_indices": list(mandatory),
        "target_retention_fraction": retention,
    }
    if selection_contract is not None:
        metadata["selection_contract"] = selection_contract
    return PRAWireRequest(
        model="tiny",
        messages=tuple(messages[index] for index in mandatory),
        resources=tuple(resources),
        session_id=session,
        max_new_tokens=1,
        metadata=metadata,
    )


def test_stateful_bridge_stores_then_loads_only_suffix_with_exact_commands() -> None:
    driver = _Driver()
    executor = VLLMCudaAgentHistoryExecutor(
        driver,
        _Tokenizer(),
        model_id="tiny",
        chat_template_digest=hashlib.sha256(b"stable").hexdigest(),
    )
    initial = (
        {"role": "system", "content": "rules-long-enough"},
        {"role": "user", "content": "task-long-enough"},
        {"role": "assistant", "content": "inspect-old"},
        {"role": "user", "content": "result-old"},
    )
    first = executor.generate(_request(initial, range(len(initial))))
    assert driver.commands[0].mode == "store"
    assert first.trace[0]["consumption_mode"] == "initial_store"

    logical = tuple(executor._sessions["s"].ledger.messages) + (
        {"role": "user", "content": "new-result"},
    )
    second = executor.generate(_request(logical, (0, 1, len(logical) - 1)))
    assert driver.commands[1].mode == "load"
    assert driver.commands[1].source_position_base == driver.commands[0].source_tokens
    assert len(driver.prompts[1]) == (
        driver.commands[1].source_tokens
        + second.trace[0]["new_suffix_tokens_submitted"]
    )
    assert second.trace[0]["consumption_mode"] == "dense_semantic_noop"
    assert second.trace[0]["full_retention"] is True
    assert second.trace[0]["realized_retention_fraction"] == 1.0
    assert second.trace[0]["selected_history_reencoded_tokens"] == 0
    assert second.trace[0]["selected_history_kv_copy_bytes"] == 0
    assert second.trace[0]["host_to_device_bytes"] == 0
    assert second.trace[0]["consumer_temporary_bytes"] == 1234
    assert second.raw["usage"]["prompt_tokens"] == second.trace[0][
        "logical_prompt_tokens"
    ]
    assert second.raw["usage"]["completion_tokens"] == 1
    assert driver.evictions


def test_plain_control_bypasses_scheduler_aliases_and_reports_usage() -> None:
    driver = _Driver()
    executor = VLLMCudaAgentHistoryExecutor(
        driver,
        _Tokenizer(),
        model_id="tiny",
        chat_template_digest=hashlib.sha256(b"stable").hexdigest(),
    )
    request = PRAWireRequest(
        model="tiny",
        messages=({"role": "user", "content": "plain task"},),
        max_new_tokens=2,
    )
    result = executor.generate(request)

    assert result.text == "A"
    assert not driver.commands
    assert result.trace[0]["native_kv_used"] is False
    assert result.trace[0]["consumer_temporary_bytes"] == 4321
    assert result.raw["usage"]["prompt_tokens"] == len(driver.prompts[0])
    assert result.raw["usage"]["completion_tokens"] == 1


def test_vllm_agent_advertises_its_enabled_automatic_prefix_cache() -> None:
    executor = VLLMCudaAgentHistoryExecutor(
        _Driver(),
        _Tokenizer(),
        model_id="tiny",
        chat_template_digest="digest",
    )

    capabilities = executor.capabilities()
    assert capabilities["prefix_cache_enabled"] is True
    assert capabilities["automatic_prefix_cache"] is True
    assert capabilities["prefix_cache_mode"] == "automatic_prefix_cache"


def test_sparse_request_uses_complete_selected_pages_and_original_extent() -> None:
    driver = _Driver()
    executor = VLLMCudaAgentHistoryExecutor(
        driver,
        _Tokenizer(),
        model_id="tiny",
        chat_template_digest="digest",
    )
    initial = (
        {"role": "system", "content": "rules-long-enough"},
        {"role": "user", "content": "task-long-enough"},
        {"role": "assistant", "content": "old-action-long-enough"},
        {"role": "user", "content": "old-output-long-enough"},
        {"role": "assistant", "content": "recent-action"},
        {"role": "user", "content": "recent-output"},
    )
    executor.generate(_request(initial, range(len(initial))))
    logical = tuple(executor._sessions["s"].ledger.messages) + (
        {"role": "user", "content": "current"},
    )
    assistant_index = len(logical) - 2
    resource = PRAWireResource(
        resource_id="latest",
        uri="pra://latest",
        text="A",
        metadata={"message_index": assistant_index},
    )
    result = executor.generate(
        _request(
            logical,
            (0, 1, len(logical) - 1),
            retention=0.9,
            resources=(resource,),
            selection_contract="arbitrary-subset-mechanism-probe",
        )
    )
    trace = result.trace[0]
    command = driver.commands[-1]
    assert trace["consumption_mode"] == "sparse_original_position_pages"
    assert trace["full_retention"] is False
    assert trace["realized_retention_fraction"] == trace[
        "engine_reported_history_kv_retention_fraction"
    ]
    assert command.source_tokens % driver.block_size == 0
    assert command.source_position_base >= command.source_tokens
    assert trace["selected_kv_tokens"] == len(trace["selected_page_indices"]) * 8
    assert trace["selected_kv_tokens"] < command.source_position_base


def test_declared_sparse_request_rounds_up_without_dropping_selected_records() -> None:
    driver = _Driver()
    executor = VLLMCudaAgentHistoryExecutor(
        driver,
        _Tokenizer(),
        model_id="tiny",
        chat_template_digest="digest",
    )
    initial = (
        {"role": "system", "content": "rules-long-enough"},
        {"role": "user", "content": "task-long-enough"},
        {"role": "assistant", "content": "old-action-long-enough"},
        {"role": "user", "content": "old-output-long-enough"},
        {"role": "assistant", "content": "recent-action"},
        {"role": "user", "content": "recent-output"},
    )
    executor.generate(_request(initial, range(len(initial))))
    logical = tuple(executor._sessions["s"].ledger.messages) + (
        {"role": "user", "content": "current"},
    )
    selected_message = len(logical) - 2
    result = executor.generate(
        _request(
            logical,
            (0, 1, len(logical) - 1),
            retention=0.9,
            resources=(PRAWireResource(
                resource_id="latest",
                uri="pra://latest",
                text="A",
                metadata={"message_index": selected_message},
            ),),
        )
    )
    trace = result.trace[0]
    assert trace["retention_rounded_up"] is True
    assert trace["realized_retention_fraction"] >= 0.9
    assert selected_message in trace["realized_selected_message_indices"]
    assert set(trace["requested_selected_message_indices"]).issubset(
        trace["realized_selected_message_indices"]
    )


def test_page_selection_rejects_underfilled_declared_retention_arm() -> None:
    with pytest.raises(RuntimeError, match="underfill the requested retention floor"):
        _enforce_page_retention_floor(
            selected_tokens=80,
            source_tokens=100,
            requested_fraction=0.9,
            selection_contract=None,
        )
    _enforce_page_retention_floor(
        selected_tokens=80,
        source_tokens=100,
        requested_fraction=0.9,
        selection_contract="arbitrary-subset-mechanism-probe",
    )
    _enforce_page_retention_floor(
        selected_tokens=80,
        source_tokens=100,
        requested_fraction=0.9,
        selection_contract="frozen-agent-memory-plan-v1",
    )


def test_frozen_agent_memory_plan_is_page_rounded_without_adding_records() -> None:
    driver = _Driver()
    executor = VLLMCudaAgentHistoryExecutor(
        driver,
        _Tokenizer(),
        model_id="tiny",
        chat_template_digest="digest",
    )
    initial = (
        {"role": "system", "content": "rules-long-enough"},
        {"role": "user", "content": "task-long-enough"},
        {"role": "assistant", "content": "old-action-long-enough"},
        {"role": "user", "content": "old-output-long-enough"},
        {"role": "assistant", "content": "recent-action"},
        {"role": "user", "content": "recent-output"},
    )
    executor.generate(_request(initial, range(len(initial))))
    logical = tuple(executor._sessions["s"].ledger.messages) + (
        {"role": "user", "content": "current"},
    )
    selected_message = len(logical) - 2
    result = executor.generate(_request(
        logical,
        (0, 1, len(logical) - 1),
        retention=0.99,
        resources=(PRAWireResource(
            resource_id="latest",
            uri="pra://latest",
            text="A",
            metadata={"message_index": selected_message},
        ),),
        selection_contract="frozen-agent-memory-plan-v1",
    ))

    trace = result.trace[0]
    assert trace["retention_rounded_up"] is False
    assert trace["requested_selected_message_indices"] == (
        trace["realized_selected_message_indices"]
    )
    assert trace["realized_retention_fraction"] < 0.99
    assert trace["selected_history_reencoded_tokens"] == 0
    assert trace["physical_kv_copy_bytes"] == 0


def test_bridge_fails_closed_when_commit_callback_receipt_is_missing() -> None:
    driver = _Driver()
    executor = VLLMCudaAgentHistoryExecutor(
        driver, _Tokenizer(), model_id="tiny", chat_template_digest="digest"
    )
    initial = (
        {"role": "system", "content": "rules-long-enough"},
        {"role": "user", "content": "task-long-enough"},
    )
    executor.generate(_request(initial, range(len(initial))))
    driver.bad_callbacks = True
    logical = tuple(executor._sessions["s"].ledger.messages) + (
        {"role": "user", "content": "next"},
    )
    with pytest.raises(RuntimeError, match="alias_commit_events"):
        executor.generate(_request(logical, (0, 1, len(logical) - 1)))


def test_real_driver_rejects_missing_scheduler_connector_callbacks() -> None:
    fake = SimpleNamespace(llm_engine=SimpleNamespace(
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(
            scheduler=SimpleNamespace(connector=SimpleNamespace())
        )),
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(block_size=16)),
    ))
    with pytest.raises(RuntimeError, match="callbacks are absent"):
        VLLMInProcessSchedulerDriver(fake)


class _Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 1
        self.is_null = False


class _Pool:
    def touch(self, blocks):
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        for block in blocks:
            block.ref_cnt -= 1


def test_registry_extends_full_source_with_request_suffix_page_identities() -> None:
    pool = _Pool()
    source = (_Block(10), _Block(11), _Block(12))
    registry = VLLMCudaSchedulerPageRegistry()
    registry.publish_source(
        "source-g1",
        generation=1,
        source_tokens=48,
        blocks_by_group=(source,),
        block_sizes=(16,),
        block_pool=pool,
    )
    # Producer request releases its original ownership; the source pin remains.
    pool.free_blocks(reversed(source))
    selection = SchedulerPageSelection(
        logical_key="selected",
        source_logical_key="source-g1",
        source_generation=1,
        selected_page_indices=(0, 2),
        selected_token_count=32,
        source_position_base=48,
    )
    installed = registry.prepare_alias(
        "request",
        selection,
        prompt_token_count=49,
        create_kv_cache_blocks=lambda groups: SimpleNamespace(blocks=groups),
    )
    registry.note_alias_hit("request")
    pool.touch(installed.blocks[0])
    suffix = _Block(20)
    request_blocks = SimpleNamespace(blocks=((source[0], source[2], suffix),))
    registry.commit_alias("request", request_blocks)

    committed = registry.publish_extended_source(
        "request",
        "source-g2",
        destination_generation=2,
        append_complete_pages=1,
        request_blocks=request_blocks,
    )
    assert committed == 64
    snapshot = registry.snapshot()
    assert snapshot["sources"]["source-g2"]["block_ids"] == [10, 11, 12, 20]
    telemetry = registry.telemetry()
    assert telemetry.alias_hit_events == 1
    assert telemetry.alias_prepare_events == 1
    assert telemetry.alias_commit_events == 1
    registry.finish_request("request")
    # The successor already pins the complete old prefix, so rollover can
    # release the superseded source pin before vLLM drains deferred request
    # aliases.  The request references keep every page live until that drain.
    assert registry.evict_source("source-g1", generation=1) == (10, 11, 12)
    assert [block.ref_cnt for block in source] == [2, 1, 2]
    pool.free_blocks(reversed(request_blocks.blocks[0]))
    assert [block.ref_cnt for block in source] == [1, 1, 1]
    assert registry.snapshot()["sources"]["source-g2"]["block_ids"] == [10, 11, 12, 20]


def test_scheduler_manifest_replay_is_idempotent_but_collision_fails(tmp_path) -> None:
    driver = VLLMInProcessSchedulerDriver.__new__(VLLMInProcessSchedulerDriver)
    driver.connector = SimpleNamespace(
        _directory=lambda logical_key: tmp_path / logical_key
    )
    command = SparseCudaConnectorCommand(
        mode="load",
        logical_key="selected",
        source_generation=1,
        source_tokens=16,
        source_position_base=32,
    )

    kwargs = {
        "parent_source_key": "source-g1",
        "selected_page_indices": (0,),
        "commit_source": ("source-g2", 2),
    }
    driver._manifest(command, **kwargs)
    driver._manifest(command, **kwargs)
    manifest = json.loads(
        (tmp_path / "selected" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["commit_source_generation"] == 2

    with pytest.raises(RuntimeError, match="collided with a different manifest"):
        driver._manifest(command, **{**kwargs, "commit_source": ("source-g3", 3)})
