from __future__ import annotations

import hashlib
import sys
from types import ModuleType

import numpy as np
import pytest
from types import SimpleNamespace

from pra_hf.deployment import PRAWireRequest, PRAWireResource
from pra_hf.deployment import PRAEngineResult
from experiments.paper4_5_agent.serve_sglang_mlx_agent_pra import _completion
from pra_sglang.agent_executor import (
    AgentHistoryLedger,
    SGLangMLXAgentHistoryExecutor,
    _MlxRepetitionPenaltyController,
    _common_prefix,
    _fixed_shape_repetition_token_ids,
    _live_kv_copy_metrics,
    _repetition_token_ids,
    _selected_sampler_prompt,
    _token_digest,
    causal_message_spans,
    configure_append_stable_template,
    enforce_retention_floor,
    qwen3_append_stable_no_thinking_template,
    selected_record_plan,
    split_generation_prompt,
)
from pra_sglang.adapter import SGLangEngineAdapter
from pra_sglang.mlx_native import _qwen_projections
from pra_hf.gateway import PRAGateway
from pra_hf.deployment import PRAGatewayMode


def test_qwen_projection_supports_qwen2_without_qk_norm_and_qwen3_with_it() -> None:
    class Projection:
        def __init__(self, width: int) -> None:
            self.width = width

        def __call__(self, x):
            return np.zeros((*x.shape[:2], self.width), dtype=np.float32)

    x = np.zeros((1, 3, 8), dtype=np.float32)
    qwen2 = SimpleNamespace(
        q_proj=Projection(8),
        k_proj=Projection(4),
        v_proj=Projection(4),
        n_heads=2,
        n_kv_heads=1,
    )
    _, queries, keys, values = _qwen_projections(qwen2, x)
    assert queries.shape == (1, 2, 3, 4)
    assert keys.shape == values.shape == (1, 1, 3, 4)

    norm_calls = []

    def normalize(value):
        norm_calls.append(value.shape)
        return value + 1

    qwen3 = SimpleNamespace(**vars(qwen2), q_norm=normalize, k_norm=normalize)
    _, queries, keys, _ = _qwen_projections(qwen3, x)
    assert norm_calls == [(1, 3, 2, 4), (1, 3, 1, 4)]
    assert np.all(queries == 1)
    assert np.all(keys == 1)


class _Tokenizer:
    """Prefix-stable chat template with one token per rendered character."""

    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_):
        text = "".join(
            f"<{row['role']}>{row.get('content', '')}</{row['role']}>"
            for row in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


def test_generation_split_keeps_short_tool_record_in_resident_source() -> None:
    tokenizer = _Tokenizer()
    messages = (
        {"role": "assistant", "content": "command"},
        {"role": "user", "content": "x"},
    )
    prompt, source, wire = split_generation_prompt(tokenizer, messages)

    assert prompt == [*source, *wire]
    assert wire == list(map(ord, "<assistant>"))
    assert source[-len("</user>") :] == list(map(ord, "</user>"))


def _manifest(messages):
    return [
        {
            "message_index": index,
            "role": row["role"],
            "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
        }
        for index, row in enumerate(messages)
    ]


def test_ledger_recovers_prior_assistant_without_tokenizing_resource_text() -> None:
    ledger = AgentHistoryLedger()
    initial = (
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    )
    first = PRAWireRequest(model="m", messages=initial)
    assert ledger.reconcile(first) == initial
    ledger.append_assistant("inspect")

    logical = (*initial, {"role": "assistant", "content": "inspect"}, {"role": "user", "content": "result"})
    request = PRAWireRequest(
        model="m",
        messages=(initial[0], initial[1], logical[-1]),
        resources=(PRAWireResource(
            resource_id="m2-0-assistant",
            uri="pra://m2",
            text="inspect",
            metadata={"message_index": 2},
        ),),
        metadata={
            "logical_message_manifest": _manifest(logical),
            "mandatory_message_indices": [0, 1, 3],
        },
    )
    assert ledger.reconcile(request) == logical


def test_ledger_rejects_stale_or_mutated_resident_history() -> None:
    ledger = AgentHistoryLedger([{"role": "system", "content": "old"}])
    request = PRAWireRequest(
        model="m",
        messages=({"role": "system", "content": "old"},),
        metadata={
            "logical_message_manifest": [{
                "message_index": 0,
                "role": "system",
                "content_sha256": hashlib.sha256(b"new").hexdigest(),
            }],
            "mandatory_message_indices": [0],
        },
    )
    with pytest.raises(RuntimeError, match="content disagrees"):
        ledger.reconcile(request)


def test_record_plan_keeps_original_extent_and_rounds_selected_child_to_record() -> None:
    tokenizer = _Tokenizer()
    messages = (
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old observation"},
        {"role": "assistant", "content": "new action"},
        {"role": "user", "content": "new observation"},
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    source_tokens = len(prompt) - 8
    spans = causal_message_spans(
        tokenizer, messages, prompt, source_tokens=source_tokens
    )
    plan = selected_record_plan(
        tokenizer,
        messages,
        prompt,
        source_tokens=source_tokens,
        retention_fraction=0.9,
        mandatory_message_indices=(0, 1, 5),
        # A selected child from message 2 materializes the whole causal record.
        selected_message_indices=(2,),
    )

    assert plan.source_position_base == source_tokens
    assert plan.selected_tokens < source_tokens
    assert {row.record_id for row in plan.intervals} == {
        "message:0:system",
        "message:1:user",
        "message:2:assistant",
        "message:5:user",
    }
    assert tuple(plan.intervals) == tuple(
        row for row in spans if row.record_id in {item.record_id for item in plan.intervals}
    )


def test_full_retention_is_one_original_position_interval() -> None:
    tokenizer = _Tokenizer()
    messages = ({"role": "user", "content": "hello"},)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    plan = selected_record_plan(
        tokenizer,
        messages,
        prompt,
        source_tokens=len(prompt) - 2,
        retention_fraction=1.0,
        mandatory_message_indices=(0,),
        selected_message_indices=(),
    )
    assert plan.full_retention
    assert plan.selected_tokens == plan.source_tokens
    # At 100%, record routing is a semantic no-op: the native request receives
    # exactly the same source-position extent as the ordinary dense prompt.
    assert plan == plan.full(plan.source_tokens)


def test_declared_retention_floor_rejects_underfilled_record_selection() -> None:
    from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan

    plan = LiveKVSelectionPlan.create(
        100,
        (LiveKVInterval(0, 89, record_id="selected", causal_group_id="turn:1"),),
        source_position_base=100,
    )
    with pytest.raises(RuntimeError, match="underfill the requested retention floor"):
        enforce_retention_floor(plan, 0.9)
    enforce_retention_floor(
        plan,
        0.9,
        selection_contract="arbitrary-subset-mechanism-probe",
    )


def test_live_projection_fails_closed_without_manifest() -> None:
    request = PRAWireRequest(
        model="m",
        messages=({"role": "user", "content": "task"},),
        metadata={"history_projection": "live-agent-kv-v1"},
    )
    ledger = AgentHistoryLedger()
    with pytest.raises(ValueError, match="requires a complete"):
        ledger.reconcile(request)


def test_sparse_plan_rejects_missing_record_span() -> None:
    tokenizer = _Tokenizer()
    messages = ({"role": "user", "content": "hello"},)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    with pytest.raises(ValueError, match="outside resident history"):
        selected_record_plan(
            tokenizer,
            messages,
            prompt,
            source_tokens=len(prompt) - 2,
            retention_fraction=0.9,
            mandatory_message_indices=(9,),
            selected_message_indices=(),
        )


def test_common_prefix_detects_exact_append_and_rewrite() -> None:
    assert _common_prefix([1, 2, 3], [1, 2, 3, 4]) == 3
    assert _common_prefix([1, 2, 3], [1, 9, 3]) == 1


def test_additional_renderer_stop_tokens_extend_tokenizer_eos() -> None:
    executor = object.__new__(SGLangMLXAgentHistoryExecutor)
    executor.tokenizer = SimpleNamespace(eos_token_id=151645)
    executor.additional_stop_token_ids = frozenset({151643, 151644})
    assert executor._eos_ids() == {151643, 151644, 151645}


def test_repetition_window_is_deduplicated_and_validated() -> None:
    history = [1, 2, 1, 3, 2]
    assert _repetition_token_ids(history, 3) == (1, 2, 3)
    assert _repetition_token_ids(history, -1) == (1, 2, 3)
    assert _repetition_token_ids(history, 0) == ()
    assert _fixed_shape_repetition_token_ids(history, 3) == (1, 2, 3)
    assert _fixed_shape_repetition_token_ids([1, 1, 1, 2, 2], 4) == (
        1, 2, 1, 1
    )
    with pytest.raises(ValueError, match="-1 or non-negative"):
        _repetition_token_ids(history, -2)


def test_repetition_penalty_precedes_greedy_argmax_without_cpu_sync(
    monkeypatch,
) -> None:
    core = ModuleType("mlx.core")
    core.int32 = np.int32
    core.uint32 = np.uint32
    core.array = lambda values, dtype=None: np.array(values, dtype=dtype)
    core.take = lambda values, indices, axis: np.take(values, indices, axis=axis)
    core.where = np.where
    core.stack = np.stack
    core.any = np.any
    core.argmax = np.argmax
    core.argpartition = np.argpartition
    core.max = np.max
    core.min = np.min

    def put_along_axis(values, indices, replacements, axis):
        result = np.array(values, copy=True)
        np.put_along_axis(result, indices, replacements, axis=axis)
        return result

    core.put_along_axis = put_along_axis
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    class Runner:
        def __init__(self):
            self._req_token_ids = {}
            self.observed_logits = None
            self._enable_sampling = True

        def _select_tokens_with_logprobs(
            self, logits, _req_ids, _caches, _edits=None, _spec=None
        ):
            self.observed_logits = logits
            return np.argmax(logits, axis=-1), None

    runner = Runner()
    controller = _MlxRepetitionPenaltyController(runner)
    # Duplicate token 1 is transformed once; zero follows llama's <=0 branch.
    controller.register("r", [1, 1, 2], penalty=2.0, repeat_last_n=64)
    chosen, _ = runner._select_tokens_with_logprobs(
        np.array([[0.0, 10.0, -2.0, 9.0]]), ["r"], [[]]
    )
    assert chosen.tolist() == [3]
    assert runner.observed_logits.tolist() == [[0.0, 5.0, -4.0, 9.0]]
    # Decode consults authoritative runner history instead of stale prefill state.
    runner._req_token_ids["r"] = [3]
    chosen, _ = runner._select_tokens_with_logprobs(
        np.array([[0.0, 10.0, -2.0, 9.0]]), ["r"], [[]]
    )
    assert chosen.tolist() == [1]
    assert controller.unregister("r") == {
        "application_calls": 2,
        "history_tokens_first_step": 3,
        "unique_tokens_first_step": 2,
        "unique_tokens_max": 2,
    }

    # With sampling disabled, the exact top-K proof bypasses a full-vocab
    # scatter but returns the same penalized greedy token.
    runner._enable_sampling = False
    controller.register("greedy", [1], penalty=2.0, repeat_last_n=64)
    chosen, _ = runner._select_tokens_with_logprobs(
        np.array([[0.0, 10.0, 9.0, 8.0]]), ["greedy"], [[]]
    )
    assert chosen.tolist() == [2]
    assert controller.unregister("greedy")["application_calls"] == 1

    tie_logits = np.array([[0.0, 1.0, 10.0, 9.0, 2.0, 9.0]])
    controller.register("tie", [2], penalty=2.0, repeat_last_n=64)
    chosen, _ = runner._select_tokens_with_logprobs(tie_logits, ["tie"], [[]])
    assert chosen.tolist() == [3]
    controller.unregister("tie")

    rng = np.random.default_rng(17)
    for case in range(100):
        logits = np.round(rng.normal(size=97), 1)
        history = rng.integers(0, 97, size=90).tolist()
        history.extend(history[-8:])  # duplicated ids must not compound
        penalty = 1.05
        window = (0, 64, -1)[case % 3]
        reference = logits.copy()
        for token_id in _repetition_token_ids(history, window):
            reference[token_id] = (
                reference[token_id] * penalty
                if reference[token_id] <= 0
                else reference[token_id] / penalty
            )
        request_id = f"property-{case}"
        controller.register(
            request_id, history, penalty=penalty, repeat_last_n=window
        )
        chosen, _ = runner._select_tokens_with_logprobs(
            logits[None, :], [request_id], [[]]
        )
        assert chosen.tolist() == [int(np.argmax(reference))]
        controller.unregister(request_id)


def test_selected_sampler_prompt_matches_full_and_sparse_visible_context() -> None:
    from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan

    source = [10, 11, 12, 13, 14, 15]
    wire = [20, 21]
    full = LiveKVSelectionPlan.full(len(source))
    full_prompt = _selected_sampler_prompt(source, wire, full)
    assert full_prompt == source + wire
    assert _token_digest(full_prompt) == _token_digest(source + wire)

    sparse = LiveKVSelectionPlan.create(
        len(source),
        (
            LiveKVInterval(0, 2, record_id="a"),
            LiveKVInterval(4, 6, record_id="b"),
        ),
        source_position_base=len(source),
    )
    same_subset = [10, 11, 14, 15, 20, 21]
    sparse_prompt = _selected_sampler_prompt(source, wire, sparse)
    assert sparse_prompt == same_subset
    assert _token_digest(sparse_prompt) == _token_digest(same_subset)


def test_adapter_exposes_frozen_effective_stop_token_contract() -> None:
    native = SimpleNamespace(
        prefix_cache_enabled=False,
        chat_template_profile="native",
        chat_template_digest="digest",
        additional_stop_token_ids=frozenset({151643, 151644}),
        default_repetition_penalty=1.05,
        default_repeat_last_n=64,
        _eos_ids=lambda: {151643, 151644, 151645},
    )
    adapter = SGLangEngineAdapter("http://in-process", native_executor=native)
    assert adapter.additional_stop_token_ids == (151643, 151644)
    assert adapter.effective_stop_token_ids == (151643, 151644, 151645)
    assert adapter.default_repetition_penalty == 1.05
    assert adapter.default_repeat_last_n == 64


def test_owner_extension_fails_closed_on_template_history_rewrite() -> None:
    executor = object.__new__(SGLangMLXAgentHistoryExecutor)
    executor.runner = SimpleNamespace(has_request=lambda _: True)
    state = SimpleNamespace(
        canonical_tokens=[1, 2, 3],
        owner_request_id="owner",
        selected_history_reencoded_tokens=0,
    )
    with pytest.raises(RuntimeError, match="append-stable template"):
        executor._ensure_owner(state, [1, 9, 3, 4])
    assert state.selected_history_reencoded_tokens == 0


def test_qwen3_stable_template_rewrites_only_two_content_branches() -> None:
    branch = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    native = f"prefix {branch} middle {branch} suffix"
    stable = qwen3_append_stable_no_thinking_template(native)
    assert stable.startswith("prefix ")
    assert stable.endswith(" suffix")
    assert stable.count("<think>\\n\\n</think>\\n\\n") == 2
    assert stable.count(branch) == 2  # retained only inside the thinking-enabled else


def test_template_profile_records_exact_digest() -> None:
    tokenizer = SimpleNamespace(
        chat_template=(
            "{{- '<|im_start|>' + message.role + '\\n' + content }}"
            "{{- '<|im_start|>' + message.role + '\\n' + content }}"
        )
    )
    digest = configure_append_stable_template(
        tokenizer, "qwen3-stable-no-thinking"
    )
    assert digest == hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()


def test_zero_copy_attach_does_not_report_selected_tokens_as_bytes() -> None:
    metrics = _live_kv_copy_metrics(
        selected_kv_tokens=4096,
        canonical_suffix_graft_d2d_bytes=8192,
    )
    assert metrics["selected_kv_tokens"] == 4096
    assert metrics["native_attach_bytes"] == 0
    assert metrics["selected_history_kv_copy_bytes"] == 0
    assert metrics["physical_kv_copy_bytes"] == 0
    assert metrics["canonical_suffix_graft_d2d_bytes"] == 8192
    assert metrics["total_kv_copy_bytes"] == 8192
    assert metrics["host_to_device_bytes"] == 0


def test_direct_completion_preserves_zero_byte_attach_accounting() -> None:
    request = PRAWireRequest(
        model="m",
        messages=({"role": "user", "content": "task"},),
        request_id="request-1",
    )
    trace = {
        "native_kv_used": True,
        "selected_kv_tokens": 4096,
        **_live_kv_copy_metrics(
            selected_kv_tokens=4096,
            canonical_suffix_graft_d2d_bytes=8192,
        ),
    }
    response = _completion(
        request,
        PRAEngineResult(
            text="done",
            raw={
                "native_attach_bytes": 0,
                "usage": {
                    "prompt_tokens": 64,
                    "completion_tokens": 2,
                    "total_tokens": 66,
                },
                "pra": trace,
            },
            trace=(trace,),
        ),
    )
    assert response["pra"]["native_kv"] is True
    assert response["pra"]["selected_kv_tokens"] == 4096
    assert response["pra"]["native_attach_bytes"] == 0
    assert response["pra"]["canonical_suffix_graft_d2d_bytes"] == 8192
    assert response["pra_trace"] == [trace]
    assert response["usage"]["total_tokens"] == 66


def test_qualified_live_projection_does_not_reset_engine_session() -> None:
    gateway = object.__new__(PRAGateway)
    gateway.mode = PRAGatewayMode.G00_PASS_THROUGH
    gateway.adapter = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(agent_history_kv_qualified=True)
    )
    turn = SimpleNamespace(
        state=SimpleNamespace(
            turns=2,
            model_revision=None,
            chat_template_digest=None,
            visible_prefix_profile=None,
        ),
        prefix_changed_reason="history_rewrite",
    )
    request = SimpleNamespace(metadata={"history_projection": "live-agent-kv-v1"})

    assert gateway._invalidation_reason(turn, request) is None
