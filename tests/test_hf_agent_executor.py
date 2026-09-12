from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from experiments.paper4_5_agent.serve_hf_agent_pra import _completion
from pra_hf.agent_executor import (
    AgentHistoryLedger,
    HFAgentHistoryExecutor,
    _cache_bytes,
    _common_prefix,
    causal_message_spans,
    configure_append_stable_template,
    enforce_retention_floor,
    qwen3_append_stable_no_thinking_template,
    selected_record_plan,
    validate_append_stable_template,
)
from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_hf.hf_live_kv import HFSparseDynamicCache, select_dynamic_cache
from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan


class _Tokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.values: list[str] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_):
        text = "".join(
            f"<{row['role']}>{row.get('content', '')}</{row['role']}>"
            for row in messages
        )
        self.values.append(text)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]

    def decode(self, values, *, skip_special_tokens):
        return "".join(chr(int(value)) for value in values)


class _RoundTripTokenizer(_Tokenizer):
    """ASCII prompt codec whose arbitrary model tokens round-trip exactly."""

    chat_template = "stable-test-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_):
        text = "".join(
            f"<{row['role']}>{row.get('content', '')}</{row['role']}>"
            for row in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) + 1 for char in text]

    def decode(self, values, *, skip_special_tokens):
        return "".join(chr(int(value) - 1) for value in values)


def _manifest(messages):
    return [
        {
            "message_index": index,
            "role": row["role"],
            "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
        }
        for index, row in enumerate(messages)
    ]


def test_ledger_restores_history_but_never_tokenizes_selected_resource() -> None:
    tokenizer = _Tokenizer()
    ledger = AgentHistoryLedger()
    initial = (
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    )
    assert ledger.reconcile(PRAWireRequest(model="m", messages=initial)) == initial
    ledger.append_assistant("inspect")
    logical = (*initial, {"role": "assistant", "content": "inspect"}, {"role": "user", "content": "result"})
    request = PRAWireRequest(
        model="m",
        messages=(initial[0], initial[1], logical[-1]),
        resources=(PRAWireResource(
            resource_id="prior-action",
            uri="pra://history/action",
            text="inspect",
            metadata={"message_index": 2},
        ),),
        metadata={
            "history_projection": "live-agent-kv-v1",
            "logical_message_manifest": _manifest(logical),
            "mandatory_message_indices": [0, 1, 3],
        },
    )
    assert ledger.reconcile(request) == logical
    assert tokenizer.values == []


def test_record_plan_preserves_full_source_extent_and_original_positions() -> None:
    tokenizer = _Tokenizer()
    messages = (
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "inspect"},
        {"role": "user", "content": "output"},
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    source = len(prompt) - 8
    spans = causal_message_spans(tokenizer, messages, prompt, source_tokens=source)
    plan = selected_record_plan(
        tokenizer,
        messages,
        prompt,
        source_tokens=source,
        retention_fraction=0.9,
        mandatory_message_indices=(0, 1),
        selected_message_indices=(2,),
    )
    assert plan.source_position_base == source
    assert plan.selected_tokens < source
    assert tuple(plan.intervals) == tuple(
        row for row in spans if row.record_id.startswith(("message:0:", "message:1:", "message:2:"))
    )


def test_declared_retention_floor_rejects_underfilled_record_selection() -> None:
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


def test_full_retention_is_dense_semantic_noop_plan() -> None:
    tokenizer = _Tokenizer()
    messages = ({"role": "user", "content": "task"},)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    source = len(prompt) - 4
    plan = selected_record_plan(
        tokenizer,
        messages,
        prompt,
        source_tokens=source,
        retention_fraction=1.0,
        mandatory_message_indices=(0,),
        selected_message_indices=(),
    )
    assert plan == LiveKVSelectionPlan.full(source)


def test_owner_extension_fails_closed_on_any_history_rewrite() -> None:
    executor = object.__new__(HFAgentHistoryExecutor)
    state = SimpleNamespace(
        canonical_tokens=[1, 2, 3],
        canonical_cache=object(),
        selected_history_reencoded_tokens=0,
    )
    with pytest.raises(RuntimeError, match="append-stable template"):
        executor._ensure_owner(state, [1, 9, 3, 4])


def test_qwen_template_profile_has_frozen_digest() -> None:
    branch = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    tokenizer = SimpleNamespace(chat_template=f"a{branch}b{branch}c")
    stable = qwen3_append_stable_no_thinking_template(tokenizer.chat_template)
    digest = configure_append_stable_template(tokenizer, "qwen3-stable-no-thinking")
    assert tokenizer.chat_template == stable
    assert digest == hashlib.sha256(stable.encode()).hexdigest()
    assert stable.count("<think>\\n\\n</think>\\n\\n") == 2


def test_template_stability_preflight_accepts_append_only_rendering() -> None:
    validate_append_stable_template(_Tokenizer())


def test_template_stability_preflight_rejects_historical_rewrite() -> None:
    class RewritingTokenizer(_Tokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_):
            text = "active:" if add_generation_prompt else "historical:"
            return [ord(char) for char in text]

    with pytest.raises(RuntimeError, match="rewrites the active assistant prefix"):
        validate_append_stable_template(RewritingTokenizer())


def _canonical_cache(layers: int = 2, tokens: int = 3) -> DynamicCache:
    cache = DynamicCache()
    for layer in range(layers):
        keys = torch.arange(1 * 2 * tokens * 4, dtype=torch.float32).reshape(1, 2, tokens, 4)
        values = keys + 1000 + layer
        cache.update(keys + layer, values, layer)
    return cache


def test_sparse_suffix_graft_reports_dynamic_cache_copy_separately() -> None:
    canonical = _canonical_cache()
    plan = LiveKVSelectionPlan.full(3)
    sparse = select_dynamic_cache(canonical, plan).cache
    assert isinstance(sparse, HFSparseDynamicCache)
    for layer in range(2):
        keys = torch.full((1, 2, 2, 4), 7.0 + layer)
        values = torch.full((1, 2, 2, 4), 9.0 + layer)
        sparse.append(layer, keys, values, torch.tensor([3, 4]))

    before = _cache_bytes(canonical)
    executor = object.__new__(HFAgentHistoryExecutor)
    metrics = executor._graft_sparse_tail(canonical, sparse, source_tokens=3)

    assert metrics.local_tokens == 2
    assert metrics.request_tail_pack_d2d_bytes == 0
    assert metrics.canonical_suffix_graft_d2d_bytes == 2 * 2 * 1 * 2 * 2 * 4 * 4
    assert metrics.canonical_reallocation_d2d_bytes == before
    assert metrics.total_kv_copy_bytes == before + metrics.canonical_suffix_graft_d2d_bytes
    assert canonical.get_seq_length() == 5


def test_direct_completion_keeps_selection_copy_zero_and_graft_visible() -> None:
    request = PRAWireRequest(
        model="m", messages=({"role": "user", "content": "task"},), request_id="r"
    )
    trace = {
        "native_kv_used": True,
        "selected_kv_tokens": 90,
        "selected_history_kv_copy_bytes": 0,
        "canonical_suffix_graft_d2d_bytes": 4096,
        "total_kv_copy_bytes": 8192,
    }
    response = _completion(
        request,
        PRAEngineResult(
            text="done",
            raw={"native_attach_bytes": 0, "pra": trace},
            trace=(trace,),
        ),
    )
    assert response["pra"]["native_attach_bytes"] == 0
    assert response["pra"]["selected_history_kv_copy_bytes"] == 0
    assert response["pra"]["canonical_suffix_graft_d2d_bytes"] == 4096
    assert response["pra"]["total_kv_copy_bytes"] == 8192


def test_common_prefix_distinguishes_append_from_rewrite() -> None:
    assert _common_prefix([1, 2, 3], [1, 2, 3, 4]) == 3
    assert _common_prefix([1, 2, 3], [1, 9, 3]) == 1


def test_tiny_qwen_executes_dense_noop_then_sparse_without_history_reencoding() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(7)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=256,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
    ).eval()
    executor = HFAgentHistoryExecutor(
        model,
        _RoundTripTokenizer(),
        model_id="tiny-qwen2",
        model_revision="test",
        wire_tail_tokens=4,
        chat_template_profile="native",
    )
    initial = (
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "inspect-a"},
        {"role": "user", "content": "result-a"},
        {"role": "assistant", "content": "inspect-b"},
        {"role": "user", "content": "result-b"},
    )
    first = executor.generate(PRAWireRequest(
        model="tiny-qwen2",
        messages=initial,
        session_id="session",
        max_new_tokens=1,
        metadata={
            "history_projection": "live-agent-kv-v1",
            "logical_message_manifest": _manifest(initial),
            "mandatory_message_indices": list(range(len(initial))),
            "target_retention_fraction": 1.0,
        },
    ))
    assert first.trace[0]["consumption_mode"] == "dense_semantic_noop"

    logical = tuple(executor._sessions["session"].ledger.messages) + (
        {"role": "user", "content": "result-c"},
    )
    assistant_index = len(logical) - 2
    second = executor.generate(PRAWireRequest(
        model="tiny-qwen2",
        messages=(logical[0], logical[1], logical[-1]),
        resources=(PRAWireResource(
            resource_id="latest-assistant",
            uri="pra://history/latest-assistant",
            text=first.text,
            metadata={"message_index": assistant_index},
        ),),
        session_id="session",
        max_new_tokens=1,
        metadata={
            "history_projection": "live-agent-kv-v1",
            "logical_message_manifest": _manifest(logical),
            "mandatory_message_indices": [0, 1, len(logical) - 1],
            "target_retention_fraction": 0.9,
            "selection_contract": "arbitrary-subset-mechanism-probe",
        },
    ))
    trace = second.trace[0]
    assert trace["consumption_mode"] == "sparse_original_position"
    assert trace["selected_history_reencoded_tokens"] == 0
    assert trace["selected_history_kv_copy_bytes"] == 0
    assert trace["selected_kv_tokens"] < trace["source_tokens"]
    state = executor._sessions["session"]
    assert state.canonical_cache.get_seq_length() == len(state.canonical_tokens)
