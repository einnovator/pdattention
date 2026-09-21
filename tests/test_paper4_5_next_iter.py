from __future__ import annotations

from experiments.paper4_5_agent.run_mlx_agent_cache_equivalence import (
    _assistant_prompts,
)
from experiments.paper4_5_agent.run_mlx_live_agent_kv_gate import (
    _qualification_passed,
)
from experiments.paper4_5_agent.sparse_gate_common import causal_message_spans
from experiments.paper4_5_runtime.build_product_matrix_v2 import build_matrix
from experiments.paper4_5_runtime.run_agent_transport_scaling import (
    run_scaling_workload,
)
from experiments.paper4_5_runtime.run_storage_policy_trace import run_policy


def test_product_matrix_builder_normalizes_existing_evidence() -> None:
    matrix = build_matrix()

    assert len(matrix.rows) >= 20
    assert {row.model_family.lower() for row in matrix.rows} >= {"qwen", "llama", "gemma"}
    assert all(not isinstance(row.ttft_ms, str) for row in matrix.rows)
    assert matrix.schema_version == "2.1"
    assert all(row.metric_statuses for row in matrix.rows)
    assert {row.integration_level for row in matrix.rows} >= {"E0", "E2"}
    assert {row.representation for row in matrix.rows} >= {
        "E0_SELECTED",
        "E2_HOT",
        "E2_WARM",
    }
    assert {
        row.engine
        for row in matrix.rows
        if row.workload.startswith("matched_e0_e2/")
    } == {"vllm", "sglang", "mlx"}
    airllm = [row for row in matrix.rows if row.engine == "airllm"]
    assert len(airllm) == 12
    assert {row.integration_level for row in airllm} == {"E0", "E1"}
    airllm_pairs = {
        (row.dataset, row.workload): row.exact_pair_parity
        for row in airllm
        if row.integration_level == "E0"
    }
    assert all(
        row.exact_pair_parity == airllm_pairs[(row.dataset, row.workload)]
        for row in airllm
        if row.integration_level == "E1"
    )
    assert max(row.exact_pair_parity or 0.0 for row in airllm) <= 0.1
    assert all(
        row.profile_status == "RESEARCH_ONLY"
        for row in airllm
        if row.integration_level == "E1"
    )
    assert all(
        row.representation == "E2_CANDIDATE_NATIVE"
        for row in airllm
        if row.integration_level == "E1"
    )
    openvino = [row for row in matrix.rows if row.engine == "openvino_genai"]
    assert len(openvino) == 45
    assert {row.integration_level for row in openvino} == {"E0"}
    assert {row.model_size for row in openvino} >= {500_000_000, 1_500_000_000}
    assert any(
        row.model_family == "tinyllama" and row.model_size == 1_100_000_000
        for row in openvino
    )
    assert all(row.gold_answer_log_probability is None for row in openvino)
    assert all("selector_frozen" in row.verified_invariants for row in openvino)


def test_long_transport_delta_survives_resync_and_updates() -> None:
    rows, summary = run_scaling_workload(20)

    assert len(rows) == 20
    assert summary["resynchronizations"] == 1
    assert summary["selection_parity"]
    assert summary["task_metadata_preserved"]
    assert summary["delta_resource_bodies"] < summary["full_resource_bodies"]
    assert summary["pra_delta_bytes"] < summary["text_bytes"]


def test_weighted_storage_trace_protects_shared_and_waiting_records() -> None:
    lru = run_policy("lru", turns=30)
    weighted = run_policy("weighted_lru", turns=30)

    assert weighted["task_protected_hit_rate"] >= lru["task_protected_hit_rate"]
    assert weighted["shared_document_hit_rate"] >= lru["shared_document_hit_rate"]
    assert weighted["utility_weighted_hit_rate"] >= lru["utility_weighted_hit_rate"]
    assert weighted["peak_warm_bytes"] <= 80_000
    assert weighted["wasted_write_bytes"] > 0


def test_mlx_gate_prompt_builder_forwards_frozen_template_kwargs() -> None:
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return [11, 12, 13]

    prompts = _assistant_prompts(
        Tokenizer(),
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        1,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert prompts == [[11, 12, 13]]
    assert calls == [
        (
            [{"role": "system", "content": "system"}],
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]


def test_sparse_span_builder_uses_the_same_template_kwargs_as_prompt() -> None:
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return list(range(1, len(messages) + 1))

    spans = causal_message_spans(
        Tokenizer(),
        [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "action"},
            {"role": "user", "content": "observation"},
        ],
        [1, 2, 3],
        source_tokens=3,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert sum(span.tokens for span in spans) == 3
    assert calls == [
        {
            "tokenize": True,
            "add_generation_prompt": False,
            "enable_thinking": False,
        }
    ] * 3


def test_mlx_sparse_gate_rejects_vacuous_exactness() -> None:
    assert not _qualification_passed(
        0.9, all_exact=True, sparse_position_gate_valid=False
    )
    assert not _qualification_passed(
        1.0, all_exact=False, sparse_position_gate_valid=False
    )
    assert _qualification_passed(
        0.9, all_exact=True, sparse_position_gate_valid=True
    )
