"""Contract tests for the Paper 6.5 M1 causal tool-use gate."""

from __future__ import annotations

import torch

from data.agent_resources import generate_agent_catalog
from data.agent_tool_language import (
    compact_definition,
    compact_query,
    expected_answer,
    expected_call,
    formatted_argument,
    make_tool_examples,
    parse_call,
    render_supervised_trajectory,
    schema_code,
)
from data.tokenizer import PRATokenizer
from experiments.paper6_5_tools.run_m1_toy_model import (
    _publish_memory,
    _supervised_tensors,
    build_model,
)


def test_opaque_identity_is_available_only_through_the_definition():
    resource = generate_agent_catalog(8, seed=11).resources[0]
    example = make_tool_examples((resource,), seed=3, count=1, prefix="unit")[0]
    assert resource.name in compact_definition(resource)
    assert schema_code(resource) in compact_definition(resource)
    assert resource.name not in compact_query(example)
    assert schema_code(resource) not in compact_query(example)
    assert expected_call(example) == (
        f"@0|{schema_code(resource)}|{formatted_argument(example)}\n"
    )


def test_supervision_masks_context_and_observation_tokens():
    resource = generate_agent_catalog(8, seed=11).resources[0]
    example = make_tool_examples((resource,), seed=3, count=1, prefix="unit")[0]
    text, spans = render_supervised_trajectory(
        example, direct_definitions=(resource,)
    )
    tokenizer = PRATokenizer()
    inputs, labels = _supervised_tensors(tokenizer, text, spans, torch.device("cpu"))
    supervised = labels[labels >= 0].tolist()
    expected = tokenizer.encode(expected_call(example) + expected_answer(example))
    assert supervised == expected
    assert inputs.shape == labels.shape


def test_call_parser_rejects_malformed_payloads():
    assert parse_call("@0|raw|x03\n") == ("@0", "raw", "x03")
    assert parse_call("tool-only\n") is None
    assert parse_call("a|b|c|d\n") is None


def test_native_kv_definition_reaches_the_declared_pra_layer():
    tokenizer = PRATokenizer()
    model = build_model(tokenizer, seed=7, max_seq_len=256, device=torch.device("cpu"))
    resource = generate_agent_catalog(8, seed=11).resources[0]
    example = make_tool_examples((resource,), seed=3, count=1, prefix="unit")[0]
    memory_tokens, _ = _publish_memory(
        model, tokenizer, resource, torch.device("cpu")
    )
    prompt = torch.tensor([tokenizer.encode(compact_query(example))])
    logits = model(prompt, use_pra_memory=True)
    diagnostics = model.pra_diagnostics_by_layer()[1]
    assert logits.shape[:2] == prompt.shape
    assert memory_tokens == len(tokenizer.encode(compact_definition(resource)))
    assert diagnostics["memory_tokens_materialized"] == memory_tokens
    assert model.selected_chunks_by_layer()[1][0][0].reference_uri == resource.uri
