"""Causal controls for matched LocalSA and layered PRA experiments."""

import torch

from pra_torch.controlled_local_sa import (
    ControlledTokenizer,
    collate_controlled,
    controlled_examples,
    make_controlled_example,
)
from pra_torch.config import PRAConfig
from pra_torch.masks import causal_attention_mask
from pra_torch.model import (
    PositionAwareTransformerBlock,
    PRATransformerBlock,
    TinyPRAModel,
    convert_sa_model_to_pra,
)
from experiments.paper2_5_iterative_pra.run_controlled_pra import _pra_patterns


def _config(**updates) -> PRAConfig:
    values = {
        "vocab_size": 41,
        "d_model": 24,
        "n_heads": 3,
        "n_layers": 4,
        "d_ff": 48,
        "max_seq_len": 12,
        "model_variant": "td_sa",
        "position_encoding": "rope",
        "dropout": 0.0,
        "device": "cpu",
    }
    values.update(updates)
    return PRAConfig(**values)


def test_local_causal_mask_has_exact_window_and_global_parity():
    global_mask = causal_attention_mask(5, "cpu")
    expected_global = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    torch.testing.assert_close(global_mask, expected_global)

    local = causal_attention_mask(5, "cpu", window=2)
    visible = ~local
    assert visible[0].nonzero().flatten().tolist() == [0]
    assert visible[2].nonzero().flatten().tolist() == [1, 2]
    assert visible[4].nonzero().flatten().tolist() == [3, 4]


def test_local_receptive_field_grows_one_window_per_layer():
    torch.manual_seed(17)
    cfg = _config(n_layers=2, self_attention_window=2)
    blocks = [PositionAwareTransformerBlock(cfg, index).eval() for index in range(2)]
    hidden = torch.randn(1, 5, cfg.d_model, requires_grad=True)
    positions = torch.arange(5)

    first = blocks[0](hidden, position_ids=positions)
    first[0, -1].sum().backward(retain_graph=True)
    assert torch.count_nonzero(hidden.grad[0, 2]).item() == 0
    assert torch.count_nonzero(hidden.grad[0, 3]).item() > 0

    hidden.grad.zero_()
    second = blocks[1](first, position_ids=positions)
    second[0, -1].sum().backward()
    assert torch.count_nonzero(hidden.grad[0, 2]).item() > 0
    assert torch.count_nonzero(hidden.grad[0, 1]).item() == 0


def test_local_attention_right_padding_does_not_create_nan_rows():
    torch.manual_seed(19)
    cfg = _config(n_layers=1, self_attention_window=2)
    model = TinyPRAModel(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
    )
    logits = model(input_ids, attention_mask=attention_mask)
    assert torch.isfinite(logits).all()


def test_layered_pra_places_only_requested_interventions():
    cfg = _config(
        model_variant="td_layered_pra",
        pra_layer_ids=(1, 3),
        self_attention_window=4,
    )
    model = TinyPRAModel(cfg)

    assert [isinstance(block, PRATransformerBlock) for block in model.blocks] == [
        False,
        True,
        False,
        True,
    ]
    assert all(
        getattr(block, "self_attention_window", cfg.self_attention_window)
        == cfg.self_attention_window
        for block in model.blocks
        if isinstance(block, PositionAwareTransformerBlock)
    )


def test_pra_spacing_patterns_are_in_bounds_and_increasing():
    patterns = _pra_patterns(6)
    assert patterns["spacing_1"] == (0, 1, 2, 3, 4, 5)
    assert patterns["iterative_matched"] == (0, 2, 4, 5)
    assert patterns["spacing_2"] == (0, 2, 4)
    assert patterns["spacing_4"] == (0, 4)
    assert patterns["spacing_8"] == (0,)
    for layers in patterns.values():
        assert tuple(sorted(set(layers))) == layers
        assert all(0 <= layer < 6 for layer in layers)


def test_local_sa_conversion_preserves_disabled_memory_logits():
    torch.manual_seed(23)
    source_cfg = _config(self_attention_window=4)
    source = TinyPRAModel(source_cfg).eval()
    target_cfg = _config(
        model_variant="td_layered_pra",
        pra_layer_ids=(1, 3),
        self_attention_window=4,
    )
    converted = convert_sa_model_to_pra(source, target_cfg).eval()
    input_ids = torch.randint(0, source_cfg.vocab_size, (2, 9))

    with torch.no_grad():
        expected = source(input_ids)
        actual = converted(input_ids, use_pra_memory=False)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_invalid_local_window_and_layered_interventions_are_rejected():
    for value in (0, -1):
        try:
            _config(self_attention_window=value)
        except ValueError as error:
            assert "self_attention_window" in str(error)
        else:
            raise AssertionError("Invalid local window was accepted.")

    for layers in ((), (4,), (-1,)):
        try:
            _config(model_variant="td_layered_pra", pra_layer_ids=layers)
        except ValueError as error:
            assert "pra_layer" in str(error)
        else:
            raise AssertionError("Invalid layered PRA placement was accepted.")


def test_controlled_examples_are_reproducible_without_fixed_answer_mapping():
    tokenizer = ControlledTokenizer(entity_count=64)
    first = controlled_examples(tokenizer, count=8, seed=41)
    second = controlled_examples(tokenizer, count=8, seed=41)
    assert first == second
    assert len({example.answer_id for example in first}) > 1
    terminal_positions = []
    for example in first:
        assert len(example.target_reference_uris) == example.depth
        evidence_by_hop = {
            ref.hop: ref.uri for ref in example.references if ref.is_evidence
        }
        assert example.target_reference_uris == tuple(
            evidence_by_hop[hop] for hop in range(example.depth)
        )
        assert example.answer_id not in example.query_input_ids
        label_relation = 15
        label_facts = [
            ref for ref in example.references if ref.relation == label_relation
        ]
        assert len(label_facts) >= 3
        assert sum(ref.is_evidence for ref in label_facts) == 1
        evidence_hops = [ref.hop for ref in example.references if ref.is_evidence]
        assert evidence_hops == sorted(evidence_hops, reverse=True)
        terminal_positions.append(
            next(index for index, ref in enumerate(example.references) if ref.hop == example.depth - 1)
        )
    assert len(set(terminal_positions)) > 1


def test_controlled_collator_preserves_each_unpadded_sequence_length():
    tokenizer = ControlledTokenizer(entity_count=64)
    examples = [
        make_controlled_example(
            tokenizer,
            seed=seed,
            depth=depth,
            distractor_count=2,
            evidence_gap=depth,
            lexical_overlap=0.5,
            relation_types=4,
            branching=1,
        )
        for seed, depth in ((11, 1), (12, 3))
    ]
    input_ids, attention_mask, answers = collate_controlled(
        examples,
        pad_token_id=tokenizer.pad_token_id,
    )
    assert input_ids.shape == attention_mask.shape
    assert answers.shape == (2,)
    for row, example in enumerate(examples):
        assert int(attention_mask[row].sum()) == len(example.full_input_ids)


def test_progressive_pra_uses_evolved_state_without_reference_replay():
    tokenizer = ControlledTokenizer(entity_count=32, relation_count=4)
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=3,
        d_ff=64,
        max_seq_len=64,
        model_max_context_tokens=64,
        model_variant="td_layered_pra",
        pra_layer_ids=(0, 2),
        self_attention_window=8,
        position_encoding="rope",
        top_k_references=1,
        top_k_chunks_per_reference=1,
        trigger_threshold=-1.0,
        collect_routing_metrics=True,
    )
    model = TinyPRAModel(cfg).eval()
    for uri, text in (
        ("controlled://a", "[FACT] E000 R00 E001 [SEP]"),
        ("controlled://b", "[FACT] E001 R01 E002 [SEP]"),
    ):
        model.pra_cache.put(
            model.encode_reference_to_cache(uri, text, tokenizer, "cpu")
        )
    query = torch.tensor(
        [tokenizer.encode("[BOS] [Q] E000 R00 R01 [A]")],
        dtype=torch.long,
    )
    logits, trace = model.forward_progressive_pra(query)
    assert logits.shape == (1, query.shape[1], tokenizer.vocab_size)
    assert [row["layer_id"] for row in trace] == [0, 2]
    selected = [uri for row in trace for uri in row["selected_reference_uris"]]
    assert len(selected) == len(set(selected))
    assert all(not row["replayed_reference_uris"] for row in trace)
    assert not torch.equal(trace[0]["query_state"], trace[1]["query_state"])
    assert len(model.pra_cache.all_entries()) == 2
