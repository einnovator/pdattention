import pytest
import torch

from pra_torch.config import PRAConfig
from pra_torch.controlled_local_sa import (
    ControlledTokenizer,
    collate_controlled,
    controlled_examples,
)
from pra_torch.model import TinyPRAModel
from pra_torch.pra_aware_training import (
    build_controlled_memory_batch,
    forward_with_differentiable_memory,
    install_adaptation_regime,
    parameter_summary,
)


def _model(tokenizer: ControlledTokenizer) -> TinyPRAModel:
    return TinyPRAModel(
        PRAConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=4,
            n_layers=3,
            d_ff=64,
            max_seq_len=128,
            model_variant="td_layered_pra",
            pra_layer_ids=(1,),
            memory_transport="native_kv",
            position_encoding="rope",
            collect_per_head_metrics=True,
            device="cpu",
        )
    )


def test_controlled_memory_conditions_preserve_blinded_token_roles():
    tokenizer = ControlledTokenizer()
    examples = controlled_examples(tokenizer, count=3, seed=71, depths=(2,))

    evidence = build_controlled_memory_batch(
        examples,
        condition="evidence_only",
        pad_token_id=tokenizer.pad_token_id,
    )
    parent = build_controlled_memory_batch(
        examples,
        condition="whole_parent",
        pad_token_id=tokenizer.pad_token_id,
    )
    distractor = build_controlled_memory_batch(
        examples,
        condition="matched_distractor",
        pad_token_id=tokenizer.pad_token_id,
    )

    assert all(length == 10 for length in evidence.lengths)
    assert evidence.evidence_mask[evidence.attention_mask.bool()].all()
    assert all(parent_length >= evidence_length for parent_length, evidence_length in zip(parent.lengths, evidence.lengths))
    assert not distractor.evidence_mask.any()
    assert distractor.lengths == evidence.lengths


def test_native_memory_forward_retains_reference_gradients_and_metrics():
    torch.manual_seed(3)
    tokenizer = ControlledTokenizer()
    examples = controlled_examples(tokenizer, count=2, seed=19, depths=(2,))
    model = _model(tokenizer)
    query_ids, query_mask, answers = collate_controlled(
        examples,
        pad_token_id=tokenizer.pad_token_id,
        query_only=True,
    )
    memory = build_controlled_memory_batch(
        examples,
        condition="whole_parent",
        pad_token_id=tokenizer.pad_token_id,
    )

    output = forward_with_differentiable_memory(
        model,
        query_ids,
        memory,
        attention_mask=query_mask,
    )
    logits = output.logits[torch.arange(len(examples)), query_mask.sum(dim=1) - 1]
    torch.nn.functional.cross_entropy(logits, answers).backward()

    attention = model.blocks[1].attn
    assert attention.k_proj.weight.grad is not None
    assert attention.v_proj.weight.grad is not None
    assert attention.q_proj.weight.grad is not None
    metrics = output.layer_metrics[1]
    assert metrics["memory_tokens_materialized"] == pytest.approx(sum(memory.lengths))
    assert metrics["evidence_attention_mass"] > 0.0
    assert metrics["distractor_attention_mass"] > 0.0


@pytest.mark.parametrize(
    ("regime", "expected_nonzero"),
    [
        ("frozen", False),
        ("consumer_lora", True),
        ("interface_lora", True),
        ("broad_lora", True),
        ("full_weight", True),
        ("native_scratch", True),
    ],
)
def test_adaptation_regimes_expose_declared_parameter_fraction(regime, expected_nonzero):
    tokenizer = ControlledTokenizer()
    model = _model(tokenizer)
    installed = install_adaptation_regime(model, regime, lora_rank=2, lora_alpha=4)
    summary = parameter_summary(model)

    assert bool(summary["trainable_parameters"]) is expected_nonzero
    assert 0.0 <= summary["trainable_fraction"] <= 1.0
    if "lora" in regime:
        assert installed
        assert all(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if "lora_" in name
        )


def test_lora_installation_is_function_preserving():
    torch.manual_seed(11)
    tokenizer = ControlledTokenizer()
    model = _model(tokenizer).eval()
    examples = controlled_examples(tokenizer, count=2, seed=29)
    query_ids, query_mask, _ = collate_controlled(
        examples,
        pad_token_id=tokenizer.pad_token_id,
        query_only=True,
    )
    memory = build_controlled_memory_batch(
        examples,
        condition="whole_parent",
        pad_token_id=tokenizer.pad_token_id,
    )
    before = forward_with_differentiable_memory(
        model, query_ids, memory, attention_mask=query_mask
    ).logits

    install_adaptation_regime(model, "interface_lora", lora_rank=2, lora_alpha=4)
    after = forward_with_differentiable_memory(
        model, query_ids, memory, attention_mask=query_mask
    ).logits

    torch.testing.assert_close(after, before)
