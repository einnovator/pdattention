"""Correctness tests for the experiment-only global readout controls."""

import torch
import torch.nn as nn

from experiments.paper2_hf.qa.run_lm_head_control import LMHeadLoRA, _clone_output_head


def test_lm_head_lora_starts_as_exact_frozen_readout_and_owns_only_factors():
    torch.manual_seed(301)
    base = nn.Linear(12, 19, bias=False).eval()
    base.weight.requires_grad_(False)
    adapted = LMHeadLoRA(base, 4, alpha=4.0)
    hidden = torch.randn(2, 5, 12)

    expected = base(hidden).float()
    actual = adapted(hidden)

    assert torch.equal(actual, expected)
    assert sum(parameter.numel() for parameter in adapted.parameters()) == 4 * (12 + 19)
    actual.square().mean().backward()
    assert base.weight.grad is None
    assert adapted.up.weight.grad is not None


def test_full_lm_head_clone_is_untied_but_initially_exact():
    torch.manual_seed(302)
    base = nn.Linear(12, 19, bias=False).to(dtype=torch.float16).eval()
    cloned = _clone_output_head(base)
    hidden = torch.randn(2, 5, 12, dtype=torch.float16)

    assert cloned.weight.data_ptr() != base.weight.data_ptr()
    assert torch.equal(cloned.weight, base.weight)
    assert torch.equal(cloned(hidden), base(hidden))
    assert cloned.weight.requires_grad is True
