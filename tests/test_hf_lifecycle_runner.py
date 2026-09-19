from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.paper4_5_agent.run_hf_live_agent_kv_lifecycle import (
    _bounded_prefill,
    _last_logit_kwargs,
    _require_live_kv_api,
)


class LogitsToKeepModel:
    def forward(self, input_ids, *, logits_to_keep=0):
        return input_ids, logits_to_keep


class NumLogitsToKeepModel:
    def forward(self, input_ids, *, num_logits_to_keep=0):
        return input_ids, num_logits_to_keep


class LegacyLiveKVRequest:
    def generate(self, model, tail_input_ids, *, max_new_tokens):
        return model, tail_input_ids, max_new_tokens


def test_hf_lifecycle_detects_last_logit_controls() -> None:
    assert _last_logit_kwargs(LogitsToKeepModel()) == {"logits_to_keep": 1}
    assert _last_logit_kwargs(NumLogitsToKeepModel()) == {
        "num_logits_to_keep": 1
    }


def test_hf_lifecycle_bounded_prefill_rejects_invalid_inputs() -> None:
    model = LogitsToKeepModel()
    with pytest.raises(ValueError, match="positive"):
        _bounded_prefill(model, [1], torch.device("cpu"), step_size=0)
    with pytest.raises(ValueError, match="at least one token"):
        _bounded_prefill(model, [], torch.device("cpu"), step_size=1)


def test_hf_lifecycle_fails_fast_on_shadowed_legacy_live_kv_api() -> None:
    with pytest.raises(RuntimeError, match="materialized_history"):
        _require_live_kv_api(LegacyLiveKVRequest)


def test_hf_lifecycle_accepts_current_live_kv_api() -> None:
    assert Path(_require_live_kv_api()).name == "hf_live_kv.py"
