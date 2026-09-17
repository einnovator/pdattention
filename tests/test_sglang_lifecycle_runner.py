from __future__ import annotations

import pytest

from experiments.paper4_5_agent.run_sglang_live_agent_kv_lifecycle import _prefill


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int], bool]] = []

    def prefill_start(
        self,
        request_id,
        input_ids,
        _origin_ids,
        _sampling,
        _stop,
        _offset,
        *,
        needs_logits,
    ):
        self.calls.append(("prefill", list(input_ids), bool(needs_logits)))
        return ("prefill", len(self.calls))

    def prefill_finalize(self, pending):
        return pending[1]

    def extend_start(self, request_id, input_ids, _sampling, *, needs_logits):
        self.calls.append(("extend", list(input_ids), bool(needs_logits)))
        return ("extend", len(self.calls))

    def extend_finalize(self, pending):
        return pending[1]

    @staticmethod
    def eval_pending(_pending) -> None:
        return None


def test_lifecycle_prefill_chunks_and_requests_only_final_logits() -> None:
    runner = FakeRunner()
    token = _prefill(runner, "request", [1, 2, 3, 4, 5], step_size=2)

    assert token == 3
    assert runner.calls == [
        ("prefill", [1, 2], False),
        ("extend", [3, 4], False),
        ("extend", [5], True),
    ]


def test_lifecycle_prefill_rejects_invalid_inputs() -> None:
    runner = FakeRunner()
    with pytest.raises(ValueError, match="positive"):
        _prefill(runner, "request", [1], step_size=0)
    with pytest.raises(ValueError, match="non-empty"):
        _prefill(runner, "request", [], step_size=1)
