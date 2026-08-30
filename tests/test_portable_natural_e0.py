from __future__ import annotations

from experiments.engine_serving.run_openai_natural_e0 import _messages, _quality


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(token_ids)


def _entry():
    return {
        "question": "Where is the answer?",
        "selected_source": "alpha beta answer gamma",
        "distractor_source": "delta epsilon zeta eta theta",
    }


def test_quality_reports_exact_f1_and_containment_separately():
    assert _quality("Paris", "Paris") == (1.0, 1.0, 1.0)
    exact, f1, containment = _quality("The answer is Paris.", "Paris")
    assert exact == 0.0
    assert 0.0 < f1 < 1.0
    assert containment == 1.0


def test_messages_bound_selected_and_full_source_independently():
    selected, selected_tokens, raw_selected_tokens = _messages(
        _entry(), _Tokenizer(), "selected_context", selected_limit=3, full_limit=6
    )
    full, full_tokens, _ = _messages(
        _entry(), _Tokenizer(), "full_context", selected_limit=3, full_limit=6
    )
    assert selected_tokens == raw_selected_tokens == 3
    assert full_tokens == 6
    assert "alpha beta answer" in selected[0]["content"]
    assert "delta epsilon zeta" in full[0]["content"]


def test_no_context_has_zero_source_tokens():
    messages, source_tokens, selected_tokens = _messages(
        _entry(), _Tokenizer(), "no_context", selected_limit=3, full_limit=6
    )
    assert source_tokens == 0
    assert selected_tokens == 3
    assert "Evidence:" not in messages[0]["content"]
