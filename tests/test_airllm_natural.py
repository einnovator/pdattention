from __future__ import annotations

from experiments.paper6_6_airllm.run_cuda_natural import quality, select_entries


def test_airllm_natural_quality_is_bounded_and_token_aware() -> None:
    exact, f1, containment = quality("The answer is Cyan-Orbit-47.", "CYAN ORBIT 47")

    assert exact == 0.0
    assert 0.0 < f1 < 1.0
    assert containment == 1.0


def test_airllm_natural_cohort_is_bounded_per_dataset() -> None:
    manifest = {
        "entries": [
            {"dataset": "qasper", "example_id": "q1"},
            {"dataset": "qasper", "example_id": "q2"},
            {"dataset": "hotpotqa", "example_id": "h1"},
        ]
    }

    selected = select_entries(manifest, ("qasper", "hotpotqa"), 1)

    assert [entry["example_id"] for entry in selected] == ["q1", "h1"]
