"""Protocol tests for the larger pretrained materialization confirmation."""

from types import SimpleNamespace

from experiments.paper3_kv_materialization import run_oracle_frontier as frontier
from pra_hf.natural_reasoning_graph import AnnotatedEvidenceNode, NaturalReasoningExample


def _example(dataset: str, identity: str, source: str = "abcdefghij"):
    return NaturalReasoningExample(
        dataset=dataset,
        example_id=identity,
        question="q",
        answer="a",
        question_type="2hop",
        annotated_hops=2,
        graph_type="chain",
        source=source,
        nodes=(
            AnnotatedEvidenceNode("0", (2, 5), (), {}),
            AnnotatedEvidenceNode("1", (8, 10), ("0",), {}),
        ),
        raw_annotation={},
    )


class _CharacterTokenizer:
    def __call__(self, text, **_kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_annotation_geometry_matches_reference_token_domain():
    geometry = frontier._annotation_geometry(
        _CharacterTokenizer(), _example("musique", "heldout")
    )
    assert geometry["source_tokens"] == 10
    assert geometry["evidence_token_spans"] == [(2, 5), (8, 10)]
    assert geometry["geometry_source"] == "dataset_annotation_character_spans"


def test_cohort_extension_preserves_manifest_order_and_partition_isolation(monkeypatch):
    musique = [_example("musique", identity) for identity in ("m0", "m1", "m2", "m3")]
    wiki = [
        _example("2wikimultihopqa", identity)
        for identity in ("w0", "w1", "w2", "w3")
    ]
    monkeypatch.setattr(frontier, "load_musique", lambda _path: musique)
    monkeypatch.setattr(frontier, "load_2wiki", lambda _path: wiki)
    discovery = [
        {"dataset": "musique", "partition": "validation", "example_id": "m0"},
        {"dataset": "musique", "partition": "test", "example_id": "m2"},
        {"dataset": "2wikimultihopqa", "partition": "validation", "example_id": "w0"},
        {"dataset": "2wikimultihopqa", "partition": "test", "example_id": "w2"},
    ]
    args = SimpleNamespace(
        phase="heldout",
        examples_per_dataset=3,
        musique_dev=None,
        twowiki_dev=None,
    )
    selected = frontier._examples(args, discovery)
    identities = [example.example_id for example in selected]
    assert identities == ["m2", "m1", "m3", "w2", "w1", "w3"]
    assert "m0" not in identities
    assert "w0" not in identities


def test_row_metrics_accept_shared_core_transport_names_for_whole_parents():
    diagnostics = {
        1: {
            "memory_tokens_materialized": 7,
            "retrieved_kv_transfer_bytes": 1024,
            "materialization_duration_seconds": 0.003,
            "selected_kv_transfer_duration_seconds": 0.002,
        },
        3: {
            "memory_tokens_materialized": 7,
            "retrieved_kv_transfer_bytes": 1024,
            "materialization_duration_seconds": 0.004,
            "selected_kv_transfer_duration_seconds": 0.001,
        },
    }
    metrics = frontier._row_metrics(diagnostics, (1, 3))
    assert metrics["materialized_unique_tokens"] == 7
    assert metrics["native_kv_token_states"] == 14
    assert metrics["native_kv_bytes"] == 2048
    assert metrics["logical_gather_seconds"] == 0.007
    assert metrics["logical_h2d_seconds"] == 0.003
