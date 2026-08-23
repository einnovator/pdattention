import torch

from pra_hf.natural_query_facets import (
    align_subquestions_to_units,
    annotation_from_2wiki,
    annotation_from_musique,
    evaluate_natural_partition,
    interleaving_statistics,
    scorable_labels,
)
from experiments.paper2_7_query_graph.run_llm_decomposition import _parse_subquestions


def test_2wiki_relation_chain_maps_to_auditable_facets():
    row = {
        "_id": "wiki-1",
        "type": "compositional",
        "question": "Who is the mother of the director of film Example?",
        "evidences": [
            ["Example", "director", "Person"],
            ["Person", "mother", "Answer"],
        ],
    }
    annotation = annotation_from_2wiki(row, split="test")
    labels, unit_ids = scorable_labels(annotation)
    assert set(labels.tolist()) == {0, 1}
    assert unit_ids.numel() >= 4
    assert all(annotation.question[u.char_start : u.char_end] == u.text for u in annotation.units)


def test_2wiki_comparison_groups_connected_evidence_into_query_branches():
    row = {
        "_id": "wiki-comparison",
        "type": "comparison",
        "question": "Which director died first, the director of Alpha Film or Beta Film?",
        "evidences": [
            ["Alpha Film", "director", "Alice Person"],
            ["Beta Film", "director", "Bob Person"],
            ["Alice Person", "date of death", "Date A"],
            ["Bob Person", "date of death", "Date B"],
        ],
    }
    annotation = annotation_from_2wiki(row, split="test")
    labels, _ = scorable_labels(annotation)
    assert len(annotation.source_facets) == 2
    assert set(labels.tolist()) == {0, 1}


def test_partial_llm_json_recovery_keeps_complete_subquestions():
    values, note = _parse_subquestions('{"subquestions":["Find the director.","Where was the director')
    assert values == ["Find the director."]
    assert note.startswith("partial_json_recovery")


def test_musique_decomposition_preserves_non_contiguous_capability():
    row = {
        "id": "musique-1",
        "question": "Who is the spouse of the Green performer?",
        "question_decomposition": [
            {"question": "Green >> performer"},
            {"question": "#1 >> spouse"},
        ],
    }
    annotation = annotation_from_musique(row, split="test")
    labels, _ = scorable_labels(annotation)
    metrics = interleaving_statistics(annotation)
    assert set(labels.tolist()) == {0, 1}
    assert metrics["normalized_switch_rate"] >= 0


def test_primary_metrics_exclude_shared_global_units():
    row = {
        "_id": "wiki-2",
        "type": "compositional",
        "question": "When did Example's father die?",
        "evidences": [
            ["Example", "father", "Parent"],
            ["Parent", "date of death", "Date"],
        ],
    }
    annotation = annotation_from_2wiki(row, split="test")
    prediction = torch.tensor([
        1 if unit.facet_id == 1 else 0 for unit in annotation.units
    ])
    result = evaluate_natural_partition(prediction, annotation)
    assert result["ari"] == 1.0
    assert result["pairwise_f1"] == 1.0


def test_generated_subquestions_are_predictions_not_gold():
    row = {
        "id": "musique-2",
        "question": "Who founded the company that distributed UHF?",
        "question_decomposition": [
            {"question": "UHF >> distributed by"},
            {"question": "#1 >> founded by"},
        ],
    }
    annotation = annotation_from_musique(row, split="test")
    labels = align_subquestions_to_units(
        annotation,
        ["Which company distributed UHF?", "Who founded that company?"],
    )
    assert labels.shape == (len(annotation.units),)
    assert set(labels.tolist()) == {0, 1}
