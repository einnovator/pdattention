import json

from pra_hf.natural_reasoning_graph import (
    char_spans_to_token_spans,
    map_example_to_parents,
    parse_2wiki_row,
    parse_musique_row,
    stable_partition,
)
from experiments.paper2_5_iterative_pra.run_natural_graph_depth import (
    _shortest_native_path,
    _transition_rank,
)
import torch


def test_musique_preserves_explicit_hops_and_dependencies():
    row = {
        "id": "3hop__1_2_3",
        "question": "final question",
        "answer": "answer",
        "answerable": True,
        "paragraphs": [
            {"idx": i, "title": f"T{i}", "paragraph_text": text, "is_supporting": True}
            for i, text in enumerate(("alpha", "beta", "gamma"))
        ],
        "question_decomposition": [
            {"id": 1, "question": "start", "answer": "a", "paragraph_support_idx": 0},
            {"id": 2, "question": "#1 next", "answer": "b", "paragraph_support_idx": 1},
            {"id": 3, "question": "combine #1 and #2", "answer": "c", "paragraph_support_idx": 2},
        ],
    }
    example = parse_musique_row(row)
    assert example.annotated_hops == 3
    assert example.annotated_edges == (("1", "2"), ("1", "3"), ("2", "3"))
    assert example.graph_type == "convergent"
    assert example.root_node_ids == ("1",)
    assert example.raw_annotation["question_decomposition"] == row["question_decomposition"]


def test_2wiki_uses_evidence_ids_and_maps_support_sentences():
    row = {
        "_id": "sample",
        "type": "compositional",
        "question": "Who is the mother of the director?",
        "answer": "C",
        "context": [
            ["Film (film)", ["A filler.", "Film was directed by B."]],
            ["B", ["Filler.", "More filler.", "B's mother was C."]],
        ],
        "supporting_facts": [["Film (film)", 1], ["B", 2]],
        "evidences": [["Film", "director", "B"], ["B", "mother", "C"]],
        "evidences_id": [["Q1", "director", "Q2"], ["Q2", "mother", "Q3"]],
    }
    example = parse_2wiki_row(row)
    assert example.annotated_edges == (("0", "1"),)
    assert all(node.text_span is not None for node in example.nodes)
    assert example.raw_annotation["edge_semantics"] == "dataset_entity_id_exact_join"


def test_2wiki_does_not_invent_edges_for_comparison():
    row = {
        "_id": "comparison",
        "type": "comparison",
        "question": "Which came first?",
        "answer": "A",
        "context": [["A", ["A was released in 2000."]], ["B", ["B was released in 2001."]]],
        "supporting_facts": [["A", 0], ["B", 0]],
        "evidences": [["A", "publication date", "2000"], ["B", "publication date", "2001"]],
        "evidences_id": [],
    }
    example = parse_2wiki_row(row)
    assert example.annotated_edges == ()
    assert example.graph_type == "independent"


def test_parent_mapping_separates_preserved_collapsed_and_unmappable_edges():
    row = {
        "id": "3hop__1_2_3",
        "question": "q",
        "answer": "a",
        "paragraphs": [
            {"idx": 0, "title": "A", "paragraph_text": "aaaa", "is_supporting": True},
            {"idx": 1, "title": "B", "paragraph_text": "bbbb", "is_supporting": True},
            {"idx": 2, "title": "C", "paragraph_text": "cccc", "is_supporting": True},
        ],
        "question_decomposition": [
            {"id": 1, "question": "start", "answer": "a", "paragraph_support_idx": 0},
            {"id": 2, "question": "#1", "answer": "b", "paragraph_support_idx": 1},
            {"id": 3, "question": "#2", "answer": "c", "paragraph_support_idx": 2},
        ],
    }
    example = parse_musique_row(row)
    mapping = map_example_to_parents(
        example,
        12,
        {"1": (0, 2), "2": (2, 4)},
        chunk_size=3,
    )
    assert mapping.preserved_edges == ((0, 1),)
    assert mapping.unmappable_node_edges == (("2", "3"),)

    collapsed = map_example_to_parents(
        example,
        12,
        {"1": (0, 1), "2": (1, 2), "3": (6, 7)},
        chunk_size=3,
    )
    assert collapsed.collapsed_node_edges == (("1", "2"),)
    assert collapsed.preserved_edges == ((0, 2),)


def test_character_offsets_map_without_text_search():
    row = {
        "id": "2hop__1_2",
        "question": "q",
        "answer": "a",
        "paragraphs": [
            {"idx": 0, "title": "A", "paragraph_text": "alpha", "is_supporting": True},
            {"idx": 1, "title": "B", "paragraph_text": "beta", "is_supporting": True},
        ],
        "question_decomposition": [
            {"id": 1, "question": "start", "answer": "a", "paragraph_support_idx": 0},
            {"id": 2, "question": "#1", "answer": "b", "paragraph_support_idx": 1},
        ],
    }
    example = parse_musique_row(row)
    offsets = [(i, i + 1) for i in range(len(example.source))]
    spans = char_spans_to_token_spans(offsets, example.nodes)
    assert example.source[slice(*spans["1"])] == "alpha"
    assert example.source[slice(*spans["2"])] == "beta"


def test_long_root_uses_one_representative_but_keeps_all_oracle_parents():
    row = {
        "id": "2hop__1_2",
        "question": "q",
        "answer": "a",
        "paragraphs": [
            {"idx": 0, "title": "A", "paragraph_text": "long root", "is_supporting": True},
            {"idx": 1, "title": "B", "paragraph_text": "target", "is_supporting": True},
        ],
        "question_decomposition": [
            {"id": 1, "question": "start", "answer": "a", "paragraph_support_idx": 0},
            {"id": 2, "question": "#1", "answer": "b", "paragraph_support_idx": 1},
        ],
    }
    example = parse_musique_row(row)
    mapping = map_example_to_parents(
        example, 12, {"1": (0, 7), "2": (8, 10)}, chunk_size=3
    )
    assert mapping.node_parent_groups["1"] == (0, 1, 2)
    assert mapping.root_parent_ids == (0,)
    assert set(mapping.oracle_parent_ids) == {0, 1, 2, 3}


def test_identity_partition_is_deterministic_and_label_free():
    assert stable_partition("same-id") == stable_partition("same-id")
    assert stable_partition.__code__.co_argcount == 1


def test_native_transition_metrics_use_frozen_scores_only():
    scores = torch.tensor(
        [
            [float("-inf"), 0.9, 0.8, 0.1],
            [0.1, float("-inf"), 0.9, 0.2],
            [0.1, 0.2, float("-inf"), 0.9],
            [0.1, 0.2, 0.3, float("-inf")],
        ]
    )
    assert _transition_rank(scores, (0,), (2,)) == 2
    assert _shortest_native_path(scores, (0,), (2,), k=1) == 2
    assert _shortest_native_path(scores, (0,), (3,), k=1) == 3
