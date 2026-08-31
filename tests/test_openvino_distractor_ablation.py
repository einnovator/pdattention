from __future__ import annotations

from experiments.paper6_3_openvino.run_distractor_ablation import (
    document_blocks,
    ranked_distractors,
)


def test_document_blocks_preserve_portable_manifest_documents() -> None:
    blocks = document_blocks("Document: Alpha\nOne.\n\nDocument: Beta\nTwo.")

    assert blocks == ["Document: Alpha\nOne.", "Document: Beta\nTwo."]


def test_ranked_distractors_separate_query_overlap() -> None:
    entry = {
        "question": "Which city hosted the orbital summit?",
        "answer": "Lisbon",
        "distractor_source": (
            "Document: Weather\nRain fell in a village.\n\n"
            "Document: Summit\nThe orbital summit met in Porto."
        ),
    }

    relevant, irrelevant = ranked_distractors(entry)

    assert relevant[0].startswith("Document: Summit")
    assert irrelevant[0].startswith("Document: Weather")
