import random

from .datasets import DatasetExample
from .references import ReferenceTable


def make_synthetic_examples(n: int = 100) -> list[DatasetExample]:
    """Create random flat memory QA examples for quick smoke training."""
    examples = []
    colors = ["red", "blue", "green", "yellow", "purple"]
    animals = ["otter", "falcon", "tiger", "whale", "lynx"]
    for i in range(n):
        color = random.choice(colors)
        animal = random.choice(animals)
        code = f"{color}-{animal}-{i}"
        uri = f"mem://doc{i}#summary"
        full = (
            f"Document {i}. This document contains project notes. "
            f"The secret code is {code}. The owner is team alpha. "
            f"The rest of the document is filler text about systems and memory."
        )
        summary = f"Document {i} summary: contains a secret code and project notes."
        reference_table = ReferenceTable()
        handle = reference_table.register(uri, summary=summary)
        prompt = f"Question: what is the secret code in {handle.token}? Answer:"
        examples.append(
            DatasetExample(
                id=f"synthetic_{i}",
                prompt=prompt,
                target=f" {code}\n",
                refs={uri: full},
                summaries={uri: summary},
                reference_table=reference_table,
                expected_ref_ids=[handle.id],
            )
        )
    return examples
