"""Shared deterministic fixtures and artifact helpers for Paper 2.7."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from pra_hf.query_graph import QueryUnitProvenance, lexical_feature_matrix


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "f1e085de0d82f0c996bc6631effc104cb33f9925"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
VALIDATION_SEEDS = (11, 23, 37, 53, 71)
TEST_SEEDS = (101, 113, 127, 139, 151)
ENTITY_WORDS = ("alder", "beryl", "cinder", "dorian", "elm", "faron")
RELATION_WORDS = ("origin", "period", "author", "region", "material", "cause")


@dataclass(frozen=True)
class ControlledQuery:
    """One synthetic query with known latent retrieval facets."""

    example_id: str
    split: str
    seed: int
    hidden: torch.Tensor
    lexical: torch.Tensor
    target_labels: torch.Tensor
    provenance: tuple[QueryUnitProvenance, ...]
    facet_count: int
    interleaved: bool
    shared_entity: bool
    distractor_units: int


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    return {
        "paper2_6_base_commit": BASE_COMMIT,
        "execution_commit": commit,
        "branch": branch,
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
    }


def resolve_artifact(relative: str) -> Path:
    """Find an inherited artifact locally or in the Paper 2.5 worktree."""

    local = ROOT / relative
    if local.exists():
        return local
    inherited = Path(r"D:/git/rd/pdattention-iter-gist") / relative
    if inherited.exists():
        return inherited
    raise FileNotFoundError(f"Inherited artifact is unavailable: {relative}")


def controlled_queries(
    split: str,
    *,
    examples_per_seed: int = 24,
    width: int = 48,
) -> list[ControlledQuery]:
    """Generate paraphrased/permuted compositional query-unit fixtures.

    Facet identities affect latent semantic and repeated-entity structure, but
    neither punctuation nor absolute order encodes the target partition.
    """

    seeds = VALIDATION_SEEDS if split == "validation" else TEST_SEEDS
    output = []
    for seed in seeds:
        rng = random.Random(seed)
        generator = torch.Generator().manual_seed(seed)
        centers = F.normalize(torch.randn((6, width), generator=generator), dim=-1)
        shared = F.normalize(torch.randn(width, generator=generator), dim=-1)
        for example_index in range(examples_per_seed):
            facet_count = 1 + (example_index % 6)
            interleaved = bool((example_index // 2) % 2)
            shared_entity = bool((example_index // 3) % 2)
            units = []
            for facet in range(facet_count):
                unit_count = 3 + ((seed + example_index + facet) % 3)
                for local_index in range(unit_count):
                    noise = 0.10 * torch.randn(width, generator=generator)
                    hidden = F.normalize(
                        0.82 * centers[facet] + 0.18 * shared + noise,
                        dim=0,
                    )
                    entity = ENTITY_WORDS[facet]
                    if shared_entity and local_index == unit_count - 1:
                        text = "shared"
                    elif local_index % 2 == 0:
                        text = entity
                    else:
                        text = RELATION_WORDS[facet]
                    units.append((facet, hidden, text))
            distractor_units = int(facet_count > 2 and example_index % 4 == 0)
            if distractor_units:
                target = rng.randrange(facet_count)
                units.append(
                    (
                        target,
                        F.normalize(0.60 * centers[target] + 0.40 * shared, dim=0),
                        "shared",
                    )
                )
            if interleaved:
                rng.shuffle(units)
            else:
                groups = list(range(facet_count))
                rng.shuffle(groups)
                units.sort(key=lambda row: groups.index(row[0]))
            labels = torch.tensor([row[0] for row in units], dtype=torch.long)
            hidden = torch.stack([row[1] for row in units])
            texts = [row[2] for row in units]
            lexical = lexical_feature_matrix(texts, buckets=128)
            provenance = tuple(
                QueryUnitProvenance(
                    unit_id=index,
                    token_start=index,
                    token_end=index + 1,
                    text=text,
                    layer=27,
                )
                for index, text in enumerate(texts)
            )
            output.append(
                ControlledQuery(
                    example_id=f"{split}-s{seed}-e{example_index:03d}",
                    split=split,
                    seed=seed,
                    hidden=hidden,
                    lexical=lexical,
                    target_labels=labels,
                    provenance=provenance,
                    facet_count=facet_count,
                    interleaved=interleaved,
                    shared_entity=shared_entity,
                    distractor_units=distractor_units,
                )
            )
    return output


def controlled_manifest(cases: list[ControlledQuery]) -> dict:
    rows = [
        {
            "example_id": row.example_id,
            "seed": row.seed,
            "facet_count": row.facet_count,
            "units": int(row.hidden.shape[0]),
            "interleaved": row.interleaved,
            "shared_entity": row.shared_entity,
            "distractor_units": row.distractor_units,
        }
        for row in cases
    ]
    payload = json.dumps(rows, sort_keys=True).encode("utf-8")
    return {
        "examples": len(rows),
        "seeds": sorted({row["seed"] for row in rows}),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "facet_range": [min(row["facet_count"] for row in rows), max(row["facet_count"] for row in rows)],
    }
