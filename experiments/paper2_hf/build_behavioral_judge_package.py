"""Build a deterministic blind LLM-judge package from Paper 2 generations.

The canonical adapter consumes ``last14_combo.json``.  It never calls a model:
prompts are rehydrated from the pinned benchmark split and every answer comes
from the recorded generation artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    # Inserting ROOT first and src second leaves src at highest precedence.
    sys.path.insert(0, value)

from experiments.paper2_hf.routing.run_query_strategies import load_split_examples


SCHEMA_VERSION = "1.0"
EVALUATION_NAME = "paper2_behavioral_equivalence"
REQUESTED_SCORES = [
    "semantic_equivalence",
    "relative_quality",
    "validity_a",
    "validity_b",
    "confidence",
]
BLIND_ITEM_KEYS = {
    "item_id",
    "task_type",
    "prompt",
    "answer_a",
    "answer_b",
    "requested_scores",
}
FORBIDDEN_ID_HINTS = {
    "native",
    "pra",
    "routed",
    "oracle",
    "adapted",
    "baseline",
    "qwen",
    "hotpot",
    "qasper",
}

JUDGE_PROMPT = """You are evaluating two model outputs for the same user prompt.

Do NOT infer which system produced either answer. Judge only the text shown.

For each item return:
1. semantic_equivalence: 0..100
   100 = effectively the same meaning and usefulness
   75 = mostly equivalent, with minor meaningful differences
   50 = partial overlap with a material difference
   25 = mostly different
   0 = contradictory or functionally unrelated
2. relative_quality: -100..100
   -100 means Answer A is much better
   0 means the answers are equally good
   +100 means Answer B is much better
3. validity_a: 0..100
4. validity_b: 0..100
5. confidence: 0..100
6. reason: at most 40 words

Judge validity from coherence, relevance, plausibility, and usefulness. For continuation
tasks, judge whether both are coherent and plausible continuations of the prompt; do not
require a hidden reference verbatim. For QA tasks, judge factual and semantic answer
equivalence and usefulness.

Return valid JSON only. Do not include Markdown or commentary outside the JSON.

Use this response schema:
{
  "schema_version": "1.0",
  "judge_name": "MODEL_NAME_OR_UNKNOWN",
  "items": [
    {
      "item_id": "judge_000001_ab",
      "semantic_equivalence": 92,
      "relative_quality": 5,
      "validity_a": 91,
      "validity_b": 94,
      "confidence": 87,
      "reason": "Both answer the same question; B is slightly clearer."
    }
  ]
}
"""


@dataclass(frozen=True)
class PairSpec:
    """One unblinded behavioral comparison before A/B randomization."""

    source_example_id: str
    dataset: str
    task_type: str
    prompt: str
    left_condition: str
    right_condition: str
    left_answer: str
    right_answer: str
    group: str
    model_id: str
    model_revision: str
    generation_mode: str = "greedy"
    generation_seed: int | None = None
    pra_selected_fraction: float | None = None
    pra_materialized_kv_fraction: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_text(rendered)


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _condition_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("example_id"),
            row.get("condition"),
            row.get("variant"),
            row.get("seed"),
        )
        if key in indexed:
            raise ValueError(f"Duplicate generation row: {key}")
        indexed[key] = row
    return indexed


def _qa_prompt(question: str) -> str:
    return f"Answer briefly and directly.\nQuestion: {question.strip()}\nAnswer:"


def _test_offset(artifact: dict[str, Any]) -> int:
    """Return the recorded split offset, with the legacy last-14 default."""

    split_metadata = artifact.get("split_metadata", {})
    if not isinstance(split_metadata, dict):
        split_metadata = {}
    return int(split_metadata.get("test_offset", 16))


def _rehydrate_examples(artifact: dict[str, Any], cache_dir: Path) -> dict[str, dict[str, Any]]:
    count = int(artifact.get("test_examples_per_dataset", 8))
    offset = _test_offset(artifact)
    seed = int(artifact.get("data_seed", 20260811))
    examples = load_split_examples(cache_dir, count, offset, seed)
    by_id = {str(example["id"]): example for example in examples}
    expected = set(artifact.get("identities", {}).get("test", []))
    missing = sorted(expected - set(by_id))
    if missing:
        raise ValueError(f"Could not rehydrate {len(missing)} test identities: {missing[:3]}")
    return by_id


def _answer(row: dict[str, Any]) -> str:
    value = str(row.get("generated_answer", "")).strip()
    if not value:
        raise ValueError(f"Missing generated answer for {row.get('example_id')}")
    return value


def _last14_pairs(
    artifact: dict[str, Any],
    cache_dir: Path,
    adapted_variant: str,
) -> tuple[list[PairSpec], dict[str, Any]]:
    """Extract the predeclared Paper 2 comparisons from last14_combo output."""

    examples = _rehydrate_examples(artifact, cache_dir)
    controls = {
        (row["dataset"], row["example_id"], row["condition"]): row
        for row in artifact["test_control_rows"]
    }
    indexed = _condition_rows(artifact["test_rows"])
    seeds = sorted(int(seed) for seed in artifact["optimization_seeds"])
    fixed_seed = seeds[0]
    model_id = str(artifact["model_id"])
    revision = str(artifact["model_revision"])
    pairs: list[PairSpec] = []

    for example_id in artifact["identities"]["test"]:
        example = examples[example_id]
        dataset = str(example["dataset"])
        prompt = _qa_prompt(str(example["question"]))
        fixed = indexed[(dataset, example_id, "routed", "fixed", fixed_seed)]
        common = {
            "source_example_id": example_id,
            "dataset": dataset,
            "task_type": "qa",
            "prompt": prompt,
            "right_condition": "pra_routed_frozen",
            "right_answer": _answer(fixed),
            "model_id": model_id,
            "model_revision": revision,
            "pra_selected_fraction": float(fixed["selected_chunk_fraction"]),
            "pra_materialized_kv_fraction": float(fixed["materialized_native_kv_fraction"]),
        }
        for condition, label, group in (
            ("none", "native_no_context", "native_no_context_vs_pra"),
            ("direct_text", "native_direct_evidence", "native_direct_evidence_vs_pra"),
            ("full_context", "native_full_context", "native_full_context_vs_pra"),
        ):
            control = controls.get((dataset, example_id, condition))
            if control is None:
                continue
            pairs.append(
                PairSpec(
                    **common,
                    left_condition=label,
                    left_answer=_answer(control),
                    group=group,
                    metadata={
                        "reference_answer": str(example["answer"]),
                        "full_context_complete": bool(control["full_context_complete"]),
                        "pra_variant": "fixed",
                        "pra_seed": fixed_seed,
                    },
                )
            )

        for seed in seeds:
            adapted = indexed[(dataset, example_id, "routed", adapted_variant, seed)]
            pairs.append(
                PairSpec(
                    source_example_id=example_id,
                    dataset=dataset,
                    task_type="qa",
                    prompt=prompt,
                    left_condition="pra_routed_frozen",
                    right_condition=f"pra_routed_{adapted_variant}",
                    left_answer=_answer(fixed),
                    right_answer=_answer(adapted),
                    group="frozen_pra_vs_adapted_pra",
                    model_id=model_id,
                    model_revision=revision,
                    generation_seed=seed,
                    pra_selected_fraction=float(adapted["selected_chunk_fraction"]),
                    pra_materialized_kv_fraction=float(
                        adapted["materialized_native_kv_fraction"]
                    ),
                    metadata={
                        "reference_answer": str(example["answer"]),
                        "frozen_seed": fixed_seed,
                        "adapted_seed": seed,
                        "adapted_variant": adapted_variant,
                    },
                )
            )

    availability = {
        "included_groups": sorted({pair.group for pair in pairs}),
        "rehydrated_split": {
            "examples_per_dataset": int(artifact.get("test_examples_per_dataset", 8)),
            "test_offset": _test_offset(artifact),
            "data_seed": int(artifact.get("data_seed", 20260811)),
        },
        "unavailable_groups": {
            "pra_fraction_sweep": (
                "The generation artifact has one realized routed operating point, not decoded "
                "5/10/20/30% generations."
            ),
            "native_sampling_variation": "All canonical generations are greedy.",
            "paraphrase_control": "No independently authored faithful paraphrases are recorded.",
        },
    }
    return pairs, availability


def load_pairs(
    input_paths: list[Path], cache_dir: Path, adapted_variant: str
) -> tuple[list[PairSpec], dict[str, Any]]:
    """Load supported artifacts, preserving source boundaries in package metadata."""

    pairs: list[PairSpec] = []
    availability: dict[str, Any] = {}
    for path in input_paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if {"test_rows", "test_control_rows", "identities"} <= set(artifact):
            extracted, source_availability = _last14_pairs(
                artifact, cache_dir, adapted_variant
            )
        else:
            raise ValueError(f"Unsupported generation artifact schema: {path}")
        pairs.extend(extracted)
        availability[str(path)] = source_availability
    if not pairs:
        raise ValueError("No behavioral comparisons were extracted.")
    return pairs, availability


def _controlled_corruption(text: str) -> tuple[str, str]:
    words = text.split()
    if len(words) >= 6:
        keep = max(2, len(words) // 2)
        return " ".join(words[:keep]).rstrip(".,;:") + ".", "meaningful_truncation"
    clean = text.rstrip(".")
    return f"It is not the case that {clean}.", "explicit_contradiction"


def add_controls(pairs: list[PairSpec]) -> list[PairSpec]:
    """Add one identical and one controlled-corruption calibration per example."""

    result = list(pairs)
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        key = (pair.dataset, pair.source_example_id)
        if key in seen or pair.left_condition not in {"native_no_context", "native_bounded"}:
            continue
        seen.add(key)
        result.append(
            PairSpec(
                **{
                    **pair.__dict__,
                    "right_condition": "control_identical_copy",
                    "right_answer": pair.left_answer,
                    "group": "calibration_identical",
                    "pra_selected_fraction": None,
                    "pra_materialized_kv_fraction": None,
                    "metadata": {"control_type": "identical_answer"},
                }
            )
        )
        corrupted, corruption = _controlled_corruption(pair.left_answer)
        result.append(
            PairSpec(
                **{
                    **pair.__dict__,
                    "right_condition": "control_corrupted_answer",
                    "right_answer": corrupted,
                    "group": "calibration_corrupted",
                    "pra_selected_fraction": None,
                    "pra_materialized_kv_fraction": None,
                    "metadata": {
                        "control_type": "corrupted_answer",
                        "corruption": corruption,
                    },
                }
            )
        )
    return result


def _blind_item(item_id: str, pair: PairSpec, swap: bool) -> tuple[dict, dict]:
    if swap:
        answer_a, answer_b = pair.right_answer, pair.left_answer
        condition_a, condition_b = pair.right_condition, pair.left_condition
    else:
        answer_a, answer_b = pair.left_answer, pair.right_answer
        condition_a, condition_b = pair.left_condition, pair.right_condition
    blind = {
        "item_id": item_id,
        "task_type": pair.task_type,
        "prompt": pair.prompt,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "requested_scores": list(REQUESTED_SCORES),
    }
    truth = {
        "item_id": item_id,
        "source_example_id": pair.source_example_id,
        "dataset": pair.dataset,
        "task_type": pair.task_type,
        "model_id": pair.model_id,
        "model_revision": pair.model_revision,
        "generation_mode": pair.generation_mode,
        "generation_seed": pair.generation_seed,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "comparison_group": pair.group,
        "pra_selected_fraction": pair.pra_selected_fraction,
        "pra_materialized_kv_fraction": pair.pra_materialized_kv_fraction,
        "order": f"{condition_a}__{condition_b}",
        "prompt_sha256": _sha256_text(pair.prompt),
        "answer_a_sha256": _sha256_text(answer_a),
        "answer_b_sha256": _sha256_text(answer_b),
        "metadata": pair.metadata,
    }
    return blind, truth


def validate_package(items: dict[str, Any], truth: dict[str, Any]) -> None:
    """Reject mismatches, duplicate IDs, and structural source-label leakage."""

    if items.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected blind schema version.")
    blind_rows = items.get("items", [])
    truth_rows = truth.get("items", [])
    blind_ids = [row.get("item_id") for row in blind_rows]
    truth_ids = [row.get("item_id") for row in truth_rows]
    if len(blind_ids) != len(set(blind_ids)):
        raise ValueError("Blind item IDs are not unique.")
    if len(truth_ids) != len(set(truth_ids)):
        raise ValueError("Truth item IDs are not unique.")
    if set(blind_ids) != set(truth_ids):
        raise ValueError("Every blinded item must have exactly one truth entry.")
    truth_by_id = {row["item_id"]: row for row in truth_rows}
    for row in blind_rows:
        if set(row) != BLIND_ITEM_KEYS:
            raise ValueError(f"Blind item has unexpected fields: {sorted(set(row)-BLIND_ITEM_KEYS)}")
        item_id = str(row["item_id"])
        if any(hint in item_id.lower() for hint in FORBIDDEN_ID_HINTS):
            raise ValueError(f"Blind item ID leaks a source hint: {item_id}")
        if row["requested_scores"] != REQUESTED_SCORES:
            raise ValueError(f"Requested scores differ from the prompt schema: {item_id}")
        mapped = truth_by_id[item_id]
        if not str(row["answer_a"]).strip() or not str(row["answer_b"]).strip():
            raise ValueError(f"Blank answer in {item_id}")
        for field in ("prompt", "answer_a", "answer_b"):
            if _sha256_text(str(row[field])) != mapped[f"{field}_sha256"]:
                raise ValueError(f"Blind/truth {field} order mismatch in {item_id}")


def _response_schema() -> dict[str, Any]:
    score = {"type": "number", "minimum": 0, "maximum": 100}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Paper 2 behavioral-equivalence judge response",
        "type": "object",
        "required": ["schema_version", "judge_name", "items"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "judge_name": {"type": "string", "minLength": 1},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "item_id",
                        "semantic_equivalence",
                        "relative_quality",
                        "validity_a",
                        "validity_b",
                        "confidence",
                        "reason",
                    ],
                    "properties": {
                        "item_id": {"type": "string"},
                        "semantic_equivalence": score,
                        "relative_quality": {"type": "number", "minimum": -100, "maximum": 100},
                        "validity_a": score,
                        "validity_b": score,
                        "confidence": score,
                        "reason": {
                            "type": "string",
                            "maxLength": 320,
                            "pattern": r"^(?:\S+\s+){0,39}\S+$",
                        },
                    },
                },
            },
        },
    }


def _llm_package(items: dict[str, Any], response_schema: dict[str, Any]) -> dict[str, Any]:
    """Bundle public instructions and blind items for a single-file LLM handoff."""

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_name": EVALUATION_NAME,
        "judge_instructions": JUDGE_PROMPT.strip(),
        "response_schema": response_schema,
        "items": items["items"],
    }


def _package_readme(item_count: int, batch_count: int) -> str:
    return f"""# Paper 2 Behavioral-Equivalence Judge Package

This directory contains {item_count} blind items in {batch_count} optional batches.

For a single-file handoff, send `behavioral_judge_llm_package.json` to each external judge. It
contains the instructions, response schema, and all blind items. Alternatively, send
`behavioral_judge_prompt.txt` followed by `behavioral_judge_items.json` or one file from
`batches/`. Never send or commit `behavioral_judge_truth.json` or its split copies; these
gitignored files are private unblinding metadata.

The blind item file intentionally contains only opaque IDs, task type, the common user prompt,
answers A/B, and requested score names. The truth file records condition labels, deterministic
order, generation metadata, source artifacts, and hashes for later unblinding.

Validate judge output against `behavioral_judge_response.schema.json`. The schema constrains
score ranges and fields; the textual prompt additionally limits each reason to 40 words.

Once external judging begins, freeze the generated files. Do not regenerate, reorder, or edit
items between judges; preserve the gitignored `behavioral_judge_truth.json` privately for
unblinding.

Calibration groups include identical-answer and controlled-corruption pairs. Paraphrase and
native-sampling controls are omitted when no independently recorded generations exist. Results
must be aggregated separately by `comparison_group`; do not silently pool native-context,
PRA-fraction, or adaptation conditions.
"""


def build_package(
    pairs: list[PairSpec],
    output_dir: Path,
    *,
    seed: int,
    include_order_reversal: bool,
    batch_size: int,
    input_paths: list[Path],
    availability: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    ordered = list(pairs)
    rng.shuffle(ordered)
    blind_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(ordered, start=1):
        base_swap = bool(rng.getrandbits(1))
        group_id = f"pair_{index:06d}"
        suffixes = (("ab", base_swap),)
        if include_order_reversal:
            suffixes += (("ba", not base_swap),)
        for suffix, swap in suffixes:
            item_id = f"judge_{index:06d}_{suffix}"
            blind, unblinded = _blind_item(item_id, pair, swap)
            unblinded["pair_group_id"] = group_id
            blind_rows.append(blind)
            truth_rows.append(unblinded)

    items = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_name": EVALUATION_NAME,
        "items": blind_rows,
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "randomization_seed": seed,
        "include_order_reversal": include_order_reversal,
        "generation_artifact_paths": [str(path) for path in input_paths],
        "generation_artifact_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths
        },
        "prompt_sha256": _sha256_text(JUDGE_PROMPT),
        "blind_items_sha256": _sha256_json(items),
        "availability": availability,
    }
    truth = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_name": EVALUATION_NAME,
        "metadata": metadata,
        "items": truth_rows,
    }
    validate_package(items, truth)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "behavioral_judge_prompt.txt").write_text(JUDGE_PROMPT, encoding="utf-8")
    (output_dir / "behavioral_judge_items.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "behavioral_judge_truth.json").write_text(
        json.dumps(truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    schema = _response_schema()
    (output_dir / "behavioral_judge_response.schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    llm_package = _llm_package(items, schema)
    (output_dir / "behavioral_judge_llm_package.json").write_text(
        json.dumps(llm_package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    example = {
        "schema_version": SCHEMA_VERSION,
        "judge_name": "MODEL_NAME_OR_UNKNOWN",
        "items": [
            {
                "item_id": blind_rows[0]["item_id"],
                "semantic_equivalence": 92,
                "relative_quality": 5,
                "validity_a": 91,
                "validity_b": 94,
                "confidence": 87,
                "reason": "Both answer the same question; B is slightly clearer.",
            }
        ],
    }
    (output_dir / "behavioral_judge_example_response.json").write_text(
        json.dumps(example, indent=2) + "\n", encoding="utf-8"
    )

    batches = [blind_rows[i : i + batch_size] for i in range(0, len(blind_rows), batch_size)]
    batch_dir = output_dir / "batches"
    batch_dir.mkdir(exist_ok=True)
    for stale_batch in batch_dir.glob("behavioral_judge_items_*.json"):
        stale_batch.unlink()
    for number, rows in enumerate(batches, start=1):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "evaluation_name": EVALUATION_NAME,
            "batch": number,
            "batch_count": len(batches),
            "items": rows,
        }
        (batch_dir / f"behavioral_judge_items_{number:03d}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (output_dir / "README.md").write_text(
        _package_readme(len(blind_rows), len(batches)), encoding="utf-8"
    )
    return items, truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-order-reversal", action="store_true")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--adapted-variant", default="residual_16")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    pairs, availability = load_pairs(args.input, args.cache_dir, args.adapted_variant)
    if args.include_controls:
        pairs = add_controls(pairs)
    items, truth = build_package(
        pairs,
        args.output_dir,
        seed=args.seed,
        include_order_reversal=args.include_order_reversal,
        batch_size=args.batch_size,
        input_paths=args.input,
        availability=availability,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "blind_items": len(items["items"]),
                "truth_items": len(truth["items"]),
                "comparison_groups": sorted(
                    {row["comparison_group"] for row in truth["items"]}
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
