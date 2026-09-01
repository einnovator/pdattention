"""Calibrate and evaluate a noncontiguous MLX PRA consumer-layer gate.

Fixed late-layer suffixes lost substantial quality in the Paper 6.2 scaling
campaign. This runner learns a static binary layer gate from a calibration
split by greedily removing layer groups only while response fidelity to the
all-layer segmented teacher remains inside explicit bounds. It then evaluates
the frozen mask on disjoint seeds. The result is a trained placement candidate,
not a production profile: held-out validation must close before promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper6_2_mlx.run_answer_quality_pressure import (  # noqa: E402
    _answer_logprob,
    _metrics,
)
from experiments.paper6_2_mlx.run_matched_e0_e2 import (  # noqa: E402
    _cache_snapshot,
    _generate_timed,
    _restore_cache,
)
from experiments.paper6_2_mlx.run_model_consumer_scaling import (  # noqa: E402
    _cohort,
    _prepared_tokens,
    _runtime_metadata,
)


@dataclass(frozen=True)
class GateScore:
    """Calibration fidelity and cost for one binary consumer-layer mask."""

    layers: tuple[int, ...]
    mean_abs_logprob_delta: float
    first_token_agreement: float
    layer_fraction: float


def layer_groups(layer_count: int, group_count: int) -> tuple[tuple[int, ...], ...]:
    """Partition decoder layers into ordered, nearly equal search groups."""

    if layer_count <= 0 or group_count <= 0:
        raise ValueError("Layer and group counts must be positive.")
    width = math.ceil(layer_count / min(group_count, layer_count))
    return tuple(
        tuple(range(start, min(start + width, layer_count)))
        for start in range(0, layer_count, width)
    )


def removable_masks(
    selected_layers: Sequence[int], groups: Iterable[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    """Return unique masks produced by removing one currently active group."""

    selected = set(map(int, selected_layers))
    masks = {
        tuple(sorted(selected.difference(map(int, group))))
        for group in groups
        if selected.intersection(map(int, group))
    }
    return tuple(sorted((mask for mask in masks if mask), key=lambda x: (len(x), x)))


def choose_gate_candidate(
    candidates: Sequence[GateScore],
    *,
    max_abs_logprob_delta: float,
    min_first_token_agreement: float,
) -> GateScore | None:
    """Choose the smallest admissible mask, breaking ties by fidelity."""

    admissible = [
        candidate
        for candidate in candidates
        if candidate.mean_abs_logprob_delta <= max_abs_logprob_delta
        and candidate.first_token_agreement >= min_first_token_agreement
    ]
    if not admissible:
        return None
    return min(
        admissible,
        key=lambda candidate: (
            candidate.layer_fraction,
            candidate.mean_abs_logprob_delta,
            -candidate.first_token_agreement,
            candidate.layers,
        ),
    )


def _prepare_cases(model, tokenizer, cohort, source_limit: int):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx import encode_native_memory

    cases = []
    for seed, example in cohort:
        source, query, answer = _prepared_tokens(tokenizer, example, source_limit)
        ordinary = make_prompt_cache(model)
        encoded = model(mx.array(source, dtype=mx.int32)[None], cache=ordinary)
        mx.eval(encoded)
        ordinary_states = _cache_snapshot(ordinary)
        memory = encode_native_memory(model, source)
        cases.append((seed, example, source, query, answer, ordinary_states, memory))
    return cases


def _score_mask(model, tokenizer, cases, layers: tuple[int, ...], teacher):
    from pra_mlx import make_native_prompt_cache

    deltas = []
    agreements = []
    for case, target in zip(cases, teacher, strict=True):
        _, _, _, query, answer, _, memory = case
        factory = lambda: make_native_prompt_cache(
            model, memory, selected_layers=layers, segmented=True
        )
        logprob = _answer_logprob(model, query, answer, factory())
        generated = _generate_timed(model, tokenizer, query, factory(), 1)
        token_ids = list(map(int, generated["output_token_ids"]))
        deltas.append(abs(logprob - float(target["logprob"])))
        agreements.append(float(token_ids == target["output_token_ids"]))
    return GateScore(
        layers=layers,
        mean_abs_logprob_delta=fmean(deltas),
        first_token_agreement=fmean(agreements),
        layer_fraction=len(layers) / len(model.layers),
    )


def _teacher(model, tokenizer, cases, layers):
    from pra_mlx import make_native_prompt_cache

    rows = []
    for case in cases:
        _, _, _, query, answer, _, memory = case
        factory = lambda: make_native_prompt_cache(
            model, memory, selected_layers=layers, segmented=True
        )
        rows.append(
            {
                "logprob": _answer_logprob(model, query, answer, factory()),
                "output_token_ids": list(
                    map(
                        int,
                        _generate_timed(model, tokenizer, query, factory(), 1)[
                            "output_token_ids"
                        ],
                    )
                ),
            }
        )
    return rows


def _evaluate(model, tokenizer, cases, learned_layers, max_new_tokens):
    from pra_mlx import make_native_prompt_cache

    all_layers = tuple(range(len(model.layers)))
    rows = []
    for seed, example, source, query, answer, ordinary_states, memory in cases:
        conditions = {
            "E0_WARM": lambda: _restore_cache(model, ordinary_states),
            "E2_SEGMENTED_ALL_LAYERS": lambda: make_native_prompt_cache(
                model, memory, selected_layers=all_layers, segmented=True
            ),
            "E2_SEGMENTED_LEARNED_GATE": lambda: make_native_prompt_cache(
                model, memory, selected_layers=learned_layers, segmented=True
            ),
        }
        baseline_ids = None
        for condition, factory in conditions.items():
            started = time.perf_counter()
            logprob = _answer_logprob(model, query, answer, factory())
            generated = _generate_timed(
                model, tokenizer, query, factory(), max_new_tokens
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            output_ids = list(map(int, generated["output_token_ids"]))
            if condition == "E0_WARM":
                baseline_ids = output_ids
            exact, token_f1 = _metrics(str(generated["output"]), example.answer)
            rows.append(
                {
                    "dataset": example.dataset,
                    "seed": seed,
                    "example_id": example.example_id,
                    "condition": condition,
                    "source_tokens": len(source),
                    "consumer_layers": (
                        []
                        if condition == "E0_WARM"
                        else list(all_layers)
                        if condition == "E2_SEGMENTED_ALL_LAYERS"
                        else list(learned_layers)
                    ),
                    "gold_answer_logprob": logprob,
                    "exact_match": exact,
                    "token_f1": token_f1,
                    "sequence_agreement_vs_e0": float(output_ids == baseline_ids),
                    "ttft_ms": generated["ttft_ms"],
                    "itl_ms": generated["itl_ms"],
                    "completion_latency_ms": generated["completion_latency_ms"],
                    "scored_request_ms": elapsed_ms,
                    "output_token_ids": output_ids,
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from huggingface_hub import model_info
    from mlx_lm import load
    from pra_mlx import install_qwen3_segmented_attention

    model, tokenizer = load(args.model, revision=args.revision)
    revision = model_info(args.model, revision=args.revision).sha
    install_qwen3_segmented_attention(model)
    cohort = _cohort(args)
    calibration = [item for item in cohort if item[0] in set(args.calibration_seed)]
    held_out = [item for item in cohort if item[0] not in set(args.calibration_seed)]
    if not calibration or not held_out:
        raise ValueError("Calibration and held-out cohorts must both be non-empty.")
    calibration_cases = _prepare_cases(
        model, tokenizer, calibration, args.max_source_tokens
    )
    all_layers = tuple(range(len(model.layers)))
    teacher = _teacher(model, tokenizer, calibration_cases, all_layers)
    groups = layer_groups(len(model.layers), args.group_count)
    selected = all_layers
    trace = []
    while True:
        scores = [
            _score_mask(model, tokenizer, calibration_cases, mask, teacher)
            for mask in removable_masks(selected, groups)
            if len(mask) < len(selected)
        ]
        chosen = choose_gate_candidate(
            scores,
            max_abs_logprob_delta=args.max_abs_logprob_delta,
            min_first_token_agreement=args.min_first_token_agreement,
        )
        trace.append(
            {
                "active_before": list(selected),
                "candidates": [score.__dict__ for score in scores],
                "accepted": None if chosen is None else chosen.__dict__,
            }
        )
        if chosen is None:
            break
        selected = chosen.layers
    del calibration_cases
    mx.clear_cache()
    held_out_cases = _prepare_cases(model, tokenizer, held_out, args.max_source_tokens)
    rows = _evaluate(
        model, tokenizer, held_out_cases, selected, args.max_new_tokens
    )
    payload = {
        "schema_version": "paper6.2-mlx-learned-consumer-gate-v1",
        "experiment": "held_out_noncontiguous_consumer_layer_gate",
        "evidence_tier": "MODEL_BACKED_NATURAL_QA_CALIBRATION",
        "profile_status": "CALIBRATION_PENDING",
        "runtime": _runtime_metadata(),
        "model_id": args.model,
        "model_revision": revision,
        "layer_count": len(model.layers),
        "calibration_seeds": args.calibration_seed,
        "held_out_seeds": sorted({seed for seed, _ in held_out}),
        "layer_groups": [list(group) for group in groups],
        "selected_layers": list(selected),
        "selected_layer_fraction": len(selected) / len(model.layers),
        "constraints": {
            "max_abs_logprob_delta": args.max_abs_logprob_delta,
            "min_first_token_agreement": args.min_first_token_agreement,
        },
        "calibration_trace": trace,
        "held_out_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("qasper", "hotpotqa", "2wikimultihopqa"),
        required=True,
    )
    parser.add_argument("--calibration-seed", action="append", type=int, default=[])
    parser.add_argument("--examples-per-seed", type=int, default=1)
    parser.add_argument("--group-count", type=int, default=8)
    parser.add_argument("--max-abs-logprob-delta", type=float, default=0.05)
    parser.add_argument("--min-first-token-agreement", type=float, default=0.9)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.calibration_seed:
        args.calibration_seed = [11]
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "selected_layers": result["selected_layers"],
                "held_out_rows": len(result["held_out_rows"]),
            },
            indent=2,
        )
    )
