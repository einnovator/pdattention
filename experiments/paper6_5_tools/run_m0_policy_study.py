"""Run the deterministic Paper 6.5 catalog and policy-selection study."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from data.agent_resources import (
    CatalogQuery,
    generate_agent_catalog,
    replace_versions,
    synthetic_semantic_vector,
)
from pra_hf.agent_resources import (
    DiscoveryDecision,
    DiscoveryHint,
    DiscoveryMode,
    DiscoveryRequest,
    PersistentResourceIndex,
    ReliabilityCalibrator,
    ResourceDiscoveryEngine,
    terms,
)


POLICIES = (
    "fixed_explicit",
    "fixed_token",
    "fixed_index",
    "fixed_semantic",
    "fixed_hybrid",
    "auto",
    "user_hint",
    "adaptive",
)
FIXED_POLICIES = POLICIES[:5]
POLICY_COST = {
    "fixed_explicit": 1.0,
    "fixed_token": 2.0,
    "fixed_index": 1.5,
    "fixed_semantic": 4.0,
    "fixed_hybrid": 6.0,
    "auto": 1.5,
    "user_hint": 1.5,
    "adaptive": 2.0,
}
SEEDS = (11, 23, 37, 53, 71)
SIZES = (8, 32, 128, 512, 2048, 8192)


def _semantic_encoder(text: str):
    return synthetic_semantic_vector(text, dimensions=96)


def _request(query: CatalogQuery, hint: DiscoveryHint) -> DiscoveryRequest:
    return DiscoveryRequest(
        query=query.query,
        hint=hint,
        namespace=query.namespace,
        explicit_reference_uris=query.explicit_reference_uris,
        tenant_id="paper6_5",
        top_k=1,
        side_effecting=False,
    )


def _hint(policy: str, stratum: str) -> DiscoveryHint:
    if policy.startswith("fixed_"):
        return DiscoveryHint(policy.removeprefix("fixed_"), strict=True)
    if policy == "auto":
        return DiscoveryHint("auto", strict=True)
    if policy == "adaptive":
        return DiscoveryHint("adaptive", strict=False)
    mapping = {
        "explicit_uri": "explicit",
        "exact_name": "token",
        "alias": "token",
        "typo": "token",
        "semantic_paraphrase": "semantic",
        "description": "index",
        "ambiguous": "adaptive",
        "nonexistent": "adaptive",
    }
    return DiscoveryHint(mapping[stratum], strict=False)


def _top1_correct(selected: tuple[str, ...], query: CatalogQuery) -> bool | None:
    if query.expected_decision != "select":
        return None
    return bool(selected and selected[0] in query.target_uris)


def _outcome_correct(trace, query: CatalogQuery) -> bool:
    if query.expected_decision == "select":
        return bool(_top1_correct(trace.selected_uris, query))
    return trace.decision in {DiscoveryDecision.ASK, DiscoveryDecision.ABSTAIN}


def _evaluate(
    *,
    engine: ResourceDiscoveryEngine,
    query: CatalogQuery,
    policy: str,
    scores,
) -> dict[str, object]:
    request = _request(query, _hint(policy, query.stratum))
    start = time.perf_counter_ns()
    trace = engine.discover(request, scored_candidates=scores)
    policy_us = (time.perf_counter_ns() - start) / 1_000.0
    top1 = _top1_correct(trace.selected_uris, query)
    return {
        "query_id": query.query_id,
        "split": query.split,
        "stratum": query.stratum,
        "policy": policy,
        "target_uris": "|".join(query.target_uris),
        "selected_uri": trace.selected_uris[0] if trace.selected_uris else "",
        "top1_correct": "" if top1 is None else int(top1),
        "outcome_correct": int(_outcome_correct(trace, query)),
        "expected_decision": query.expected_decision,
        "decision": trace.decision.value,
        "false_act": int(
            query.expected_decision != "select" and trace.decision == DiscoveryDecision.SELECT
        ),
        "confidence": trace.confidence,
        "margin": trace.margin,
        "executed_path": ">".join(trace.executed_path),
        "retrieval_stages": len(trace.executed_path),
        "fallback_count": trace.fallback_count,
        "hint_complied": int(trace.hint_complied),
        "policy_us": policy_us,
        "candidate_count": len(trace.candidates),
        "index_fingerprint": trace.index_fingerprint,
    }


def _fit_calibrators(validation_rows: list[dict[str, object]]) -> dict[str, ReliabilityCalibrator]:
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for row in validation_rows:
        if row["top1_correct"] == "":
            continue
        grouped[str(row["policy"])].append(
            (float(row["confidence"]), bool(int(row["top1_correct"])))
        )
    return {
        policy: ReliabilityCalibrator.fit(grouped.get(policy, ()), bins=8)
        for policy in POLICIES
    }


def _add_oracle_fields(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["catalog_size"]), int(row["seed"]), str(row["query_id"]))].append(row)
    for query_rows in grouped.values():
        fixed = [row for row in query_rows if row["policy"] in FIXED_POLICIES]
        selectable = [row for row in fixed if row["top1_correct"] != ""]
        oracle = min(
            selectable,
            key=lambda row: (-int(row["top1_correct"]), POLICY_COST[str(row["policy"])]),
        ) if selectable else None
        for row in query_rows:
            if oracle is None:
                row["oracle_policy"] = ""
                row["oracle_correct"] = ""
                row["quality_regret"] = ""
                row["cost_regret"] = ""
                continue
            oracle_correct = int(oracle["top1_correct"])
            row_correct = int(row["top1_correct"]) if row["top1_correct"] != "" else 0
            row["oracle_policy"] = oracle["policy"]
            row["oracle_correct"] = oracle_correct
            row["quality_regret"] = oracle_correct - row_correct
            row["cost_regret"] = max(
                0.0,
                POLICY_COST[str(row["policy"])] - POLICY_COST[str(oracle["policy"])],
            ) if row_correct == oracle_correct else ""


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _index_cost_rows(
    *,
    catalog,
    index: PersistentResourceIndex,
    size: int,
    seed: int,
    build_ms: float,
    latency_samples: int,
) -> list[dict[str, object]]:
    definition_tokens = [
        len(terms(f"{resource.description} {resource.content}"))
        for resource in catalog.resources
    ]
    eager_prompt_tokens = sum(definition_tokens)
    active_materialized_tokens = mean(definition_tokens)
    mutation_count = max(1, round(size * 0.01))
    mutation_indices = random.Random(seed + 1000).sample(range(size), mutation_count)
    start = time.perf_counter_ns()
    mutated = PersistentResourceIndex(
        replace_versions(catalog.resources, mutation_indices),
        semantic_encoder=_semantic_encoder,
        fingerprint_metadata={"semantic_encoder": "synthetic-concept-hash-32-v1"},
    )
    mutation_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    sample = catalog.split("test")[:latency_samples]
    rows = []
    for mode in (
        DiscoveryMode.EXPLICIT,
        DiscoveryMode.TOKEN,
        DiscoveryMode.INDEX,
        DiscoveryMode.SEMANTIC,
        DiscoveryMode.HYBRID,
    ):
        timings = []
        candidates = []
        for query in sample:
            request = _request(query, DiscoveryHint(mode, strict=True))
            # One warmup keeps import and first allocator effects out of the audit.
            index.score(request, channels=(mode,))
            start = time.perf_counter_ns()
            scored = index.score(request, channels=(mode,))
            timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
            candidates.append(len(scored))
        rows.append(
            {
                "catalog_size": size,
                "seed": seed,
                "mode": mode.value,
                "build_ms": build_ms,
                "mutation_fraction": mutation_count / size,
                "mutation_rebuild_ms": mutation_ms,
                "source_fingerprint_changed": int(
                    index.fingerprint.digest != mutated.fingerprint.digest
                ),
                "index_bytes": index.estimated_bytes,
                "logical_definition_tokens": eager_prompt_tokens,
                "active_materialized_tokens": active_materialized_tokens,
                "active_fraction": active_materialized_tokens / max(eager_prompt_tokens, 1),
                "warm_query_mean_ms": mean(timings),
                "warm_query_median_ms": median(timings),
                "mean_candidates_scored": mean(candidates),
                "latency_samples": len(sample),
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for size in args.sizes:
        for seed in args.seeds:
            catalog = generate_agent_catalog(
                size,
                seed=seed,
                validation_identities=args.validation_identities,
                test_identities=args.test_identities,
            )
            start = time.perf_counter_ns()
            index = PersistentResourceIndex(
                catalog.resources,
                semantic_encoder=_semantic_encoder,
                fingerprint_metadata={"semantic_encoder": "synthetic-concept-hash-32-v1"},
            )
            build_ms = (time.perf_counter_ns() - start) / 1_000_000.0

            validation_rows: list[dict[str, object]] = []
            validation_scores = {}
            for query in catalog.split("validation"):
                base = _request(query, DiscoveryHint("hybrid", strict=True))
                scores = index.score(base, channels=("hybrid",))
                validation_scores[query.query_id] = scores
                for policy in POLICIES:
                    engine = ResourceDiscoveryEngine(index)
                    validation_rows.append(
                        _evaluate(engine=engine, query=query, policy=policy, scores=scores)
                    )
            calibrators = _fit_calibrators(validation_rows)
            for row in validation_rows:
                row.update({"catalog_size": size, "seed": seed, "calibrated": 0})
                all_rows.append(row)

            for query in catalog.split("test"):
                base = _request(query, DiscoveryHint("hybrid", strict=True))
                scores = index.score(base, channels=("hybrid",))
                for policy in POLICIES:
                    engine = ResourceDiscoveryEngine(index, calibrator=calibrators[policy])
                    row = _evaluate(engine=engine, query=query, policy=policy, scores=scores)
                    row.update({"catalog_size": size, "seed": seed, "calibrated": 1})
                    all_rows.append(row)

            cost_rows.extend(
                _index_cost_rows(
                    catalog=catalog,
                    index=index,
                    size=size,
                    seed=seed,
                    build_ms=build_ms,
                    latency_samples=args.latency_samples,
                )
            )
            print(f"completed size={size} seed={seed} build_ms={build_ms:.2f}", flush=True)

    _add_oracle_fields(all_rows)
    # Put identifying columns first without coupling evaluators to dict insertion order.
    ordered = []
    for row in all_rows:
        prefix = {
            "catalog_size": row.pop("catalog_size"),
            "seed": row.pop("seed"),
            "calibrated": row.pop("calibrated"),
        }
        ordered.append({**prefix, **row})
    _write_csv(output / "m0_policy_per_query.csv", ordered)
    _write_csv(output / "m0_index_costs.csv", cost_rows)
    manifest = {
        "study": "paper6_5_m0_policy",
        "catalog_sizes": list(args.sizes),
        "seeds": list(args.seeds),
        "policies": list(POLICIES),
        "policy_cost": POLICY_COST,
        "validation_identities": args.validation_identities,
        "test_identities": args.test_identities,
        "semantic_control": "deterministic action/object/family concept vector, 96 dimensions",
        "index_update": "full immutable rebuild after one-percent version mutation",
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    (output / "m0_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="docs/papers/shared/results/paper6_5_tools",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--validation-identities", type=int, default=6)
    parser.add_argument("--test-identities", type=int, default=18)
    parser.add_argument("--latency-samples", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
