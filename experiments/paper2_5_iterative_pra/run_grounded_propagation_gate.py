"""Gate B: validate frozen native A-to-B proposals with current query facets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.run_grounded_facet_gate import (
    FacetConfig,
    _support_span,
    build_facets,
)
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    SEEDS,
    evidence_parent_groups,
    native_qk_parent_scores,
    validation_partition,
)
from pra_hf.grounded_propagation import (
    generate_associative_candidates,
    query_validate_candidates,
    rank_grounded_candidates,
)
from pra_torch.hf import load_hf_routing_projection


CANDIDATE_KS = (4, 8)
GROUNDING_SCOPES = ("all", "residual")
RANK_LAMBDAS = (0.25, 0.5, 1.0, 2.0)
MATERIAL_R1_GAIN = 0.10
MAX_R4_LOSS = 0.05


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _facet_text(tokenizer, input_ids: torch.Tensor, start: int, end: int) -> str:
    if start < 0:
        return "[global contextual query]"
    return tokenizer.decode(input_ids[start:end], skip_special_tokens=True).strip()


def _target_rank(selected: tuple[int, ...], targets: set[int], candidate_k: int) -> int:
    return next(
        (rank for rank, parent in enumerate(selected, start=1) if parent in targets),
        candidate_k + 1,
    )


def _margin(validation, candidates, targets: set[int]) -> float | None:
    target_scores = [
        float(validation.scores[index])
        for index, parent in enumerate(candidates.parent_indices)
        if parent in targets
    ]
    distractors = [
        float(validation.scores[index])
        for index, parent in enumerate(candidates.parent_indices)
        if parent not in targets
    ]
    if not target_scores:
        return None
    return max(target_scores) - (max(distractors) if distractors else float("-inf"))


def _evaluate(base: dict, *, mode: str, scope: str, rank_lambda=1.0, threshold=float("-inf")) -> dict:
    validation = base[scope]
    ranking = rank_grounded_candidates(
        base["candidates"],
        validation,
        mode=mode,
        final_k=base["candidate_k"],
        rank_lambda=rank_lambda,
        query_threshold=threshold,
        root_facet=base["root_facet"],
    )
    rank = _target_rank(ranking.selected, base["targets"], base["candidate_k"])
    provenance = []
    for candidate in ranking.candidates:
        facet = base["facets"].provenance[candidate.validating_facet]
        provenance.append(
            {
                **asdict(candidate),
                "validating_facet_kind": facet.kind,
                "validating_facet_family": facet.family,
                "validating_facet_span": [facet.token_start, facet.token_end],
                "validating_facet_text": _facet_text(
                    base["tokenizer"], base["prompt_ids"], facet.token_start, facet.token_end
                ),
            }
        )
    return {
        "partition": base["partition"],
        "dataset": "hotpotqa",
        "example_id": base["example_id"],
        "seed": base["seed"],
        "transition": base["transition"],
        "candidate_k": base["candidate_k"],
        "mode": mode,
        "grounding_scope": scope,
        "rank_lambda": rank_lambda if mode == "rank_conjunction" else None,
        "query_threshold": threshold if mode == "threshold_conjunction" else None,
        "source_oracle_parents": json.dumps(sorted(base["source_group"])),
        "target_oracle_parents": json.dumps(sorted(base["targets"])),
        "target_in_associative_candidates": float(
            bool(set(base["candidates"].parent_indices) & base["targets"])
        ),
        "target_rank": rank,
        "mrr": 1.0 / rank,
        "recall_at_1": float(rank <= 1),
        "recall_at_2": float(rank <= 2),
        "recall_at_4": float(rank <= 4),
        "query_oracle_margin": _margin(validation, base["candidates"], base["targets"]),
        "root_facet": base["root_facet"],
        "root_facet_kind": base["root_provenance"].kind,
        "root_facet_family": base["root_provenance"].family,
        "root_facet_span": json.dumps(
            [base["root_provenance"].token_start, base["root_provenance"].token_end]
        ),
        "root_facet_text": base["root_facet_text"],
        "candidate_parent_ids": json.dumps(base["candidates"].parent_indices),
        "ranked_admitted_parent_ids": json.dumps(ranking.selected),
        "top1_selected_parent_ids": json.dumps(ranking.selected[:1]),
        "candidate_provenance": json.dumps(provenance, sort_keys=True),
        "validating_facet_equals_root_rate": statistics.fmean(
            float(item.validating_facet_is_root) for item in ranking.candidates
        ),
        "query_support_tokens": base["query_support_tokens"],
        "query_facet_count": len(base["facets"].provenance),
        "query_root_comparisons": base["query_root_comparisons"],
        "association_comparisons": ranking.association_comparisons,
        "validation_comparisons": ranking.validation_comparisons,
        "total_search_comparisons": (
            ranking.association_comparisons
            + base["query_root_comparisons"]
            + ranking.validation_comparisons
        ),
        "final_parent_count": 1,
        "final_parent_budget": 1,
        "final_kv_budget_tokens": 0,
        "materialized_kv_tokens_added_during_diagnostic": 0,
        "association_wall_time": base["association_wall_time"],
        "validation_wall_time": base["validation_wall_time"],
        "routing_wall_time": base["association_wall_time"] + base["validation_wall_time"],
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    dimensions = (
        "partition", "candidate_k", "mode", "grounding_scope", "rank_lambda", "query_threshold"
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(row)
    output = []
    metrics = (
        "target_in_associative_candidates", "target_rank", "mrr", "recall_at_1",
        "recall_at_2", "recall_at_4", "query_oracle_margin",
        "validating_facet_equals_root_rate", "query_support_tokens", "query_facet_count",
        "query_root_comparisons", "association_comparisons", "validation_comparisons",
        "total_search_comparisons", "association_wall_time", "validation_wall_time",
        "routing_wall_time",
    )
    for key, values in grouped.items():
        record = dict(zip(dimensions, key))
        record["rows"] = len(values)
        record["edges"] = len({(row["example_id"], row["transition"]) for row in values})
        record["seeds"] = len({row["seed"] for row in values})
        for metric in metrics:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            if samples:
                record[metric] = statistics.fmean(samples)
        output.append(record)
    return sorted(output, key=lambda row: tuple(str(row[key]) for key in dimensions))


def _config_name(row: dict) -> str:
    if row["mode"] == "rank_conjunction":
        suffix = f"lambda={row['rank_lambda']}"
    elif row["mode"] == "threshold_conjunction":
        suffix = f"threshold={row['query_threshold']}"
    else:
        suffix = "fixed"
    return f"{row['mode']}:{row['grounding_scope']}:{suffix}"


def _select_validation(summary: list[dict]) -> tuple[dict, list[dict]]:
    validation = [
        row for row in summary
        if row["partition"] == "validation" and row["candidate_k"] == 4
        and row["mode"] != "association"
    ]
    baseline = next(
        row for row in summary
        if row["partition"] == "validation" and row["candidate_k"] == 4
        and row["mode"] == "association" and row["grounding_scope"] == "all"
    )
    audit = []
    for row in validation:
        record = {
            "candidate": _config_name(row),
            "recall_at_1": row["recall_at_1"],
            "mrr": row["mrr"],
            "recall_at_4": row["recall_at_4"],
            "r4_preserved": row["recall_at_4"] >= baseline["recall_at_4"] - MAX_R4_LOSS,
            "validation_comparisons": row["validation_comparisons"],
        }
        audit.append(record)
    admissible = [row for row in validation if row["recall_at_4"] >= baseline["recall_at_4"] - MAX_R4_LOSS]
    winner = max(
        admissible,
        key=lambda row: (
            row["recall_at_1"], row["mrr"], row["recall_at_4"],
            -row["validation_comparisons"], _config_name(row),
        ),
    )
    return winner, audit


def _synthetic_controls() -> list[dict]:
    controls = []
    association = torch.tensor([float("-inf"), 0.95, 0.70, 0.20])
    candidates = generate_associative_candidates(
        association, source_parents={0}, candidate_k=3, comparisons=3
    )
    scores = torch.tensor([[0.0, 0.10, 0.90, -0.10], [0.0, 0.05, 0.95, -0.20]])
    validation = query_validate_candidates(scores, candidates)
    assoc = rank_grounded_candidates(candidates, validation, mode="association")
    grounded = rank_grounded_candidates(candidates, validation, mode="query_rerank")
    controls.append({
        "control": "grounded_association", "association_top1": assoc.selected[0],
        "grounded_top1": grounded.selected[0], "expected_grounded": 2,
        "passed": grounded.selected[0] == 2 and assoc.selected[0] == 1,
    })
    drift = rank_grounded_candidates(
        candidates, validation, mode="threshold_conjunction", final_k=3, query_threshold=0.5
    )
    controls.append({
        "control": "drift_stop", "selected": list(drift.selected), "expected": [2],
        "passed": drift.selected == (2,),
    })
    residual_scores = torch.tensor([[0.0, 0.95, 0.20, 0.0], [0.0, 0.10, 0.90, 0.0]])
    residual = query_validate_candidates(
        residual_scores, candidates, root_facet=0, residual_only=True
    )
    bridge = rank_grounded_candidates(candidates, residual, mode="query_rerank")
    controls.append({
        "control": "unresolved_bridge", "selected": list(bridge.selected),
        "expected_top1": 2, "passed": bridge.selected[0] == 2,
    })
    if not all(row["passed"] for row in controls):
        raise AssertionError("A grounded propagation synthetic control failed.")
    return controls


def _plot(summary: list[dict], winner: dict, output_dir: Path) -> None:
    baseline = next(
        row for row in summary
        if row["partition"] == "test" and row["candidate_k"] == 4
        and row["mode"] == "association" and row["grounding_scope"] == "all"
    )
    heldout = next(
        row for row in summary
        if row["partition"] == "test" and row["candidate_k"] == 4
        and row["mode"] == winner["mode"]
        and row["grounding_scope"] == winner["grounding_scope"]
        and row["rank_lambda"] == winner["rank_lambda"]
        and row["query_threshold"] == winner["query_threshold"]
    )
    labels = ["Association", "Query-grounded"]
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    x = [0, 1]
    width = 0.22
    for offset, metric, label, color in (
        (-width, "recall_at_1", "R@1", "#4c78a8"),
        (0, "mrr", "MRR", "#54a24b"),
        (width, "recall_at_4", "R@4", "#f58518"),
    ):
        axis.bar([value + offset for value in x], [baseline[metric], heldout[metric]], width, label=label, color=color)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.04)
    axis.set_ylabel("Conditional successor retrieval")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"grounded_propagation_gate.{suffix}", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    sources = torch.load(args.source_feature_file, map_location="cpu", weights_only=False)
    query_features = torch.load(args.query_feature_file, map_location="cpu", weights_only=False)
    query_by_id = {row["example_id"]: row for row in query_features}
    gate_a = json.loads(args.gate_a_file.read_text(encoding="utf-8"))
    facet_config = FacetConfig(**gate_a["selected_facet_config"])
    support_mode = gate_a["selected_query_support"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    projections = {}
    for seed in args.seeds:
        checkpoint = args.projection_dir / "checkpoints" / (
            f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        )
        projections[seed] = load_hf_routing_projection(checkpoint, device=device)

    bases = []
    for feature in sources:
        groups = evidence_parent_groups(feature)
        if feature["dataset"] != "hotpotqa" or len(groups) < 2:
            continue
        query_feature = query_by_id[feature["example_id"]]
        enriched = {**query_feature, **{key: feature[key] for key in ("parent_spans", "source_tokens", "parent_positive_mask", "evidence_spans")}}
        facets = build_facets(enriched, facet_config, support_mode, tokenizer)
        support_start, support_end = _support_span(query_feature, support_mode)
        for transition, (source_group, targets) in enumerate(zip(groups, groups[1:])):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            association_started = time.perf_counter()
            association_scores, native_dots = native_qk_parent_scores(
                feature, source_group, device,
                token_reduction="top_m_mean", head_reduction="top_m_mean", top_m=4,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            association_wall_time = time.perf_counter() - association_started
            proposals = {
                candidate_k: generate_associative_candidates(
                    association_scores,
                    source_parents=source_group,
                    candidate_k=candidate_k,
                    comparisons=native_dots,
                )
                for candidate_k in CANDIDATE_KS
            }
            scored_parents = sorted(
                source_group | set(proposals[max(CANDIDATE_KS)].parent_indices)
            )
            scored_parent_tensor = torch.tensor(scored_parents, device=device)
            for seed, projection in projections.items():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_started = time.perf_counter()
                parent_memory = projection.project_memory(feature["parent_hidden"].to(device))
                projected_facets = projection.project_query(facets.hidden.to(device))
                sparse_scores = projected_facets.float() @ parent_memory[scored_parent_tensor].float().T
                component_scores = torch.full(
                    (projected_facets.shape[0], len(feature["parent_spans"])),
                    float("-inf"), device=device,
                )
                component_scores[:, scored_parent_tensor] = sparse_scores
                source_ids = torch.tensor(sorted(source_group), device=device)
                root_flat = component_scores[:, source_ids].reshape(-1).argmax()
                root_facet = int(root_flat // len(source_group))
                root_provenance = facets.provenance[root_facet]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_wall_time = time.perf_counter() - validation_started
                for candidate_k in CANDIDATE_KS:
                    candidates = proposals[candidate_k]
                    all_validation = query_validate_candidates(
                        component_scores, candidates, root_facet=root_facet
                    )
                    residual_validation = query_validate_candidates(
                        component_scores, candidates, root_facet=root_facet, residual_only=True
                    )
                    bases.append({
                        "partition": validation_partition(feature["example_id"]),
                        "example_id": feature["example_id"], "seed": seed,
                        "transition": transition, "candidate_k": candidate_k,
                        "source_group": source_group, "targets": targets,
                        "candidates": candidates, "all": all_validation,
                        "residual": residual_validation, "root_facet": root_facet,
                        "root_provenance": root_provenance,
                        "root_facet_text": _facet_text(
                            tokenizer, query_feature["prompt_input_ids"],
                            root_provenance.token_start, root_provenance.token_end,
                        ),
                        "facets": facets, "tokenizer": tokenizer,
                        "prompt_ids": query_feature["prompt_input_ids"],
                        "query_support_tokens": support_end - support_start,
                        "query_root_comparisons": len(facets.provenance) * len(source_group),
                        "association_wall_time": association_wall_time,
                        "validation_wall_time": validation_wall_time,
                    })
        print(f"[grounded-edge] {feature['example_id']} transitions={max(0, len(groups)-1)}", flush=True)

    # Association/query rerank/rank conjunction can be evaluated directly.
    rows = []
    for base in bases:
        rows.append(_evaluate(base, mode="association", scope="all"))
        for scope in GROUNDING_SCOPES:
            rows.append(_evaluate(base, mode="query_rerank", scope=scope))
            for rank_lambda in RANK_LAMBDAS:
                rows.append(_evaluate(base, mode="rank_conjunction", scope=scope, rank_lambda=rank_lambda))

    # Threshold candidates are fitted only from validation query scores and then frozen.
    thresholds = {}
    for scope in GROUNDING_SCOPES:
        values = sorted({
            float(score)
            for base in bases if base["partition"] == "validation" and base["candidate_k"] == 4
            for score in base[scope].scores.tolist()
        })
        candidates = [float("-inf"), *values]
        trial_rows = []
        for threshold in candidates:
            trial = [
                _evaluate(base, mode="threshold_conjunction", scope=scope, threshold=threshold)
                for base in bases if base["partition"] == "validation" and base["candidate_k"] == 4
            ]
            aggregate = _aggregate(trial)[0]
            trial_rows.append((threshold, aggregate))
        baseline_r4 = _aggregate([
            _evaluate(base, mode="association", scope="all")
            for base in bases if base["partition"] == "validation" and base["candidate_k"] == 4
        ])[0]["recall_at_4"]
        admissible = [item for item in trial_rows if item[1]["recall_at_4"] >= baseline_r4 - MAX_R4_LOSS]
        thresholds[scope] = max(
            admissible,
            key=lambda item: (item[1]["recall_at_1"], item[1]["mrr"], item[0]),
        )[0]
        for base in bases:
            rows.append(_evaluate(
                base, mode="threshold_conjunction", scope=scope, threshold=thresholds[scope]
            ))

    summary = _aggregate(rows)
    winner, selection_audit = _select_validation(summary)
    heldout_baseline = next(
        row for row in summary
        if row["partition"] == "test" and row["candidate_k"] == 4
        and row["mode"] == "association" and row["grounding_scope"] == "all"
    )
    heldout_winner = next(
        row for row in summary
        if row["partition"] == "test" and row["candidate_k"] == 4
        and row["mode"] == winner["mode"]
        and row["grounding_scope"] == winner["grounding_scope"]
        and row["rank_lambda"] == winner["rank_lambda"]
        and row["query_threshold"] == winner["query_threshold"]
    )
    r1_gain = heldout_winner["recall_at_1"] - heldout_baseline["recall_at_1"]
    r4_loss = heldout_baseline["recall_at_4"] - heldout_winner["recall_at_4"]
    material_success = r1_gain >= MATERIAL_R1_GAIN and r4_loss <= MAX_R4_LOSS
    synthetic = _synthetic_controls()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "1.0", "runtime": runtime_metadata(),
        "diagnostic_only": True, "production_default_changed": False,
        "training_performed": False, "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "seeds": list(args.seeds),
        "gate_a_frozen_facet_config": asdict(facet_config),
        "gate_a_frozen_query_support": support_mode,
        "association_geometry": "native_qk_top4_token_and_head_mean",
        "candidate_ks": list(CANDIDATE_KS), "primary_candidate_k": 4,
        "rank_lambdas": list(RANK_LAMBDAS), "frozen_thresholds": thresholds,
        "raw_score_families_mixed": False, "static_query_grounding": True,
        "conditional_final_parent_budget": 1,
        "conditional_final_kv_budget_tokens": 0,
        "kv_materialization_performed": False,
        "validation_selection": {key: winner[key] for key in (
            "mode", "grounding_scope", "rank_lambda", "query_threshold",
            "recall_at_1", "mrr", "recall_at_4"
        )},
        "selection_audit": selection_audit,
        "heldout_association": heldout_baseline,
        "heldout_query_grounded": heldout_winner,
        "heldout_r1_gain": r1_gain, "heldout_r4_loss": r4_loss,
        "material_success_rule": {
            "minimum_r1_gain": MATERIAL_R1_GAIN, "maximum_r4_loss": MAX_R4_LOSS
        },
        "conditional_gate_passed": material_success,
        "end_to_end_run": False,
        "end_to_end_reason": (
            "conditional_success_requires_separate_same-budget_evaluation"
            if material_success else "stopped_by_predeclared_conditional_gate"
        ),
        "synthetic_controls": synthetic,
        "sdk_change_in_this_iteration": False,
    }
    (args.output_dir / "grounded_propagation_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "grounded_propagation_rows.csv", rows)
    _write_csv(args.output_dir / "grounded_propagation_summary.csv", summary)
    _write_csv(args.output_dir / "grounded_propagation_selection.csv", selection_audit)
    _plot(summary, winner, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    result_root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument("--source-feature-file", type=Path, default=result_root / "native_qk_closure/native_qk_features_test.pt")
    parser.add_argument("--query-feature-file", type=Path, default=result_root / "query_entry_facets/query_entry_features.pt")
    parser.add_argument("--gate-a-file", type=Path, default=result_root / "grounded_query_facets/grounded_facet_gate_results.json")
    parser.add_argument("--projection-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    parser.add_argument("--output-dir", type=Path, default=result_root / "grounded_query_facets")
    args = parser.parse_args()
    args.seeds = tuple(map(int, args.seeds.split(",")))
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "validation_selection": result["validation_selection"],
        "heldout_association": result["heldout_association"],
        "heldout_query_grounded": result["heldout_query_grounded"],
        "conditional_gate_passed": result["conditional_gate_passed"],
    }, indent=2))
