"""Gate A: select query facet type and robust query-support context."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
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
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    SEEDS,
    canonical_oracle_parent_indices,
    evidence_parent_groups,
    oracle_set_metrics,
    validation_partition,
)
from pra_hf.query_facets import (
    QueryFacetSet,
    build_contextual_query_facets,
    build_multiscale_query_facets,
    build_span_query_facets,
    build_token_query_facets,
    clip_query_support,
    deterministic_phrase_spans,
    global_query_facet,
    score_semantic_query_facets,
    select_bounded_parents,
    target_rank_metrics,
)
from pra_torch.hf import load_hf_routing_projection


FRACTIONS = (0.10, 0.20, 0.30, 0.40)
PRIMARY_FRACTION = 0.20
SUPPORT_MODES = ("question", "latest_message", "tail16", "tail32", "tail64", "full_prompt")


@dataclass(frozen=True)
class FacetConfig:
    """One parameter-free query-facet construction."""

    name: str
    family: str
    window: int | None = None
    include_global: bool = True
    reduction: str = "max"


FACET_CONFIGS = (
    FacetConfig("global", "global"),
    *(FacetConfig(f"w{window}_s{max(1, window // 2)}", "window", window) for window in (2, 4, 8, 16)),
    FacetConfig("w4_s2_top2mean", "window", 4, reduction="top_m_mean"),
    FacetConfig("w4_s2_local_only", "window", 4, include_global=False),
    FacetConfig("token", "token"),
    FacetConfig("multiscale_2_4_8_16", "multiscale"),
    FacetConfig("phrase_relation", "phrase"),
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _token_texts(tokenizer, ids: torch.Tensor) -> list[str]:
    return [tokenizer.decode([int(value)], skip_special_tokens=False) for value in ids]


def _support_span(feature: dict, mode: str) -> tuple[int, int]:
    token_count = int(feature["query_hidden_states"].shape[0])
    if mode == "question":
        return tuple(map(int, feature["question_span"]))
    if mode == "latest_message":
        return tuple(map(int, feature.get("latest_message_span", feature["question_span"])))
    if mode == "full_prompt":
        return 0, token_count
    if mode.startswith("tail"):
        return clip_query_support(
            token_count, max_support_tokens=int(mode.removeprefix("tail"))
        )
    raise ValueError(f"Unsupported query-support mode: {mode}")


def build_facets(
    feature: dict,
    config: FacetConfig,
    support_mode: str,
    tokenizer,
) -> QueryFacetSet:
    """Build one facet family from a single captured contextual sequence."""
    hidden = feature["query_hidden_states"].float()
    native = feature.get("query_pre_query")
    native = native.float() if native is not None else None
    support = _support_span(feature, support_mode)
    if config.family == "global":
        return global_query_facet(hidden, native)
    if config.family == "window":
        return build_contextual_query_facets(
            hidden,
            support,
            window=int(config.window),
            stride=max(1, int(config.window) // 2),
            include_global=config.include_global,
            native_query=native,
        )
    if config.family == "token":
        return build_token_query_facets(
            hidden, support, include_global=config.include_global, native_query=native
        )
    if config.family == "multiscale":
        return build_multiscale_query_facets(
            hidden,
            support,
            windows=(2, 4, 8, 16),
            include_global=config.include_global,
            native_query=native,
        )
    if config.family == "phrase":
        spans = deterministic_phrase_spans(
            _token_texts(tokenizer, feature["prompt_input_ids"]), support
        )
        return build_span_query_facets(
            hidden,
            spans,
            include_global=config.include_global,
            family="phrase",
            native_query=native,
        )
    raise ValueError(f"Unsupported facet family: {config.family}")


def _score_row(
    feature: dict,
    facets: QueryFacetSet,
    scores,
    *,
    config: FacetConfig,
    support_mode: str,
    seed: int,
    fraction: float,
    scoring_seconds: float,
    control: str,
) -> dict:
    budget = max(1, math.ceil(len(feature["parent_spans"]) * fraction))
    selection = select_bounded_parents(scores, budget)
    selected = set(selection.parent_indices)
    groups = evidence_parent_groups(feature)
    first_group = groups[0]
    oracle = canonical_oracle_parent_indices(feature)
    rank = target_rank_metrics(scores.scores, first_group, selected)
    oracle_metrics = oracle_set_metrics(selected, oracle)
    target = int(rank["target_parent"])
    winner = int(scores.winning_facet[target])
    provenance = facets.provenance[winner]
    selected_tokens = sum(
        int(feature["parent_spans"][parent][1])
        - int(feature["parent_spans"][parent][0])
        for parent in selected
    )
    stale_parent = feature.get("stale_parent")
    return {
        "control": control,
        "partition": validation_partition(feature["example_id"]),
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "seed": seed,
        "fraction": fraction,
        "budget": budget,
        "facet_config": config.name,
        "facet_family": config.family,
        "facet_reduction": config.reduction,
        "include_global": config.include_global,
        "query_support_mode": support_mode,
        "query_support_span": json.dumps(_support_span(feature, support_mode)),
        "query_support_tokens": _support_span(feature, support_mode)[1]
        - _support_span(feature, support_mode)[0],
        "query_facet_count": len(facets.provenance),
        "oracle_root_parent": target,
        "oracle_root_rank": rank["target_rank"],
        "oracle_root_present": rank["target_present"],
        "mrr": rank["mrr"],
        "recall_at_1": rank["recall_at_1"],
        "recall_at_2": rank["recall_at_2"],
        "recall_at_4": rank["recall_at_4"],
        "recall_at_8": rank["recall_at_8"],
        "oracle_margin": rank["oracle_margin"],
        "score_entropy": rank["score_entropy"],
        "oracle_recall": oracle_metrics["oracle_recall"],
        "oracle_precision": oracle_metrics["oracle_precision"],
        "oracle_jaccard": oracle_metrics["oracle_jaccard"],
        "complete_oracle": oracle_metrics["complete_oracle"],
        "false_positive_parent_count": len(selected - oracle),
        "winning_facet": winner,
        "winning_facet_kind": provenance.kind,
        "winning_facet_family": provenance.family,
        "winning_facet_span": json.dumps([provenance.token_start, provenance.token_end]),
        "selected_parent_ids": json.dumps(selection.parent_indices),
        "stale_parent": stale_parent,
        "stale_parent_selected": (
            float(int(stale_parent) in selected) if stale_parent is not None else None
        ),
        "junk_target_tokens": feature.get("junk_target_tokens"),
        "junk_observed_tokens": feature.get("junk_observed_tokens"),
        "q_memory_comparisons": scores.comparisons,
        "final_parent_count": len(selection.parent_indices),
        "active_final_kv_tokens": selected_tokens,
        "active_final_kv_fraction": selected_tokens / max(int(feature["source_tokens"]), 1),
        "scoring_wall_time": scoring_seconds,
        "full_query_forward_count": feature["full_query_forward_count"],
        "independent_window_forward_count": feature["independent_window_forward_count"],
    }


METRICS = (
    "oracle_root_present", "oracle_root_rank", "mrr", "recall_at_1", "recall_at_2",
    "recall_at_4", "recall_at_8", "oracle_margin", "score_entropy", "oracle_recall",
    "oracle_precision", "oracle_jaccard", "complete_oracle", "false_positive_parent_count",
    "query_support_tokens", "query_facet_count", "q_memory_comparisons",
    "active_final_kv_tokens", "active_final_kv_fraction", "scoring_wall_time",
    "stale_parent_selected",
)


def _aggregate(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = dict(zip(dimensions, key))
        record["rows"] = len(values)
        record["identities"] = len({row["example_id"] for row in values})
        record["seeds"] = len({row["seed"] for row in values})
        for metric in METRICS:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            if samples:
                record[metric] = statistics.fmean(samples)
        output.append(record)
    return output


def _select_facet(summary: list[dict]) -> tuple[FacetConfig, list[dict]]:
    primary = [
        row for row in summary
        if row["partition"] == "validation" and row["fraction"] == PRIMARY_FRACTION
    ]
    baseline_qasper = next(
        row for row in primary if row["dataset"] == "qasper" and row["facet_config"] == "global"
    )
    audit = []
    for config in FACET_CONFIGS:
        hotpot = next(
            row for row in primary
            if row["dataset"] == "hotpotqa" and row["facet_config"] == config.name
        )
        qasper = next(
            row for row in primary
            if row["dataset"] == "qasper" and row["facet_config"] == config.name
        )
        audit.append(
            {
                "candidate": config.name,
                "hotpot_root_present": hotpot["oracle_root_present"],
                "hotpot_mrr": hotpot["mrr"],
                "hotpot_false_positives": hotpot["false_positive_parent_count"],
                "qasper_root_present": qasper["oracle_root_present"],
                "qasper_preserved": qasper["oracle_root_present"] >= baseline_qasper["oracle_root_present"] - 0.05,
                "comparisons": hotpot["q_memory_comparisons"] + qasper["q_memory_comparisons"],
            }
        )
    admissible = [row for row in audit if row["qasper_preserved"]]
    winner = max(
        admissible,
        key=lambda row: (
            row["hotpot_root_present"], row["hotpot_mrr"],
            -row["hotpot_false_positives"], -row["comparisons"], row["candidate"],
        ),
    )
    return next(config for config in FACET_CONFIGS if config.name == winner["candidate"]), audit


def _select_support(
    clean_summary: list[dict], contamination_summary: list[dict], baseline_qasper: float
) -> tuple[str, list[dict]]:
    audit = []
    for mode in SUPPORT_MODES:
        clean_h = next(row for row in clean_summary if row["partition"] == "validation" and row["dataset"] == "hotpotqa" and row["query_support_mode"] == mode)
        clean_q = next(row for row in clean_summary if row["partition"] == "validation" and row["dataset"] == "qasper" and row["query_support_mode"] == mode)
        stale_h_values = [row for row in contamination_summary if row["partition"] == "validation" and row["dataset"] == "hotpotqa" and row["query_support_mode"] == mode and int(row["junk_target_tokens"]) > 0]
        stale_q_values = [row for row in contamination_summary if row["partition"] == "validation" and row["dataset"] == "qasper" and row["query_support_mode"] == mode and int(row["junk_target_tokens"]) > 0]
        record = {
            "candidate": mode,
            "clean_hotpot_root_present": clean_h["oracle_root_present"],
            "clean_hotpot_mrr": clean_h["mrr"],
            "clean_qasper_root_present": clean_q["oracle_root_present"],
            "stale_hotpot_root_present": statistics.fmean(row["oracle_root_present"] for row in stale_h_values),
            "stale_qasper_root_present": statistics.fmean(row["oracle_root_present"] for row in stale_q_values),
            "stale_parent_selection_rate": statistics.fmean(row["stale_parent_selected"] for row in stale_h_values + stale_q_values),
            "comparisons": clean_h["q_memory_comparisons"] + clean_q["q_memory_comparisons"],
        }
        record["qasper_preserved"] = record["clean_qasper_root_present"] >= baseline_qasper - 0.05
        audit.append(record)
    admissible = [row for row in audit if row["qasper_preserved"]]
    winner = max(
        admissible,
        key=lambda row: (
            row["clean_hotpot_root_present"] + row["stale_hotpot_root_present"],
            row["stale_qasper_root_present"], -row["stale_parent_selection_rate"],
            row["clean_hotpot_mrr"], -row["comparisons"], row["candidate"],
        ),
    )
    return str(winner["candidate"]), audit


def _score_features(features, source_by_id, projections, tokenizer, configs, supports, fractions, control):
    rows = []
    for feature_index, query_feature in enumerate(features, start=1):
        source = source_by_id[query_feature["example_id"]]
        enriched = {**query_feature, **{key: source[key] for key in ("parent_spans", "source_tokens", "parent_positive_mask", "evidence_spans")}}
        for seed, projection in projections.items():
            device = next(projection.parameters()).device
            parent_memory = projection.project_memory(source["parent_hidden"].to(device))
            for config in configs:
                for support in supports:
                    facets = build_facets(enriched, config, support, tokenizer)
                    if parent_memory.device.type == "cuda":
                        torch.cuda.synchronize(parent_memory.device)
                    started = time.perf_counter()
                    projected = projection.project_query(facets.hidden.to(parent_memory.device))
                    result = score_semantic_query_facets(
                        projected, parent_memory, facet_reduction=config.reduction, top_m=2
                    )
                    if parent_memory.device.type == "cuda":
                        torch.cuda.synchronize(parent_memory.device)
                    elapsed = time.perf_counter() - started
                    for fraction in fractions:
                        rows.append(
                            _score_row(
                                enriched, facets, result, config=config, support_mode=support,
                                seed=seed, fraction=fraction, scoring_seconds=elapsed, control=control,
                            )
                        )
        print(f"[facet-gate {control} {feature_index}/{len(features)}] {query_feature['dataset']} {query_feature['example_id']}", flush=True)
    return rows


def _plot(facet_summary, support_summary, contamination_summary, output_dir: Path) -> None:
    test = [row for row in facet_summary if row["partition"] == "test" and row["fraction"] == PRIMARY_FRACTION]
    configs = [config.name for config in FACET_CONFIGS]
    figure, axis = plt.subplots(figsize=(9.4, 4.4))
    x = range(len(configs)); width = 0.38
    for offset, dataset, color in ((-width/2, "hotpotqa", "#4c78a8"), (width/2, "qasper", "#f58518")):
        lookup = {row["facet_config"]: row for row in test if row["dataset"] == dataset}
        axis.bar([i + offset for i in x], [lookup[name]["oracle_root_present"] for name in configs], width, label=dataset, color=color)
    axis.set_xticks(list(x), configs, rotation=28, ha="right")
    axis.set_ylabel("First evidence group in root Top-B")
    axis.set_ylim(0, 1.02); axis.grid(axis="y", alpha=.25); axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(output_dir / f"facet_type_gate.{suffix}", dpi=180)
    plt.close(figure)

    contam = [row for row in contamination_summary if row["partition"] == "test" and row["dataset"] == "hotpotqa"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    colors = {"question":"#4c78a8", "latest_message":"#54a24b", "tail16":"#f58518", "tail32":"#e45756", "tail64":"#9c755f", "full_prompt":"#b279a2"}
    for mode in SUPPORT_MODES:
        values = sorted([row for row in contam if row["query_support_mode"] == mode], key=lambda row: int(row["junk_target_tokens"]))
        axes[0].plot([int(row["junk_target_tokens"]) for row in values], [row["oracle_root_present"] for row in values], marker="o", label=mode, color=colors[mode])
        axes[1].plot([int(row["junk_target_tokens"]) for row in values], [row["stale_parent_selected"] for row in values], marker="o", label=mode, color=colors[mode])
    axes[0].set_ylabel("Correct root presence"); axes[1].set_ylabel("Stale parent selected")
    for axis in axes:
        axis.set_xlabel("Injected stale-memory tokens"); axis.set_ylim(0,1.02); axis.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(output_dir / f"stale_support_gate.{suffix}", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    source_features = torch.load(args.source_feature_file, map_location="cpu", weights_only=False)
    clean_features = torch.load(args.query_feature_file, map_location="cpu", weights_only=False)
    contamination = torch.load(args.contamination_feature_file, map_location="cpu", weights_only=False)
    source_by_id = {row["example_id"]: row for row in source_features}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    projections = {}
    for seed in args.seeds:
        checkpoint = args.projection_dir / "checkpoints" / f"asymmetric_linear_d128_last_joint_seed{seed}_margin_exhaustive.pt"
        projections[seed] = load_hf_routing_projection(checkpoint, device=device)

    facet_rows = _score_features(clean_features, source_by_id, projections, tokenizer, FACET_CONFIGS, ("question",), args.fractions, "clean_facet_type")
    facet_summary = _aggregate(facet_rows, ("partition","dataset","fraction","facet_config","query_support_mode"))
    selected_facet, facet_audit = _select_facet(facet_summary)

    support_rows = _score_features(clean_features, source_by_id, projections, tokenizer, (selected_facet,), SUPPORT_MODES, (PRIMARY_FRACTION,), "clean_support")
    support_summary = _aggregate(support_rows, ("partition","dataset","fraction","facet_config","query_support_mode"))
    contamination_rows = _score_features(contamination, source_by_id, projections, tokenizer, (selected_facet,), SUPPORT_MODES, (PRIMARY_FRACTION,), "stale_contamination")
    contamination_summary = _aggregate(contamination_rows, ("partition","dataset","fraction","facet_config","query_support_mode","junk_target_tokens"))
    baseline_qasper = next(row["oracle_root_present"] for row in facet_summary if row["partition"]=="validation" and row["dataset"]=="qasper" and row["fraction"]==PRIMARY_FRACTION and row["facet_config"]=="global")
    selected_support, support_audit = _select_support(support_summary, contamination_summary, baseline_qasper)

    # Reproduce the old global and w4 controls before accepting a new gate.
    prior = json.loads(args.prior_result_file.read_text(encoding="utf-8"))
    test_primary = [row for row in facet_summary if row["partition"]=="test" and row["fraction"]==PRIMARY_FRACTION]
    global_hotpot = next(row for row in test_primary if row["dataset"]=="hotpotqa" and row["facet_config"]=="global")
    w4_hotpot = next(row for row in test_primary if row["dataset"]=="hotpotqa" and row["facet_config"]=="w4_s2")
    prior_lookup = {(row["dataset"],row["variant"]):row for row in prior["primary_heldout_summary"]}
    if not math.isclose(global_hotpot["oracle_root_present"], prior_lookup["hotpotqa","A_global_semantic"]["oracle_root_present"], abs_tol=1e-12):
        raise AssertionError("Global root baseline changed.")
    if not math.isclose(w4_hotpot["oracle_root_present"], prior_lookup["hotpotqa","B_multi_span_semantic"]["oracle_root_present"], abs_tol=1e-12):
        raise AssertionError("Winning w4 facet baseline changed.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_test = [row for row in support_summary if row["partition"]=="test" and row["query_support_mode"]==selected_support]
    artifact = {
        "schema_version":"1.0", "runtime":runtime_metadata(), "diagnostic_only":True,
        "production_default_changed":False, "training_performed":False,
        "model_id":MODEL_ID, "model_revision":MODEL_REVISION, "seeds":list(args.seeds),
        "fractions":list(args.fractions), "primary_fraction":PRIMARY_FRACTION,
        "memory_representation_frozen":True, "full_query_forward_count_per_capture":1,
        "independent_window_forward_count":0,
        "facet_candidates":[asdict(config) for config in FACET_CONFIGS],
        "support_candidates":list(SUPPORT_MODES), "facet_selection_audit":facet_audit,
        "support_selection_audit":support_audit, "selected_facet_config":asdict(selected_facet),
        "selected_query_support":selected_support, "heldout_selected_summary":selected_test,
        "baseline_reproduction":{"global_hotpot_root":global_hotpot["oracle_root_present"],"w4_hotpot_root":w4_hotpot["oracle_root_present"]},
        "gate_a_recommendation":f"freeze_{selected_facet.name}_{selected_support}",
        "sdk_change_in_this_iteration":False,
    }
    (args.output_dir / "grounded_facet_gate_results.json").write_text(json.dumps(artifact,indent=2,sort_keys=True),encoding="utf-8")
    _write_csv(args.output_dir / "facet_type_rows.csv", facet_rows)
    _write_csv(args.output_dir / "facet_type_summary.csv", facet_summary)
    _write_csv(args.output_dir / "facet_selection_audit.csv", facet_audit)
    _write_csv(args.output_dir / "query_support_rows.csv", support_rows)
    _write_csv(args.output_dir / "query_support_summary.csv", support_summary)
    _write_csv(args.output_dir / "stale_contamination_rows.csv", contamination_rows)
    _write_csv(args.output_dir / "stale_contamination_summary.csv", contamination_summary)
    _write_csv(args.output_dir / "support_selection_audit.csv", support_audit)
    _plot(facet_summary, support_summary, contamination_summary, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds",default=",".join(map(str,SEEDS)))
    parser.add_argument("--fractions",default=",".join(map(str,FRACTIONS)))
    result_root=ROOT/"docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument("--source-feature-file",type=Path,default=result_root/"native_qk_closure/native_qk_features_test.pt")
    parser.add_argument("--query-feature-file",type=Path,default=result_root/"query_entry_facets/query_entry_features.pt")
    parser.add_argument("--contamination-feature-file",type=Path,default=result_root/"grounded_query_facets/grounded_query_features.pt")
    parser.add_argument("--projection-dir",type=Path,default=ROOT/"docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    parser.add_argument("--prior-result-file",type=Path,default=result_root/"query_entry_facets/query_entry_results.json")
    parser.add_argument("--output-dir",type=Path,default=result_root/"grounded_query_facets")
    args=parser.parse_args(); args.seeds=tuple(map(int,args.seeds.split(","))); args.fractions=tuple(map(float,args.fractions.split(","))); return args


if __name__=="__main__":
    result=run(parse_args()); print(json.dumps({"selected_facet_config":result["selected_facet_config"],"selected_query_support":result["selected_query_support"]},indent=2))
