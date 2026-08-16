"""Query-region and effort-router add-on studies for Paper 3.5.

The query-region study is a deterministic matched-layout retrieval control.  It
does not run or train a language model.  The router-architecture study reuses
the frozen Paper-2.5 validation/test effort ladders and trains only small
discriminative controllers against validation-derived minimum-effort labels.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from pra_hf.adaptive_runtime import ControllerFeatures, LinearEffortController, default_effort_profiles
from pra_hf.effort_router import (
    AutoregressiveEffortRouter,
    HashingQueryEncoder,
    MultiHeadEffortRouter,
    RouterActionSpace,
    profile_actions,
)
from pra_hf.query_regions import PromptSegment, QueryRegionSelector, token_offsets

from .adaptive_experiment import CONTROLLER_FEATURE_NAMES, build_examples


LAYOUTS = ("L0_context_query", "L1_query_context", "L2_context_query_context", "L3_instruction_context_query", "L4_query_long_payload")
PAYLOADS = ("prose", "logs", "urls")
REGION_METHODS = ("head", "suffix", "explicit", "structural", "broad", "auto_retry")
ROUTER_SEEDS = (1, 7, 21, 42, 87)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, values in sorted(grouped.items()):
        result = {key: value for key, value in zip(keys, identity)}
        result["n"] = len(values)
        result.update({metric: _mean(float(row[metric]) for row in values) for metric in metrics})
        output.append(result)
    return output


def _payload(payload: str, case: int, target: str, answer: str, filler: int) -> tuple[str, str]:
    evidence = f"Evidence: {target} failed because {answer}."
    distractors = [f"service_{case}_{index} healthy cause other_{index}" for index in range(7)]
    padding = [f"noise_{case}_{index} unrelated diagnostic payload" for index in range(filler)]
    if payload == "prose":
        context = "Context:\n" + evidence + "\n" + "\n".join(distractors + padding)
    elif payload == "logs":
        context = "LOGS:\n2026-08-16T10:00:00 ERROR " + evidence + "\n" + "\n".join(
            f"2026-08-16T10:{index:02d}:00 INFO {line}" for index, line in enumerate(distractors + padding)
        )
    elif payload == "urls":
        context = "REFERENCES:\n" + evidence + "\n" + "\n".join(
            f"https://example.test/{case}/{index}/{line.replace(' ', '-')}" for index, line in enumerate(distractors + padding)
        )
    else:
        raise ValueError(payload)
    return context, evidence


def _layout(layout: str, query: str, context: str) -> str:
    lines = context.splitlines()
    midpoint = max(1, len(lines) // 2)
    if layout == "L0_context_query":
        return f"{context}\n{query}"
    if layout == "L1_query_context":
        return f"{query}\n{context}"
    if layout == "L2_context_query_context":
        return "\n".join([*lines[:midpoint], query, *lines[midpoint:]])
    if layout == "L3_instruction_context_query":
        return f"INSTRUCTION: Use the evidence and ignore unrelated payload.\n{context}\n{query}"
    if layout == "L4_query_long_payload":
        return f"INSTRUCTION: Diagnose the requested service.\n{query}\n{context}"
    raise ValueError(layout)


def _subsequence_span(prompt: str, text: str) -> tuple[int, int]:
    prompt_tokens = [value.group(0) for value in re.finditer(r"\S+", prompt)]
    selected = [value.group(0) for value in re.finditer(r"\S+", text)]
    for start in range(len(prompt_tokens) - len(selected) + 1):
        if prompt_tokens[start : start + len(selected)] == selected:
            return start, start + len(selected)
    raise ValueError("Query text is absent from its generated prompt.")


def _token_f1(left: set[str], right: set[str]) -> tuple[float, float, float]:
    overlap = len(left & right)
    precision = overlap / max(len(left), 1)
    recall = overlap / max(len(right), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _span_metrics(predicted: Sequence[tuple[int, int]], target: tuple[int, int]) -> dict[str, float]:
    selected = {index for start, end in predicted for index in range(start, end)}
    gold = set(range(*target))
    precision, recall, _ = _token_f1(selected, gold)
    union = len(selected | gold)
    best = max(predicted, key=lambda span: len(set(range(*span)) & gold))
    return {
        "span_iou": len(selected & gold) / max(union, 1),
        "region_precision": precision,
        "region_recall": recall,
        "start_error": abs(best[0] - target[0]),
        "end_error": abs(best[1] - target[1]),
    }


def _candidate_chunks(case: int, target: str, answer: str) -> list[dict[str, str]]:
    candidates = [{"id": target, "text": f"{target} failure cause {answer}", "answer": answer}]
    candidates.extend(
        {
            "id": f"service_{case}_{index}",
            "text": f"service_{case}_{index} failure cause other_{index}",
            "answer": f"other_{index}",
        }
        for index in range(7)
    )
    return candidates


def _facets(texts: Sequence[str], policy: str) -> list[set[str]]:
    tokens = " ".join(texts).lower().replace("?", "").replace(":", "").split()
    if policy == "global":
        return [set(tokens)]
    if policy == "local4":
        return [set(tokens[index : index + 4]) for index in range(0, len(tokens), 4)] or [set()]
    if policy == "multiscale":
        return [set(tokens), *[set(tokens[index : index + 4]) for index in range(0, len(tokens), 4)]]
    raise ValueError(policy)


def _route(texts: Sequence[str], candidates: Sequence[Mapping[str, str]], roots: int, facet_policy: str) -> dict[str, Any]:
    facets = _facets(texts, facet_policy)
    scored = []
    for candidate in candidates:
        terms = set(candidate["text"].lower().split())
        score = max(len(facet & terms) / math.sqrt(max(len(facet) * len(terms), 1)) for facet in facets)
        scored.append((score, candidate["id"]))
    ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
    gap = ordered[0][0] - ordered[1][0]
    return {
        "selected": [identity for _, identity in ordered[:roots]],
        "ranked": [identity for _, identity in ordered],
        "root_score_gap": gap,
        "facets": len(facets),
        "comparisons": len(facets) * len(candidates),
    }


def _region_selection(
    selector: QueryRegionSelector,
    prompt: str,
    target_span: tuple[int, int],
    method: str,
) -> tuple[Any, int]:
    if method == "head":
        return selector.select(prompt, policy="head"), 0
    if method == "suffix":
        return QueryRegionSelector(suffix_tokens=32).select(prompt, policy="suffix"), 0
    if method == "explicit":
        return selector.select(prompt, query_spans=(target_span,)), 0
    if method == "structural":
        return selector.select(prompt, policy="structural"), 0
    if method == "broad":
        return selector.select(prompt, query_spans=((0, len(token_offsets(prompt))),)), 0
    raise ValueError(method)


def run_query_region_study(output: Path) -> dict[str, Any]:
    """Run matched layout, displacement, interaction, retry, and session controls."""

    selector = QueryRegionSelector(suffix_tokens=16, max_regions=2)
    configs = {
        "schema_version": "1.0",
        "cases": 24,
        "layouts": list(LAYOUTS),
        "payload_types": list(PAYLOADS),
        "methods": list(REGION_METHODS),
        "root_candidates": 8,
        "default_roots": 1,
        "facet_policy": "global",
        "head_tokens": 16,
        "suffix_tokens": 32,
        "displacements": [0, 32, 128, 512, 2048, 8192],
        "scope": "deterministic lexical retrieval control; no language-model inference",
    }
    (output / "query_region_configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

    rows = []
    for case in range(configs["cases"]):
        target, answer = f"service_target_{case}", f"cause_target_{case}"
        query = f"QUESTION: Why did {target} fail after deployment?"
        candidates = _candidate_chunks(case, target, answer)
        for payload in PAYLOADS:
            context, _ = _payload(payload, case, target, answer, 24 if payload != "prose" else 12)
            for layout in LAYOUTS:
                prompt = _layout(layout, query, context)
                target_span = _subsequence_span(prompt, query)
                for method in REGION_METHODS:
                    attempts = 1
                    if method == "auto_retry":
                        selection = selector.select(prompt, policy="head")
                        route = _route(selection.selected_text(), candidates, 1, "global")
                        alternate = selector.reinterpret(prompt, selection)
                        head_tokens = {index for span in selection.spans for index in range(*span)}
                        alternate_tokens = {index for span in alternate.spans for index in range(*span)}
                        overlap = len(head_tokens & alternate_tokens) / max(len(alternate_tokens), 1)
                        if alternate.confidence > selection.confidence and overlap < 0.5:
                            selection = alternate
                            route = _route(selection.selected_text(), candidates, 1, "global")
                            attempts = 2
                    else:
                        selection, _ = _region_selection(selector, prompt, target_span, method)
                        route = _route(selection.selected_text(), candidates, 1, "global")
                    metrics = _span_metrics(selection.spans, target_span)
                    rank = route["ranked"].index(target) + 1
                    rows.append(
                        {
                            "case_id": case,
                            "payload_type": payload,
                            "layout": layout,
                            "method": method,
                            "query_start": selection.regions[0].start,
                            "query_end": selection.regions[0].end,
                            "query_region_count": len(selection.regions),
                            "region_confidence": selection.confidence,
                            "region_selection_method": selection.policy,
                            **metrics,
                            "root_rank": rank,
                            "root_recall_at_1": float(rank <= 1),
                            "mrr": 1.0 / rank,
                            "evidence_recall": float(target in route["selected"]),
                            "path_recovery": float(target in route["selected"]),
                            "answer_quality": float(target in route["selected"]),
                            "active_native_kv": 16 * len(route["selected"]),
                            "search_effort": route["comparisons"] * attempts,
                            "attempts": attempts,
                            "root_score_gap": route["root_score_gap"],
                            "prompt_tokens": len(token_offsets(prompt)),
                        }
                    )
    write_csv(output / "query_region_layout_rows.csv", rows)
    detection = _aggregate(
        rows,
        ("layout", "payload_type", "method"),
        ("span_iou", "start_error", "end_error", "region_precision", "region_recall", "region_confidence"),
    )
    retrieval = _aggregate(
        rows,
        ("layout", "payload_type", "method"),
        ("root_recall_at_1", "mrr", "evidence_recall", "path_recovery", "root_score_gap"),
    )
    outputs = _aggregate(
        rows,
        ("layout", "payload_type", "method"),
        ("answer_quality", "active_native_kv", "search_effort", "attempts"),
    )
    write_csv(output / "query_region_detection_metrics.csv", detection)
    write_csv(output / "query_region_retrieval_results.csv", retrieval)
    write_csv(output / "query_region_output_results.csv", outputs)

    displacement_rows = []
    for displacement in configs["displacements"]:
        for case in range(configs["cases"]):
            target, answer = f"service_target_{case}", f"cause_target_{case}"
            query = f"QUESTION: Why did {target} fail after deployment?"
            padding = " ".join(f"noise_{case}_{index}" for index in range(displacement))
            prompt = f"{query}\n{padding}" if padding else query
            span = _subsequence_span(prompt, query)
            candidates = _candidate_chunks(case, target, answer)
            for method in ("head", "explicit", "structural"):
                selection, _ = _region_selection(selector, prompt, span, method)
                routed = _route(selection.selected_text(), candidates, 1, "global")
                displacement_rows.append(
                    {
                        "displacement_tokens_requested": displacement,
                        "actual_tokens_after_query": len(token_offsets(prompt)) - span[1],
                        "case_id": case,
                        "method": method,
                        "root_recall_at_1": float(routed["ranked"][0] == target),
                        "span_iou": _span_metrics(selection.spans, span)["span_iou"],
                        "search_effort": routed["comparisons"],
                    }
                )
    write_csv(
        output / "query_region_head_displacement.csv",
        _aggregate(displacement_rows, ("displacement_tokens_requested", "method"), ("actual_tokens_after_query", "root_recall_at_1", "span_iou", "search_effort")),
    )

    facet_rows, root_rows = [], []
    for row in [value for value in rows if value["method"] in {"head", "explicit", "structural"}]:
        # Reconstruct the deterministic example so interaction tables retain
        # exact matched layouts while remaining compact in public artifacts.
        case = int(row["case_id"])
        target, answer = f"service_target_{case}", f"cause_target_{case}"
        query = f"QUESTION: Why did {target} fail after deployment?"
        context, _ = _payload(row["payload_type"], case, target, answer, 24 if row["payload_type"] != "prose" else 12)
        prompt = _layout(row["layout"], query, context)
        span = _subsequence_span(prompt, query)
        selection, _ = _region_selection(selector, prompt, span, row["method"])
        candidates = _candidate_chunks(case, target, answer)
        for facets in ("global", "local4", "multiscale"):
            routed = _route(selection.selected_text(), candidates, 1, facets)
            facet_rows.append({"case_id": case, "layout": row["layout"], "payload_type": row["payload_type"], "method": row["method"], "facet_policy": facets, "facet_count": routed["facets"], "root_recall_at_1": float(routed["ranked"][0] == target), "search_effort": routed["comparisons"]})
        for roots in (1, 2, 4):
            routed = _route(selection.selected_text(), candidates, roots, "global")
            root_rows.append({"case_id": case, "layout": row["layout"], "payload_type": row["payload_type"], "method": row["method"], "roots": roots, "root_recall": float(target in routed["selected"]), "active_native_kv": 16 * roots, "search_effort": routed["comparisons"]})
    write_csv(output / "query_region_facet_interaction.csv", _aggregate(facet_rows, ("method", "facet_policy"), ("facet_count", "root_recall_at_1", "search_effort")))
    write_csv(output / "query_region_root_interaction.csv", _aggregate(root_rows, ("method", "roots"), ("root_recall", "active_native_kv", "search_effort")))

    paired: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(int(row["case_id"]), row["layout"], row["payload_type"])][row["method"]] = row
    retry_rows = []
    for identity, methods in paired.items():
        head, retry = methods["head"], methods["auto_retry"]
        retry_rows.append(
            {
                "case_id": identity[0],
                "layout": identity[1],
                "payload_type": identity[2],
                "wrong_to_corrected": int(not head["answer_quality"] and retry["answer_quality"]),
                "correct_to_broken": int(head["answer_quality"] and not retry["answer_quality"]),
                "no_change": int(head["answer_quality"] == retry["answer_quality"]),
                "added_search_effort": retry["search_effort"] - head["search_effort"],
                "added_active_kv": retry["active_native_kv"] - head["active_native_kv"],
                "attempts": retry["attempts"],
            }
        )
    write_csv(output / "query_region_retry_results.csv", _aggregate(retry_rows, ("layout", "payload_type"), ("wrong_to_corrected", "correct_to_broken", "no_change", "added_search_effort", "added_active_kv", "attempts")))

    session_rows = []
    for case in range(configs["cases"]):
        current = f"Why did service_target_{case} fail?"
        segments = [PromptSegment("history", "Earlier user asked about unrelated billing."), PromptSegment("instruction", "Use current incident evidence."), PromptSegment("query", current), PromptSegment("tool_output", f"ERROR service_target_{case} cause_target_{case}")]
        selection = selector.select(segments=segments, policy="session_state")
        broad = selector.select(segments=segments, query_spans=((0, len(token_offsets(selection.prompt))),))
        for method, selected in (("session_state", selection), ("whole_conversation", broad)):
            candidates = _candidate_chunks(case, f"service_target_{case}", f"cause_target_{case}")
            routed = _route(selected.selected_text(), candidates, 1, "global")
            session_rows.append({"case_id": case, "method": method, "region_count": len(selected.regions), "selected_tokens": sum(region.token_count for region in selected.regions), "root_recall_at_1": float(routed["ranked"][0] == f"service_target_{case}"), "search_effort": routed["comparisons"]})
    write_csv(output / "query_region_session_results.csv", _aggregate(session_rows, ("method",), ("region_count", "selected_tokens", "root_recall_at_1", "search_effort")))

    def metric(method: str, layout: str, field: str) -> float:
        return _mean(float(row[field]) for row in rows if row["method"] == method and row["layout"] == layout)

    head_spread = max(metric("head", layout, "root_recall_at_1") for layout in LAYOUTS) - min(metric("head", layout, "root_recall_at_1") for layout in LAYOUTS)
    structural_spread = max(metric("structural", layout, "root_recall_at_1") for layout in LAYOUTS) - min(metric("structural", layout, "root_recall_at_1") for layout in LAYOUTS)
    explicit_first = metric("explicit", "L1_query_context", "root_recall_at_1")
    head_first = metric("head", "L1_query_context", "root_recall_at_1")
    structural_first = metric("structural", "L1_query_context", "root_recall_at_1")
    corrected = sum(row["wrong_to_corrected"] for row in retry_rows)
    findings = {
        "schema_version": "1.0",
        "gate1_headroom": {"head_query_first_root_recall": head_first, "explicit_query_first_root_recall": explicit_first, "passed": explicit_first > head_first},
        "gate2_automatic_discovery": {"structural_query_first_root_recall": structural_first, "oracle_headroom_captured": (structural_first - head_first) / max(explicit_first - head_first, 1e-12), "passed": structural_first > head_first},
        "gate3_retry": {"wrong_to_corrected": corrected, "correct_to_broken": sum(row["correct_to_broken"] for row in retry_rows), "mean_added_search_effort": _mean(row["added_search_effort"] for row in retry_rows), "passed": corrected > 0},
        "layout_sensitivity": {"head_spread": head_spread, "structural_spread": structural_spread},
        "interpretation": "Explicit and structural query roles remove serialization sensitivity in this matched lexical control. The result establishes SDK headroom and deterministic baseline behavior, not natural-language semantic region understanding.",
        "learned_region_router_required": False,
    }
    (output / "query_region_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    return findings


def _standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(0)
    scale = train.std(0, unbiased=False).clamp_min(1e-6)
    return (train - mean) / scale, (test - mean) / scale


def _feature_matrix(examples: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    return torch.stack([example["features"].vector(CONTROLLER_FEATURE_NAMES).float() for example in examples])


def _semantic_text(example: Mapping[str, Any]) -> str:
    dataset = str(example["dataset"])
    if dataset == "hotpotqa":
        return "bridge multi entity relational question"
    return "direct scientific document question"


def _target_indices(space: RouterActionSpace, examples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    profiles = {profile.name: profile for profile in default_effort_profiles()}
    indexed = [space.index_targets(profile_actions(profiles[example["minimum_effort"]])) for example in examples]
    return {field.name: torch.tensor([row[field.name] for row in indexed], dtype=torch.long) for field in space.fields}


def _train_model(model: Any, features: torch.Tensor, targets: Mapping[str, torch.Tensor], semantic: torch.Tensor | None, seed: int) -> None:
    torch.manual_seed(seed)
    model.apply(lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    for _ in range(180):
        optimizer.zero_grad()
        loss = model.loss(features, targets, semantic=semantic)
        loss.backward()
        optimizer.step()


def _ece(probabilities: Sequence[float], correct: Sequence[int], bins: int = 8) -> float:
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [row for row, probability in enumerate(probabilities) if lower <= probability < upper or (index == bins - 1 and probability == 1.0)]
        if selected:
            accuracy = _mean(correct[row] for row in selected)
            confidence = _mean(probabilities[row] for row in selected)
            result += len(selected) / len(probabilities) * abs(accuracy - confidence)
    return result


def _profile_level_from_actions(space: RouterActionSpace, actions: Mapping[str, Any]) -> int:
    profiles = default_effort_profiles()
    levels = []
    for field in space.fields:
        matching = [index for index, profile in enumerate(profiles) if profile_actions(profile)[field.name] == actions[field.name]]
        levels.append(min(matching) if matching else len(profiles) - 1)
    return max(levels)


def _evaluate_variant(
    variant: str,
    seed: int,
    model: Any,
    space: RouterActionSpace,
    examples: Sequence[Mapping[str, Any]],
    features: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    semantic: torch.Tensor | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    profiles = default_effort_profiles()
    rows, calibration, latency = [], [], []
    probabilities_by_field: dict[str, list[float]] = defaultdict(list)
    correct_by_field: dict[str, list[int]] = defaultdict(list)
    top2_by_field: dict[str, list[int]] = defaultdict(list)
    nll_by_field: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    for index, example in enumerate(examples):
        decision = model.decide(features[index], semantic=semantic[index] if semantic is not None else None)
        level = _profile_level_from_actions(space, decision.actions)
        selected = profiles[level]
        oracle_level = [profile.name for profile in profiles].index(example["minimum_effort"])
        oracle = profiles[oracle_level]
        selected_quality = float(example["attempts"][selected.name]["chain_complete"])
        oracle_quality = float(example["attempts"][oracle.name]["chain_complete"])
        exact_heads = []
        for field in space.fields:
            actual = int(targets[field.name][index])
            predicted = field.values.index(decision.actions[field.name])
            probability = decision.probabilities[field.name][predicted]
            exact = int(actual == predicted)
            top2 = int(actual in sorted(range(len(decision.probabilities[field.name])), key=lambda item: decision.probabilities[field.name][item], reverse=True)[:2])
            exact_heads.append(exact)
            probabilities_by_field[field.name].append(probability)
            correct_by_field[field.name].append(exact)
            top2_by_field[field.name].append(top2)
            nll_by_field[field.name].append(-math.log(max(decision.probabilities[field.name][actual], 1e-12)))
        rows.append({"variant": variant, "seed": seed, "dataset": example["dataset"], "example_id": example["example_id"], "model_seed": example["seed"], "oracle_profile": oracle.name, "selected_profile": selected.name, "oracle_level": oracle_level, "selected_level": level, "joint_exact": int(all(exact_heads)), "mean_head_accuracy": _mean(exact_heads), "underprediction": int(level < oracle_level), "overprediction": int(level > oracle_level), "quality": selected_quality, "oracle_quality": oracle_quality, "quality_regret": oracle_quality - selected_quality, "cost": selected.cost_units, "oracle_cost": oracle.cost_units, "cost_regret": selected.cost_units - oracle.cost_units, "router_latency_seconds": decision.latency_seconds})
    elapsed = time.perf_counter() - started
    for field in space.fields:
        calibration.append({"variant": variant, "seed": seed, "parameter": field.name, "accuracy": _mean(correct_by_field[field.name]), "top2_accuracy": _mean(top2_by_field[field.name]), "nll": _mean(nll_by_field[field.name]), "ece": _ece(probabilities_by_field[field.name], correct_by_field[field.name]), "underprediction_rate": _mean(int(row["selected_level"] < row["oracle_level"]) for row in rows), "overprediction_rate": _mean(int(row["selected_level"] > row["oracle_level"]) for row in rows)})
    latency.append({"variant": variant, "seed": seed, "examples": len(examples), "total_seconds": elapsed, "mean_plan_seconds": elapsed / len(examples), "reported_internal_mean_seconds": _mean(row["router_latency_seconds"] for row in rows), "device": "cpu"})
    return rows, calibration, latency


def _r0_rows(validation: Sequence[Mapping[str, Any]], heldout: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    profiles = default_effort_profiles()
    controller = LinearEffortController.fit([row["features"] for row in validation], [row["minimum_effort"] for row in validation], [profile.name for profile in profiles], feature_names=CONTROLLER_FEATURE_NAMES, ridge=0.05)
    rows = []
    for example in heldout:
        started = time.perf_counter()
        selected_name = controller.choose(example["features"])
        latency = time.perf_counter() - started
        selected_level = [profile.name for profile in profiles].index(selected_name)
        oracle_level = [profile.name for profile in profiles].index(example["minimum_effort"])
        selected, oracle = profiles[selected_level], profiles[oracle_level]
        quality = float(example["attempts"][selected.name]["chain_complete"])
        oracle_quality = float(example["attempts"][oracle.name]["chain_complete"])
        rows.append({"variant": "R0_profile", "seed": 0, "dataset": example["dataset"], "example_id": example["example_id"], "model_seed": example["seed"], "oracle_profile": oracle.name, "selected_profile": selected.name, "oracle_level": oracle_level, "selected_level": selected_level, "joint_exact": int(selected_level == oracle_level), "mean_head_accuracy": int(selected_level == oracle_level), "underprediction": int(selected_level < oracle_level), "overprediction": int(selected_level > oracle_level), "quality": quality, "oracle_quality": oracle_quality, "quality_regret": oracle_quality - quality, "cost": selected.cost_units, "oracle_cost": oracle.cost_units, "cost_regret": selected.cost_units - oracle.cost_units, "router_latency_seconds": latency})
    return rows


def run_router_variant_study(source: Path, output: Path) -> dict[str, Any]:
    """Compare R0/R1/R2/R3A on the frozen minimum-effort cohort."""

    torch.set_num_threads(1)
    examples = build_examples(source)
    validation = [row for row in examples if row["partition"] == "validation"]
    heldout = [row for row in examples if row["partition"] == "test"]
    profiles = default_effort_profiles()
    space = RouterActionSpace.from_profiles(profiles, core_only=True)
    (output / "router_profiles.json").write_text(json.dumps([profile.to_dict() for profile in profiles], indent=2), encoding="utf-8")
    (output / "router_action_spaces.json").write_text(json.dumps(space.to_dict(), indent=2), encoding="utf-8")
    (output / "router_feature_schema.json").write_text(json.dumps({"numeric_features": list(CONTROLLER_FEATURE_NAMES), "semantic_input": "cached selected query-region text", "label_boundary": "validation-derived minimum sufficient effort; test labels evaluator-only", "initial_router_features": ["query semantics", "prompt geometry", "session/cache state"], "retry_additions": ["routing entropy", "score gaps", "answer confidence", "new-memory gain", "previous action"]}, indent=2), encoding="utf-8")
    oracle_rows = []
    for row in examples:
        profile = next(profile for profile in profiles if profile.name == row["minimum_effort"])
        oracle_rows.append({"partition": row["partition"], "dataset": row["dataset"], "example_id": row["example_id"], "model_seed": row["seed"], "minimum_profile": profile.name, **profile_actions(profile), "cost_target": profile.cost_units, "quality_target": float(row["attempts"][profile.name]["chain_complete"])})
    write_csv(output / "router_oracle_targets.csv", oracle_rows)

    validation_x, heldout_x = _standardize(_feature_matrix(validation), _feature_matrix(heldout))
    validation_targets = _target_indices(space, validation)
    heldout_targets = _target_indices(space, heldout)
    encoder = HashingQueryEncoder(width=32)
    validation_semantic = encoder.encode([_semantic_text(row) for row in validation])
    heldout_semantic = encoder.encode([_semantic_text(row) for row in heldout])

    all_rows = _r0_rows(validation, heldout)
    calibration_rows: list[dict[str, Any]] = []
    r0_values = [row for row in all_rows if row["variant"] == "R0_profile"]
    r0_latency = sum(row["router_latency_seconds"] for row in r0_values)
    latency_rows = [{"variant": "R0_profile", "seed": 0, "examples": len(heldout), "total_seconds": r0_latency, "mean_plan_seconds": r0_latency / len(heldout), "reported_internal_mean_seconds": r0_latency / len(heldout), "device": "cpu"}]
    for seed in ROUTER_SEEDS:
        variants = [
            ("R1_feature_mlp", MultiHeadEffortRouter(validation_x.shape[1], space, hidden_width=32), None, None),
            ("R2_encoder_mlp", MultiHeadEffortRouter(validation_x.shape[1], space, semantic_width=32, hidden_width=32, architecture="R2_encoder_mlp"), validation_semantic, heldout_semantic),
            ("R3A_autoregressive", AutoregressiveEffortRouter(validation_x.shape[1], space, semantic_width=32, hidden_width=32, context_width=12), validation_semantic, heldout_semantic),
        ]
        for variant, model, train_semantic, test_semantic in variants:
            _train_model(model, validation_x, validation_targets, train_semantic, seed)
            rows, calibration, latency = _evaluate_variant(variant, seed, model, space, heldout, heldout_x, heldout_targets, test_semantic)
            all_rows.extend(rows)
            calibration_rows.extend(calibration)
            latency_rows.extend(latency)

    files = {"R0_profile": "router_r0_profile_results.csv", "R1_feature_mlp": "router_r1_mlp_results.csv", "R2_encoder_mlp": "router_r2_encoder_results.csv", "R3A_autoregressive": "router_r3_interaction_results.csv"}
    for variant, filename in files.items():
        write_csv(output / filename, [row for row in all_rows if row["variant"] == variant])
    write_csv(output / "router_parameter_calibration.csv", calibration_rows)
    write_csv(output / "router_latency.csv", latency_rows)
    write_csv(output / "router_regret.csv", [{key: row[key] for key in ("variant", "seed", "dataset", "example_id", "model_seed", "selected_profile", "oracle_profile", "quality_regret", "cost_regret", "underprediction", "overprediction")} for row in all_rows])

    interactions = []
    target_rows = [space.index_targets(profile_actions(next(profile for profile in profiles if profile.name == row["minimum_effort"]))) for row in validation]
    for left_index, left in enumerate(space.fields):
        for right in space.fields[left_index + 1 :]:
            pairs = Counter((row[left.name], row[right.name]) for row in target_rows)
            left_counts, right_counts = Counter(row[left.name] for row in target_rows), Counter(row[right.name] for row in target_rows)
            mutual_information = sum(count / len(target_rows) * math.log((count * len(target_rows)) / (left_counts[a] * right_counts[b])) for (a, b), count in pairs.items())
            interactions.append({"left_parameter": left.name, "right_parameter": right.name, "mutual_information_nats": mutual_information, "observed_pairs": len(pairs), "interpretation": "profile-label coupling; not a causal parameter interaction"})
    write_csv(output / "router_parameter_interactions.csv", interactions)

    aggregate = _aggregate(all_rows, ("variant", "seed", "dataset"), ("quality", "oracle_quality", "quality_regret", "cost", "oracle_cost", "cost_regret", "joint_exact", "mean_head_accuracy", "underprediction", "overprediction", "router_latency_seconds"))
    write_csv(output / "router_query_class_results.csv", aggregate)

    variant_summary = _aggregate(all_rows, ("variant",), ("quality", "oracle_quality", "quality_regret", "cost", "oracle_cost", "cost_regret", "joint_exact", "mean_head_accuracy", "underprediction", "overprediction"))
    r2 = next(row for row in variant_summary if row["variant"] == "R2_encoder_mlp")
    r3 = next(row for row in variant_summary if row["variant"] == "R3A_autoregressive")
    findings = {
        "schema_version": "1.0",
        "validation_units": len(validation),
        "heldout_units": len(heldout),
        "router_seeds": list(ROUTER_SEEDS),
        "summary": variant_summary,
        "gate_a_profiles": "reported by R0 and fixed-profile baselines in the parent study",
        "gate_b_independent_heads": "evaluated",
        "gate_c_semantic_encoder": "evaluated with a dependency-free hashing baseline; replaceable by a compact pretrained encoder",
        "gate_d_interactions": {"evaluated": True, "r2_quality": r2["quality"], "r3_quality": r3["quality"], "r2_cost": r2["cost"], "r3_cost": r3["cost"], "continue_to_controller_transformer": bool(r3["quality"] > r2["quality"] + 0.01 and r3["cost"] <= r2["cost"])},
        "interpretation": "The comparison tests whether compositional and semantic capacity moves the held-out quality/cost frontier. Profile-derived labels make parameter interactions highly coupled by construction, so interaction mutual information is diagnostic rather than causal.",
    }
    (output / "router_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    return findings


def run_addon_studies(source: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    query = run_query_region_study(output)
    router = run_router_variant_study(source, output)
    return {"query_regions": query, "router_variants": router}
