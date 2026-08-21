"""Staged self-encoded query-router experiment for Paper 3.5.

Representation gates use grouped cross-validation inside the validation split.
The held-out split is evaluated only after representation and controller
codebooks are fixed.  Retrieval outcomes replay the existing frozen factorized
surface; no generated-answer quality is imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_5_iterative_pra.run_semantic_graph_search import (  # noqa: E402
    _prepare_examples,
)
from experiments.paper3_5_adaptive_pra.factorized_study import (  # noqa: E402
    ROUTER_SEEDS,
    UnitSurface,
    _action_from_values,
    _action_space,
    _best_available,
    _existing_or_sibling,
    _mean,
    _row_for_action,
    _standardize,
    _targets,
    _write_csv,
    build_oracle_artifacts,
    build_surfaces,
)
from pra_hf.effort_router import (  # noqa: E402
    ActionField,
    AutoregressiveEffortRouter,
    HashingQueryEncoder,
    MultiHeadEffortRouter,
    RouterActionSpace,
)
from pra_hf.factorized_control import (  # noqa: E402
    BUDGET_LEVELS,
    FACET_LEVELS,
    FactorizedEffortAction,
    allocation_outcome,
)
from pra_hf.self_router import (  # noqa: E402
    QueryPrefillAccounting,
    ValidationProjector,
    decode_grouped_action,
    reuse_is_semantically_valid,
)


PROFILE_EPOCHS = 140
FACTOR_EPOCHS = 220
CV_FOLDS = 4
PROJECTION_WIDTH = 16
FIRST_MEMORY_LAYER = 24


class ProfileRouter(nn.Module):
    """Small shared profile head used for every representation source."""

    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, classes)
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.network(rows.float())


def _train_profile(
    features: torch.Tensor, targets: torch.Tensor, classes: int, seed: int
) -> ProfileRouter:
    torch.manual_seed(seed)
    model = ProfileRouter(features.shape[1], classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    for _ in range(PROFILE_EPOCHS):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(features), targets)
        loss.backward()
        optimizer.step()
    return model.eval()


def _train_factor_router(model: nn.Module, features, targets, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model.apply(
        lambda module: module.reset_parameters()
        if hasattr(module, "reset_parameters")
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    for _ in range(FACTOR_EPOCHS):
        optimizer.zero_grad()
        loss = model.loss(features, targets)
        loss.backward()
        optimizer.step()
    return model.eval()


def _representation_metadata(feature_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    sample = feature_rows[0]["representations"]
    metadata = {
        "S0_observable": {
            "family": "observable",
            "pooling": "none",
            "depth": 0,
            "contextual": False,
        },
        "S1_signed_hash": {
            "family": "signed_hash",
            "pooling": "none",
            "depth": 0,
            "contextual": False,
        },
    }
    for name, value in sample.items():
        metadata[name] = {
            "family": value["family"],
            "pooling": value["pooling"],
            "depth": int(value["depth"]),
            "contextual": name.startswith("S8_context_"),
        }
    return metadata


def _feature_map(feature_rows: Sequence[Mapping[str, Any]], name: str) -> dict[tuple[str, str], torch.Tensor]:
    if name == "S1_signed_hash":
        encoder = HashingQueryEncoder(width=32)
        vectors = encoder.encode([str(row["question"]) for row in feature_rows])
        return {
            (str(row["dataset"]), str(row["example_id"])): vectors[index]
            for index, row in enumerate(feature_rows)
        }
    return {
        (str(row["dataset"]), str(row["example_id"])): row["representations"][name]["vector"].float()
        for row in feature_rows
    }


def _unique_representation_rows(
    units: Sequence[UnitSurface], vectors: Mapping[tuple[str, str], torch.Tensor]
) -> torch.Tensor:
    seen = set()
    rows = []
    for unit in units:
        identity = (unit.dataset, unit.example_id)
        if identity not in seen:
            rows.append(vectors[identity])
            seen.add(identity)
    return torch.stack(rows)


def _controller_features(
    training: Sequence[UnitSurface],
    evaluation: Sequence[UnitSurface],
    feature_rows: Sequence[Mapping[str, Any]],
    representation: str,
    *,
    query_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_observable, eval_observable = _standardize(
        torch.stack([unit.features for unit in training]),
        torch.stack([unit.features for unit in evaluation]),
    )
    if representation == "S0_observable":
        return train_observable, eval_observable
    vectors = _feature_map(feature_rows, representation)
    projector = ValidationProjector(PROJECTION_WIDTH).fit(
        _unique_representation_rows(training, vectors),
        ["validation"] * len({(unit.dataset, unit.example_id) for unit in training}),
    )
    train_semantic = projector.transform(
        torch.stack([vectors[(unit.dataset, unit.example_id)] for unit in training])
    )
    eval_semantic = projector.transform(
        torch.stack([vectors[(unit.dataset, unit.example_id)] for unit in evaluation])
    )
    if query_only:
        return train_semantic, eval_semantic
    return (
        torch.cat((train_observable, train_semantic), dim=1),
        torch.cat((eval_observable, eval_semantic), dim=1),
    )


def _profile_targets(units: Sequence[UnitSurface]) -> torch.Tensor:
    targets = []
    for unit in units:
        rows = [
            _row_for_action(unit, FactorizedEffortAction.profile(level))
            for level in range(3)
        ]
        best = _best_available(rows)
        targets.append(
            next(
                level
                for level in range(3)
                if FactorizedEffortAction.profile(level).identifier == best["config_id"]
            )
        )
    return torch.tensor(targets, dtype=torch.long)


def _representation_overhead(
    feature: Mapping[str, Any], metadata: Mapping[str, Any], *, reused: bool = False
) -> dict[str, float | int]:
    name = str(metadata.get("name", ""))
    if name in {"S0_observable", "S1_signed_hash"}:
        return {
            "prefill_cost": 0.0,
            "prefill_latency_seconds": 0.0,
            "processed_token_layers": 0,
            "recomputed_query_tokens": 0,
        }
    contextual = bool(metadata["contextual"])
    prompt_tokens = int(
        feature["context_prompt_tokens"] if contextual else feature["query_prompt_tokens"]
    )
    accounting = QueryPrefillAccounting(
        prompt_tokens,
        int(feature["query_tokens"]),
        int(metadata["depth"]),
        28,
        reused=reused,
    )
    value = feature["representations"][name]
    return {
        "prefill_cost": accounting.normalized_cost,
        "prefill_latency_seconds": float(value["prefill_latency_seconds"]),
        "processed_token_layers": accounting.processed_token_layers,
        "recomputed_query_tokens": accounting.recomputed_query_tokens,
    }


def _feature_by_identity(feature_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["dataset"]), str(row["example_id"])): row for row in feature_rows
    }


def load_factorized_oracles(
    path: Path,
    expected_keys: set[tuple[str, str, int]] | None = None,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load compact evaluator targets with identity and schema checks."""

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    oracles: dict[tuple[str, str, int], dict[str, Any]] = {}
    required = {
        "dataset",
        "example_id",
        "seed",
        "config_id",
        "chain_complete",
        "abstract_cost",
        *FactorizedEffortAction.__dataclass_fields__,
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Factorized oracle row is missing fields: {sorted(missing)}")
        key = (row["dataset"], row["example_id"], int(row["seed"]))
        if key in oracles:
            raise ValueError(f"Duplicate factorized oracle identity: {key}")
        converted = dict(row)
        for name in FactorizedEffortAction.__dataclass_fields__:
            converted[name] = int(row[name])
        for name in ("chain_complete", "abstract_cost"):
            converted[name] = float(row[name])
        # Construction validates every categorical value and budget invariant.
        FactorizedEffortAction(
            *(converted[name] for name in FactorizedEffortAction.__dataclass_fields__)
        )
        oracles[key] = converted
    if expected_keys is not None and set(oracles) != expected_keys:
        raise ValueError("Factorized oracle CSV does not match the frozen unit identities.")
    return oracles


def _evaluate_action(
    unit: UnitSurface,
    action: FactorizedEffortAction,
    oracle: Mapping[str, Any],
    overhead: Mapping[str, float | int],
) -> dict[str, Any]:
    selected = _row_for_action(unit, action)
    retrieval_cost = float(selected["abstract_cost"])
    total_cost = retrieval_cost + float(overhead["prefill_cost"])
    return {
        "selected_config": action.identifier,
        "oracle_config": oracle["config_id"],
        "quality": float(selected["chain_complete"]),
        "retrieval_cost": retrieval_cost,
        "total_cost": total_cost,
        "oracle_cost": float(oracle["abstract_cost"]),
        "oracle_regret": total_cost - float(oracle["abstract_cost"]),
        "allocation_outcome": allocation_outcome(selected, oracle),
        **overhead,
    }


def _fold(example_id: str) -> int:
    return int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16) % CV_FOLDS


def _aggregate(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, selected in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result.append(
            {
                **dict(zip(keys, key)),
                "quality": _mean(selected, "quality"),
                "retrieval_cost": _mean(selected, "retrieval_cost"),
                "total_cost": _mean(selected, "total_cost"),
                "oracle_regret": _mean(selected, "oracle_regret"),
                "under_allocation": statistics.fmean(
                    row["allocation_outcome"] == "under_allocation" for row in selected
                ),
                "over_allocation": statistics.fmean(
                    row["allocation_outcome"] == "over_allocation" for row in selected
                ),
                "prefill_cost": _mean(selected, "prefill_cost"),
                "prefill_latency_seconds": _mean(selected, "prefill_latency_seconds"),
                "rows": len(selected),
            }
        )
    return result


def run_profile_gate(
    units: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation = [unit for unit in units if unit.partition == "validation"]
    heldout = [unit for unit in units if unit.partition == "test"]
    by_identity = _feature_by_identity(feature_rows)
    cv_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    for representation in metadata:
        rep_meta = {**metadata[representation], "name": representation}
        for router_seed in ROUTER_SEEDS:
            for fold in range(CV_FOLDS):
                train = [unit for unit in validation if _fold(unit.example_id) != fold]
                evaluate = [unit for unit in validation if _fold(unit.example_id) == fold]
                if not train or not evaluate:
                    continue
                train_x, eval_x = _controller_features(
                    train, evaluate, feature_rows, representation
                )
                model = _train_profile(
                    train_x, _profile_targets(train), 3, router_seed + fold * 1000
                )
                with torch.no_grad():
                    predictions = torch.argmax(model(eval_x), dim=1)
                for index, unit in enumerate(evaluate):
                    action = FactorizedEffortAction.profile(int(predictions[index]))
                    overhead = _representation_overhead(
                        by_identity[(unit.dataset, unit.example_id)], rep_meta
                    )
                    cv_rows.append(
                        {
                            "partition": "validation_cv",
                            "representation": representation,
                            "router_seed": router_seed,
                            "dataset": unit.dataset,
                            "example_id": unit.example_id,
                            "model_seed": unit.seed,
                            **_evaluate_action(unit, action, oracles[unit.key], overhead),
                        }
                    )

            train_x, test_x = _controller_features(
                validation, heldout, feature_rows, representation
            )
            model = _train_profile(
                train_x, _profile_targets(validation), 3, router_seed
            )
            with torch.no_grad():
                predictions = torch.argmax(model(test_x), dim=1)
            for index, unit in enumerate(heldout):
                action = FactorizedEffortAction.profile(int(predictions[index]))
                overhead = _representation_overhead(
                    by_identity[(unit.dataset, unit.example_id)], rep_meta
                )
                heldout_rows.append(
                    {
                        "partition": "test",
                        "target_architecture": "global_frozen_profile",
                        "representation": representation,
                        "router_seed": router_seed,
                        "dataset": unit.dataset,
                        "example_id": unit.example_id,
                        "model_seed": unit.seed,
                        **_evaluate_action(unit, action, oracles[unit.key], overhead),
                    }
                )
    cv_summary = _aggregate(cv_rows, ("representation",))
    heldout_summary = _aggregate(
        heldout_rows, ("representation", "router_seed", "dataset")
    )
    _write_csv(output / "self_router_profile_results.csv", heldout_summary)
    _write_csv(output / "self_router_under_over_allocation.csv", heldout_rows)
    return cv_summary, heldout_rows


def _select_self_representations(
    cv_summary: Sequence[Mapping[str, Any]], metadata: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    candidates = [
        row
        for row in cv_summary
        if row["representation"].startswith(("S2_", "S3_", "S4_", "S5_", "S6_", "S7_"))
    ]
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row["quality"]),
            float(row["total_cost"]),
            float(row["under_allocation"]),
            str(row["representation"]),
        ),
    )
    selected: list[str] = []
    for row in ordered:
        name = str(row["representation"])
        family = str(metadata[name]["family"])
        if family not in {str(metadata[value]["family"]) for value in selected} or len(selected) < 2:
            selected.append(name)
        if len(selected) == 3:
            break
    return selected


def run_feature_ablation(
    units: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    representation: str,
    output: Path,
) -> list[dict[str, Any]]:
    """Compare query state alone with query plus pre-search observables."""

    validation = [unit for unit in units if unit.partition == "validation"]
    heldout = [unit for unit in units if unit.partition == "test"]
    by_identity = _feature_by_identity(feature_rows)
    rep_meta = {**metadata[representation], "name": representation}
    rows = []
    for mode, query_only in (("query_only", True), ("query_plus_static", False)):
        train_x, test_x = _controller_features(
            validation,
            heldout,
            feature_rows,
            representation,
            query_only=query_only,
        )
        for router_seed in ROUTER_SEEDS:
            model = _train_profile(
                train_x, _profile_targets(validation), 3, router_seed
            )
            with torch.no_grad():
                predictions = torch.argmax(model(test_x), dim=1)
            for index, unit in enumerate(heldout):
                overhead = _representation_overhead(
                    by_identity[(unit.dataset, unit.example_id)], rep_meta
                )
                action = FactorizedEffortAction.profile(int(predictions[index]))
                rows.append(
                    {
                        "feature_mode": mode,
                        "representation": representation,
                        "router_seed": router_seed,
                        "dataset": unit.dataset,
                        **_evaluate_action(unit, action, oracles[unit.key], overhead),
                    }
                )
    summary = _aggregate(rows, ("feature_mode", "dataset"))
    summary.append(
        {
            "feature_mode": "query_plus_runtime",
            "dataset": "all",
            "status": "retry_only_not_eligible_for_initial_router",
            "quality": "",
            "retrieval_cost": "",
            "total_cost": "",
            "oracle_regret": "",
            "under_allocation": "",
            "over_allocation": "",
            "prefill_cost": "",
            "prefill_latency_seconds": "",
            "rows": 0,
        }
    )
    _write_csv(output / "self_router_feature_ablation.csv", summary)
    return rows


def _global_codebook(
    validation: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    count: int = 4,
) -> tuple[FactorizedEffortAction, ...]:
    frequencies = Counter(str(oracles[unit.key]["config_id"]) for unit in validation)
    actions = {
        row["config_id"]: FactorizedEffortAction(
            int(row["facets"]),
            int(row["roots"]),
            int(row["neighbors"]),
            int(row["hops"]),
            int(row["search_budget"]),
            int(row["kv_budget"]),
        )
        for unit in validation
        for row in unit.rows
    }
    ordered = sorted(
        frequencies,
        key=lambda identifier: (
            -frequencies[identifier],
            statistics.fmean(
                float(_row_for_action(unit, actions[identifier])["abstract_cost"])
                for unit in validation
            ),
            identifier,
        ),
    )
    return tuple(actions[identifier] for identifier in ordered[:count])


def _search_codebook(
    validation: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    count: int = 5,
) -> tuple[tuple[int, int, int, int], ...]:
    values = Counter(
        (
            int(oracles[unit.key]["roots"]),
            int(oracles[unit.key]["neighbors"]),
            int(oracles[unit.key]["hops"]),
            int(oracles[unit.key]["search_budget"]),
        )
        for unit in validation
    )
    return tuple(value for value, _ in values.most_common(count))


def _global_targets(units: Sequence[UnitSurface], codebook) -> torch.Tensor:
    values = []
    for unit in units:
        best = _best_available([_row_for_action(unit, action) for action in codebook])
        values.append(next(i for i, action in enumerate(codebook) if action.identifier == best["config_id"]))
    return torch.tensor(values, dtype=torch.long)


def _group_space(search_codebook) -> RouterActionSpace:
    return RouterActionSpace(
        (
            ActionField("interpret", FACET_LEVELS),
            ActionField("search", tuple(search_codebook), ordered=False),
            ActionField("admit", BUDGET_LEVELS),
        )
    )


def _group_targets(
    units: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    search_codebook,
) -> dict[str, torch.Tensor]:
    interpret, search, admit = [], [], []
    for unit in units:
        oracle = oracles[unit.key]
        interpret.append(FACET_LEVELS.index(int(oracle["facets"])))
        admit.append(BUDGET_LEVELS.index(int(oracle["kv_budget"])))
        candidates = []
        for value in search_codebook:
            action, _ = decode_grouped_action(
                int(oracle["facets"]), value, int(oracle["kv_budget"])
            )
            candidates.append(_row_for_action(unit, action))
        best = _best_available(candidates)
        search.append(
            next(
                i
                for i, value in enumerate(search_codebook)
                if decode_grouped_action(
                    int(oracle["facets"]), value, int(oracle["kv_budget"])
                )[0].identifier
                == best["config_id"]
            )
        )
    return {
        "interpret": torch.tensor(interpret),
        "search": torch.tensor(search),
        "admit": torch.tensor(admit),
    }


def run_target_gate(
    units: Sequence[UnitSurface],
    oracles: Mapping[tuple[str, str, int], Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    representation: str,
    output: Path,
) -> list[dict[str, Any]]:
    validation = [unit for unit in units if unit.partition == "validation"]
    heldout = [unit for unit in units if unit.partition == "test"]
    train_x, test_x = _controller_features(
        validation, heldout, feature_rows, representation
    )
    by_identity = _feature_by_identity(feature_rows)
    rep_meta = {**metadata[representation], "name": representation}
    global_book = _global_codebook(validation, oracles)
    search_book = _search_codebook(validation, oracles)
    group_space = _group_space(search_book)
    factor_space = _action_space()
    global_targets = _global_targets(validation, global_book)
    group_targets = _group_targets(validation, oracles, search_book)
    factor_targets = _targets(factor_space, validation, oracles)
    rows = []
    for router_seed in ROUTER_SEEDS:
        global_model = _train_profile(
            train_x, global_targets, len(global_book), router_seed
        )
        group_model = _train_factor_router(
            MultiHeadEffortRouter(train_x.shape[1], group_space, hidden_width=32),
            train_x,
            group_targets,
            router_seed,
        )
        independent = _train_factor_router(
            MultiHeadEffortRouter(train_x.shape[1], factor_space, hidden_width=32),
            train_x,
            factor_targets,
            router_seed,
        )
        interaction = _train_factor_router(
            AutoregressiveEffortRouter(
                train_x.shape[1], factor_space, hidden_width=32, context_width=12
            ),
            train_x,
            factor_targets,
            router_seed,
        )
        with torch.no_grad():
            global_predictions = torch.argmax(global_model(test_x), dim=1)
        for index, unit in enumerate(heldout):
            overhead = _representation_overhead(
                by_identity[(unit.dataset, unit.example_id)], rep_meta
            )
            variants = {
                "global_profile": (global_book[int(global_predictions[index])], False),
            }
            group_decision = group_model.decide(test_x[index])
            variants["group_profiles"] = decode_grouped_action(
                int(group_decision.actions["interpret"]),
                group_decision.actions["search"],
                int(group_decision.actions["admit"]),
            )
            independent_decision = independent.decide(test_x[index])
            variants["independent_params"] = _action_from_values(
                independent_decision.actions
            )
            interaction_decision = interaction.decide(test_x[index])
            variants["interaction_params"] = _action_from_values(
                interaction_decision.actions
            )
            for architecture, (action, repaired) in variants.items():
                rows.append(
                    {
                        "partition": "test",
                        "target_architecture": architecture,
                        "representation": representation,
                        "router_seed": router_seed,
                        "dataset": unit.dataset,
                        "example_id": unit.example_id,
                        "model_seed": unit.seed,
                        "invalid_combination_repaired": int(repaired),
                        **_evaluate_action(unit, action, oracles[unit.key], overhead),
                    }
                )
    summary = _aggregate(rows, ("target_architecture", "dataset"))
    for row in summary:
        selected = [
            item
            for item in rows
            if item["target_architecture"] == row["target_architecture"]
            and item["dataset"] == row["dataset"]
        ]
        row["invalid_repair_rate"] = _mean(selected, "invalid_combination_repaired")
    _write_csv(output / "self_router_target_architecture_summary.csv", summary)
    _write_csv(
        output / "self_router_group_profile_results.csv",
        [row for row in rows if row["target_architecture"] == "group_profiles"],
    )
    _write_csv(
        output / "self_router_parameter_results.csv",
        [row for row in rows if row["target_architecture"] == "independent_params"],
    )
    _write_csv(
        output / "self_router_interaction_results.csv",
        [row for row in rows if row["target_architecture"] == "interaction_params"],
    )
    codebooks = {
        "fit_partition": "validation",
        "global_profiles": [action.to_dict() for action in global_book],
        "search_profiles": [
            {"roots": r, "neighbors": k, "hops": h, "search_budget": budget}
            for r, k, h, budget in search_book
        ],
    }
    (output / "self_router_target_codebooks.json").write_text(
        json.dumps(codebooks, indent=2), encoding="utf-8"
    )
    return rows


def build_drift(
    feature_rows: Sequence[Mapping[str, Any]], output: Path
) -> list[dict[str, Any]]:
    pairs = (
        ("S4_hidden_l14_mean", "S8_context_hidden_l14_mean"),
        ("S5_hidden_l28_mean", "S8_context_hidden_l28_mean"),
        ("S6_native_q_l13_mean", "S8_context_native_q_l13_mean"),
        ("S6_native_q_l23_mean", "S8_context_native_q_l23_mean"),
    )
    rows = []
    for feature in feature_rows:
        for query_name, context_name in pairs:
            left = feature["representations"][query_name]["vector"].float()
            right = feature["representations"][context_name]["vector"].float()
            rows.append(
                {
                    "dataset": feature["dataset"],
                    "example_id": feature["example_id"],
                    "query_representation": query_name,
                    "contextual_representation": context_name,
                    "depth": feature["representations"][query_name]["depth"],
                    "cosine_similarity": float(
                        torch.nn.functional.cosine_similarity(left, right, dim=0)
                    ),
                    "cosine_drift": 1.0
                    - float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
                }
            )
    _write_csv(output / "self_router_query_context_drift.csv", rows)
    return rows


def build_accounting(
    selected_representation: str,
    feature_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    by_identity = _feature_by_identity(feature_rows)
    selected = [row for row in profile_rows if row["representation"] == selected_representation]
    base_quality = _mean(selected, "quality")
    base_retrieval = _mean(selected, "retrieval_cost")
    meta = {**metadata[selected_representation], "name": selected_representation}
    overheads = [
        _representation_overhead(row, meta)
        for row in by_identity.values()
        if any(
            item["dataset"] == row["dataset"] and item["example_id"] == row["example_id"]
            for item in selected
        )
    ]
    reusable = reuse_is_semantically_valid(
        depth=int(meta["depth"]),
        first_memory_layer=FIRST_MEMORY_LAYER,
        ordinary_context_precedes_query=False,
    )
    rows = [
        {
            "mode": "D0_double_query_prefill",
            "representation": selected_representation,
            "supported": 1,
            "quality": base_quality,
            "retrieval_cost": base_retrieval,
            "added_prefill_cost": _mean(overheads, "prefill_cost"),
            "total_cost": base_retrieval + _mean(overheads, "prefill_cost"),
            "router_latency_seconds": _mean(overheads, "prefill_latency_seconds"),
            "recomputed_query_tokens": _mean(overheads, "recomputed_query_tokens"),
            "context_tokens_processed": 0,
            "parity_status": "fresh normal execution; prepass does not mutate model state",
        },
        {
            "mode": "D1_reused_query_prefill",
            "representation": selected_representation,
            "supported": int(reusable),
            "quality": base_quality if reusable else "",
            "retrieval_cost": base_retrieval if reusable else "",
            "added_prefill_cost": 0.0 if reusable else "",
            "total_cost": base_retrieval if reusable else "",
            "router_latency_seconds": _mean(overheads, "prefill_latency_seconds"),
            "recomputed_query_tokens": 0 if reusable else "",
            "context_tokens_processed": 0,
            "parity_status": "exact prefix continuation tested; valid only before first PRA consumer"
            if reusable
            else "representation is deeper than first PRA consumer",
        },
        {
            "mode": "D2_contextual_upper_bound",
            "representation": "best_S8_contextual",
            "supported": 1,
            "quality": "reported_in_contextual_upper_bound_csv",
            "retrieval_cost": "reported_in_contextual_upper_bound_csv",
            "added_prefill_cost": "charged_per_contextual_representation",
            "total_cost": "reported_in_contextual_upper_bound_csv",
            "router_latency_seconds": "reported_in_contextual_upper_bound_csv",
            "recomputed_query_tokens": 0,
            "context_tokens_processed": statistics.fmean(
                int(row["context_prompt_tokens"]) for row in by_identity.values()
            ),
            "parity_status": "diagnostic only; ordinary source-context processing changes the architecture",
        },
        {
            "mode": "D3_external_encoder",
            "representation": "not_run",
            "supported": 0,
            "quality": "",
            "retrieval_cost": "",
            "added_prefill_cost": "",
            "total_cost": "",
            "router_latency_seconds": "",
            "recomputed_query_tokens": 0,
            "context_tokens_processed": 0,
            "parity_status": "deferred by staged stop rule pending self-encoding headroom",
        },
    ]
    _write_csv(output / "self_router_double_vs_reuse.csv", rows)
    _write_csv(
        output / "self_router_external_encoder_results.csv",
        [
            {
                "status": "not_run",
                "gate": 5,
                "reason": "external encoder is permitted only after self-encoding headroom is established",
            }
        ],
    )
    return rows


def _plot(
    profile_summary: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    drift_rows: Sequence[Mapping[str, Any]],
    accounting_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> None:
    heldout = _aggregate(profile_summary, ("representation",))
    self_rows = [
        row for row in heldout if str(row["representation"]).startswith(("S2_", "S3_", "S4_", "S5_", "S6_", "S7_"))
    ]
    figures = []

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.scatter(
        [metadata[row["representation"]]["depth"] for row in self_rows],
        [row["quality"] for row in self_rows],
        c=[row["total_cost"] for row in self_rows],
        cmap="viridis",
        s=48,
    )
    axis.set(xlabel="query-prefill depth", ylabel="held-out retrieval sufficiency")
    axis.grid(alpha=0.25)
    figures.append((figure, "self_router_quality_by_layer"))

    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.scatter(
        [metadata[row["representation"]]["depth"] for row in self_rows],
        [row["total_cost"] for row in self_rows],
        c=[row["quality"] for row in self_rows],
        cmap="plasma",
        s=48,
    )
    axis.set(xlabel="query-prefill depth", ylabel="total abstract cost")
    axis.grid(alpha=0.25)
    figures.append((figure, "self_router_cost_by_layer"))

    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    ordered = sorted(heldout, key=lambda row: float(row["oracle_regret"]))
    axis.barh(
        [str(row["representation"]).replace("_", " ") for row in ordered],
        [row["oracle_regret"] for row in ordered],
    )
    axis.set_xlabel("oracle regret including representation cost")
    axis.grid(axis="x", alpha=0.25)
    figures.append((figure, "self_router_oracle_regret"))

    target = _aggregate(target_rows, ("target_architecture",))
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    annotation_offsets = {
        "independent_params": (5, 8),
        "interaction_params": (5, -12),
    }
    for row in target:
        axis.scatter(row["total_cost"], row["quality"], s=70)
        label = str(row["target_architecture"]).replace("_", " ")
        axis.annotate(
            label,
            (row["total_cost"], row["quality"]),
            xytext=annotation_offsets.get(row["target_architecture"], (5, 3)),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(xlabel="total abstract cost", ylabel="held-out retrieval sufficiency")
    axis.grid(alpha=0.25)
    figures.append((figure, "self_router_target_frontier"))

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    drift_summary = _aggregate(
        [
            {
                **row,
                "quality": row["cosine_similarity"],
                "retrieval_cost": 0,
                "total_cost": 0,
                "oracle_regret": 0,
                "allocation_outcome": "matched",
                "prefill_cost": 0,
                "prefill_latency_seconds": 0,
            }
            for row in drift_rows
        ],
        ("query_representation",),
    )
    axis.bar(
        range(len(drift_summary)),
        [row["quality"] for row in drift_summary],
        tick_label=[str(row["query_representation"]).replace("_mean", "").replace("_", "\n") for row in drift_summary],
    )
    axis.set_ylabel("query-only/contextual cosine similarity")
    axis.tick_params(axis="x", labelsize=7)
    axis.grid(axis="y", alpha=0.25)
    figures.append((figure, "self_router_query_context_drift"))

    for dataset in ("hotpotqa", "qasper"):
        selected = _aggregate(
            [row for row in profile_summary if row["dataset"] == dataset],
            ("representation",),
        )
        figure, axis = plt.subplots(figsize=(7.6, 4.5))
        axis.scatter(
            [row["total_cost"] for row in selected],
            [row["quality"] for row in selected],
            s=42,
        )
        for row in sorted(selected, key=lambda value: (-float(value["quality"]), float(value["total_cost"])))[:5]:
            axis.annotate(str(row["representation"]), (row["total_cost"], row["quality"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        axis.set(xlabel="total abstract cost", ylabel="retrieval sufficiency", title=dataset.upper())
        axis.grid(alpha=0.25)
        figures.append((figure, f"self_router_{dataset}_frontier"))

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.scatter(
        [row["total_cost"] for row in heldout],
        [row["under_allocation"] for row in heldout],
        c=[row["quality"] for row in heldout],
        cmap="viridis",
        s=45,
    )
    axis.set(xlabel="total abstract cost", ylabel="under-allocation rate")
    axis.grid(alpha=0.25)
    figures.append((figure, "self_router_underallocation_cost"))

    numeric_accounting = [row for row in accounting_rows if isinstance(row["total_cost"], (int, float))]
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.bar(
        [str(row["mode"]).replace("_", "\n") for row in numeric_accounting],
        [row["total_cost"] for row in numeric_accounting],
    )
    axis.set_ylabel("total abstract cost")
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="y", alpha=0.25)
    figures.append((figure, "self_router_double_vs_reuse"))

    for figure, name in figures:
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=190)
        figure.savefig(output / f"{name}.pdf")
        plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = torch.load(
        args.representation_file, map_location="cpu", weights_only=False
    )
    metadata = _representation_metadata(features)
    prep_args = SimpleNamespace(
        source_feature_file=args.source_feature_file,
        query_feature_file=args.query_feature_file,
        facet_gate_file=args.facet_gate_file,
        projection_dir=args.projection_dir,
        seeds=args.seeds,
    )
    prepared = _prepare_examples(prep_args, torch.device(args.device))
    units = build_surfaces(prepared, args.seeds)
    oracle_path = args.output_dir / "factorized_oracle_rows.csv"
    if oracle_path.exists():
        oracles = load_factorized_oracles(
            oracle_path, {unit.key for unit in units}
        )
    else:
        oracles = build_oracle_artifacts(units, args.output_dir)

    cv_summary, profile_rows = run_profile_gate(
        units, oracles, features, metadata, args.output_dir
    )
    selected = _select_self_representations(cv_summary, metadata)
    best = selected[0]
    feature_ablation = run_feature_ablation(
        units, oracles, features, metadata, best, args.output_dir
    )
    target_rows = run_target_gate(
        units, oracles, features, metadata, best, args.output_dir
    )
    drift_rows = build_drift(features, args.output_dir)
    accounting_rows = build_accounting(
        best, features, metadata, profile_rows, args.output_dir
    )

    profile_summary = _aggregate(
        profile_rows, ("representation", "dataset")
    )
    _write_csv(args.output_dir / "self_router_dataset_breakdown.csv", profile_summary)
    layer_rows = []
    cv_by_name = {row["representation"]: row for row in cv_summary}
    for row in _aggregate(profile_rows, ("representation",)):
        meta = metadata[row["representation"]]
        layer_rows.append(
            {
                **row,
                "validation_cv_quality": cv_by_name[row["representation"]]["quality"],
                "validation_cv_total_cost": cv_by_name[row["representation"]]["total_cost"],
                "family": meta["family"],
                "pooling": meta["pooling"],
                "depth": meta["depth"],
                "contextual_upper_bound": int(meta["contextual"]),
                "selected_by_validation": int(row["representation"] in selected),
            }
        )
    _write_csv(args.output_dir / "self_router_layer_sweep.csv", layer_rows)
    _write_csv(
        args.output_dir / "self_router_pooling.csv",
        [row for row in layer_rows if row["pooling"] in {"mean", "last"}],
    )
    _write_csv(
        args.output_dir / "self_router_contextual_upper_bound.csv",
        [row for row in layer_rows if row["contextual_upper_bound"]],
    )
    _plot(profile_rows, target_rows, drift_rows, accounting_rows, metadata, args.output_dir)

    heldout_all = _aggregate(profile_rows, ("representation",))
    lookup = {row["representation"]: row for row in heldout_all}
    target_summary = _aggregate(target_rows, ("target_architecture",))
    self_improvement = float(lookup[best]["quality"]) - float(
        lookup["S0_observable"]["quality"]
    )
    contextual_cv = [
        row
        for row in cv_summary
        if str(row["representation"]).startswith("S8_context_")
    ]
    selected_contextual = min(
        contextual_cv,
        key=lambda row: (
            -float(row["quality"]),
            float(row["total_cost"]),
            str(row["representation"]),
        ),
    )["representation"]
    findings = {
        "schema_version": "1.0",
        "scope": "frozen_qwen_query_self_encoding_for_factorized_effort_control",
        "examples": len(features),
        "units": len(units),
        "validation_selection": "four-fold grouped CV by example identity",
        "heldout_used_for_selection": False,
        "router_seeds": list(ROUTER_SEEDS),
        "selected_self_representations": selected,
        "validation_selected_self_representation": best,
        "validation_selected_self_heldout_quality": lookup[best]["quality"],
        "validation_selected_self_total_cost": lookup[best]["total_cost"],
        "observable_heldout_quality": lookup["S0_observable"]["quality"],
        "signed_hash_heldout_quality": lookup["S1_signed_hash"]["quality"],
        "self_quality_delta_vs_observable": self_improvement,
        "validation_selected_contextual_representation": selected_contextual,
        "validation_selected_contextual_heldout": lookup[selected_contextual],
        "heldout_descriptive_maxima_are_not_used_for_selection": True,
        "target_architectures": target_summary,
        "feature_ablation": _aggregate(
            feature_ablation, ("feature_mode", "dataset")
        ),
        "external_encoder_gate": "deferred" if self_improvement <= 0 else "eligible_follow_up",
        "reuse": {
            "first_memory_layer": FIRST_MEMORY_LAYER,
            "best_representation_depth": metadata[best]["depth"],
            "eligible": reuse_is_semantically_valid(
                depth=int(metadata[best]["depth"]),
                first_memory_layer=FIRST_MEMORY_LAYER,
                ordinary_context_precedes_query=False,
            ),
            "boundary": "reuse only up to first PRA consumer; corrected query regions require recapture",
        },
        "claim_boundaries": [
            "Backbone and frozen native graph/query scores are unchanged.",
            "The endpoint is evidence-chain sufficiency, not generated-answer quality.",
            "Representation selection uses grouped validation CV and never held-out labels.",
            "Contextual source-plus-query rows are diagnostic upper bounds and pay their 256-token prefill.",
            "Query-prefill reuse is exact only before the first PRA memory consumer.",
            "No compact external encoder was run before the self-encoding gate.",
            "2Wiki and MuSiQue lack the frozen factorized action surface used by this add-on.",
        ],
    }
    (args.output_dir / "self_router_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    result_root = Path("docs/papers/shared/results/paper2_5_iterative_pra")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument(
        "--representation-file",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper3_5_adaptive_pra/self_router_representations.pt",
    )
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=_existing_or_sibling(
            str(result_root / "native_qk_closure/native_qk_features_test.pt")
        ),
    )
    parser.add_argument(
        "--query-feature-file",
        type=Path,
        default=_existing_or_sibling(
            str(result_root / "query_entry_facets/query_entry_features.pt")
        ),
    )
    parser.add_argument(
        "--facet-gate-file",
        type=Path,
        default=ROOT
        / result_root
        / "grounded_query_facets/grounded_facet_gate_results.json",
    )
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra",
    )
    args = parser.parse_args()
    args.seeds = tuple(int(value) for value in args.seeds.split(","))
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
