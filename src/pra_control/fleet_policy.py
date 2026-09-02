"""Presentation-neutral desired/observed fleet policy."""

from __future__ import annotations

from typing import Any, Mapping

from .config import EngineTargetConfig


def match_desired(target: EngineTargetConfig, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = []
    for row in rows:
        if row.get("environment") != target.environment or row.get("cluster") != target.cluster:
            continue
        selector = dict(row.get("engine_instance_selector") or {})
        if selector.get("name") and selector["name"] != target.name:
            continue
        if selector.get("host") and selector["host"] != target.host:
            continue
        if selector.get("namespace") and selector["namespace"] != target.namespace:
            continue
        labels = dict(selector.get("labels") or {})
        if any(target.labels.get(key) != value for key, value in labels.items()):
            continue
        matches.append(row)
    return sorted(matches, key=lambda row: (-int(row.get("desired_revision", 0)), str(row.get("id"))))[0] if matches else None


def compare_desired_observed(desired: Mapping[str, Any] | None, snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        return {"status": "OFFLINE", "differences": []}
    if desired is None:
        return {"status": "UNKNOWN", "differences": []}
    models = snapshot.get("models", {}).get("items", snapshot.get("models", []))
    desired_models = desired.get("desired_models")
    legacy = not isinstance(desired_models, list) or not desired_models
    if legacy:
        desired_models = [{
            "runtime_model_id": "default",
            "model_id": desired.get("desired_model_id"),
            "bundle_id": desired.get("desired_bundle_id"),
            "profile_id": desired.get("desired_profile_id"),
            "mode": desired.get("desired_mode"),
        }]
    observed_by_id = {
        str(row.get("runtime_model_id") or ("default" if len(models) == 1 else row.get("model_id"))): row
        for row in models
    }
    differences: list[dict[str, Any]] = []
    per_model: dict[str, Any] = {}
    expected_ids: set[str] = set()
    for expected_model in desired_models:
        runtime_id = str(expected_model.get("runtime_model_id") or "default")
        expected_ids.add(runtime_id)
        observed = observed_by_id.get(runtime_id, {})
        pairs = {
            "model": (expected_model.get("model_id"), observed.get("model_id")),
            "bundle": (expected_model.get("bundle_id"), observed.get("pra_bundle_id")),
            "profile": (expected_model.get("profile_id"), observed.get("profile")),
            "mode": (expected_model.get("mode"), observed.get("execution_mode")),
        }
        model_differences = [
            {"field": field, "desired": expected, "observed": actual, **({} if legacy else {"runtime_model_id": runtime_id})}
            for field, (expected, actual) in pairs.items()
            if expected is not None and expected != actual
        ]
        if not observed:
            model_differences.insert(0, {
                "field": "MODEL_NOT_LOADED", "desired": runtime_id, "observed": None,
                **({} if legacy else {"runtime_model_id": runtime_id}),
            })
        per_model[runtime_id] = {"status": "DRIFT" if model_differences else "IN_SYNC", "differences": model_differences}
        differences.extend(model_differences)
    if not bool(desired.get("allow_extra_models", True)):
        for runtime_id in observed_by_id.keys() - expected_ids:
            difference = {"field": "UNAPPROVED_MODEL_LOADED", "desired": None, "observed": runtime_id, "runtime_model_id": runtime_id}
            differences.append(difference)
            per_model[runtime_id] = {"status": "DRIFT", "differences": [difference]}
    return {
        "status": "DRIFT" if differences else "IN_SYNC", "differences": differences,
        "models": per_model, "desired_revision": desired.get("desired_revision"),
    }


def light_metrics(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    metrics = snapshot.get("storage", {}).get("metrics", {})
    return {
        "selected_full_token_ratio": metrics.get("selected_full_token_ratio"),
        "visible_reuse": metrics.get("visible_reuse"), "native_reuse": metrics.get("native_reuse"),
        "storage_reloads": metrics.get("reloads"), "request_rate": metrics.get("request_rate"),
        "ttft_p95_ms": metrics.get("ttft_p95_ms"), "error_rate": metrics.get("error_rate"),
    }


def alerts(snapshot: Mapping[str, Any], drift: Mapping[str, Any]) -> list[str]:
    values = ["desired state drift"] if drift["status"] == "DRIFT" else []
    metrics = snapshot.get("storage", {}).get("metrics", {})
    if float(metrics.get("reload_rate", 0) or 0) > 0.25:
        values.append("high storage reload rate")
    return values
