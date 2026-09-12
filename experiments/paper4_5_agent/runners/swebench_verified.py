"""Run the source-matched mini-swe-agent fixed-50 no-PRA baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..benchmark import load_benchmark_card
from ..context_treatment import (
    CONSUMPTION_POLICIES,
    ContextTreatment,
    TreatmentProxy,
    session_id_for_messages,
)


EXPECTED_PACKAGES = {
    "mini-swe-agent": "2.4.0",
    "swebench": "4.1.0",
    "vllm": "0.22.1",
}
PINNED_DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"


def derive_task_card(
    card: dict[str, Any], task_index: int | None, *, source: str | Path,
) -> dict[str, Any]:
    """Derive one ordered task from a locked parent card without resampling."""

    if task_index is None:
        return card
    instance_ids = list(card["instance_ids"])
    if task_index < 1 or task_index > len(instance_ids):
        raise ValueError(
            f"task index must be in 1..{len(instance_ids)}, observed {task_index}"
        )
    instance_id = instance_ids[task_index - 1]
    digest = hashlib.sha256(f"{instance_id}\n".encode("utf-8")).hexdigest()
    return {
        **card,
        "benchmark": f"{card['benchmark']}: task {task_index:02d}",
        "expected_count": 1,
        "parent_cohort": card.get("parent_cohort", card["benchmark"]),
        "parent_cohort_ids_sha256": card["canonical_ids_sha256"],
        "stratum": f"{card.get('stratum', 'cohort')}_task",
        "stratum_definition": (
            f"Predeclared task {task_index} in the locked parent cohort; "
            "no outcome-dependent resampling."
        ),
        "stratum_reference": str(source),
        "canonical_ids_sha256": digest,
        "instance_ids": [instance_id],
        "task_metadata": [
            row for row in card.get("task_metadata", ())
            if row.get("instance_id") == instance_id
        ],
        "task_index": task_index,
    }


def treatment_placement(mode: str) -> dict[str, Any]:
    """Describe transport and PRA ownership independently for one treatment."""

    return {
        "no-pra": {
            "connection": "direct", "engine_pra_enabled": False,
            "gateway_pra_enabled": False, "gateway_mode": None,
        },
        ContextTreatment.TRUNCATION.value: {
            "connection": "direct", "engine_pra_enabled": False,
            "gateway_pra_enabled": False, "gateway_mode": None,
        },
        ContextTreatment.PASSTHROUGH.value: {
            "connection": "gateway", "engine_pra_enabled": False,
            "gateway_pra_enabled": False, "gateway_mode": "G00",
        },
        ContextTreatment.HEADROOM.value: {
            "connection": "gateway", "engine_pra_enabled": False,
            "gateway_pra_enabled": False, "gateway_mode": "HEADROOM",
        },
        ContextTreatment.PRA_SELECTED_CONTEXT.value: {
            "connection": "gateway", "engine_pra_enabled": False,
            "gateway_pra_enabled": True, "gateway_mode": "G10",
        },
        ContextTreatment.DIRECT_NATIVE_PRA.value: {
            "connection": "direct", "engine_pra_enabled": True,
            "gateway_pra_enabled": False, "gateway_mode": None,
        },
        ContextTreatment.GATEWAY_NATIVE_PRA.value: {
            "connection": "gateway", "engine_pra_enabled": True,
            "gateway_pra_enabled": True, "gateway_mode": "G11",
        },
    }[mode]


def gateway_preflight(
    args: argparse.Namespace, *, base_url: str | None = None,
) -> dict[str, Any] | None:
    """Reject an unpinned or incorrectly configured treatment endpoint early."""

    expected_mode = {
        ContextTreatment.PASSTHROUGH.value: "G00",
        ContextTreatment.PRA_SELECTED_CONTEXT.value: "G10",
        ContextTreatment.GATEWAY_NATIVE_PRA.value: "G11",
    }.get(args.mode)
    native_required = args.mode in {
        ContextTreatment.DIRECT_NATIVE_PRA.value,
        ContextTreatment.GATEWAY_NATIVE_PRA.value,
    }
    if (
        expected_mode is None
        and not native_required
        and not bool(getattr(args, "require_endpoint_preflight", False))
    ):
        return None
    root = (base_url or args.base_url).rstrip("/").removesuffix("/v1")

    def read(path: str) -> dict[str, Any]:
        """Read an idempotent qualification endpoint with bounded LAN retries."""

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"{root}{path}", timeout=15) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (
                urllib.error.URLError, TimeoutError, json.JSONDecodeError,
            ) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(attempt + 1)
        raise RuntimeError(
            f"gateway preflight failed for {root}{path} after 3 attempts: {last_error}"
        ) from last_error

    health = read("/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"endpoint preflight health failed: {health!r}")
    if expected_mode is not None and health.get("gateway_mode") != expected_mode:
        raise RuntimeError(
            "gateway preflight mode mismatch: "
            f"expected {expected_mode}, observed {health.get('gateway_mode')!r}"
        )
    effective = health.get("effective_capabilities") or health
    engine = health.get("engine") or {}
    chat_template_profile = health.get(
        "chat_template_profile", effective.get("chat_template_profile")
    )
    chat_template_digest = health.get(
        "chat_template_digest", effective.get("chat_template_digest")
    )
    if native_required and not bool(effective.get("native_kv") or engine.get("native_kv")):
        raise RuntimeError("native PRA treatment requires effective native_kv capability")
    agent_history_qualified = bool(
        effective.get("agent_history_kv_qualified")
        or engine.get("agent_history_kv_qualified")
    )
    full_retention_control = float(
        getattr(args, "budget_fraction", 1.0)
    ) >= 1.0
    if native_required and not agent_history_qualified:
        full_history_capable = all(bool(
            effective.get(name) or engine.get(name)
        ) for name in (
            "session_state", "live_prefix_kv_capture",
            "zero_selected_text_reencoding",
        ))
        if not full_retention_control or not full_history_capable:
            raise RuntimeError(
                "sparse native agent-history PRA requires an engine-specific "
                "agent_history_kv_qualified=true capability; a 100% control may "
                "proceed only with session state, live-prefix capture, and zero "
                "selected-text re-encoding"
            )
    prefix_mode = str(
        engine.get("prefix_cache_mode", effective.get("prefix_cache_mode", "unknown"))
    )
    prefix_supported = bool(
        engine.get("automatic_prefix_cache")
        or engine.get("explicit_prefix_cache")
        or effective.get("automatic_prefix_cache")
        or effective.get("explicit_prefix_cache")
        or prefix_mode in {
            "automatic_prefix_cache", "explicit_prefix_handle", "session_state"
        }
    )
    prefix_active = health.get("prefix_cache_enabled")
    if prefix_active is None:
        prefix_active = engine.get("prefix_cache_enabled", effective.get("prefix_cache_enabled"))
    if bool(getattr(args, "prefix_caching", False)):
        if not prefix_supported:
            raise RuntimeError("prefix-cache treatment requires an advertised cache capability")
        if prefix_active is not True:
            raise RuntimeError(
                "prefix-cache treatment requires prefix_cache_enabled=true, not capability alone"
            )
    elif (
        native_required
        or (
            bool(getattr(args, "require_endpoint_preflight", False))
            and expected_mode is None
        )
    ) and prefix_active is not False:
        raise RuntimeError(
            "cache-off engine treatment requires prefix_cache_enabled=false"
        )
    catalog = read("/v1/models")
    model_ids = tuple(
        str(row.get("id"))
        for row in catalog.get("data", ())
        if isinstance(row, dict) and row.get("id")
    )
    if args.served_model not in model_ids:
        raise RuntimeError(
            "gateway must pin and advertise the frozen backend model: "
            f"expected {args.served_model!r}, observed {list(model_ids)!r}"
        )
    prior_receipt_path = getattr(args, "endpoint_preflight_receipt", None)
    if prior_receipt_path:
        prior_payload = json.loads(
            Path(prior_receipt_path).read_text(encoding="utf-8")
        )
        prior = prior_payload.get("gateway_preflight", prior_payload)
        if not isinstance(prior, dict) or prior.get("generation_probe") != "passed":
            raise RuntimeError("endpoint preflight receipt lacks a passed generation probe")
        if prior.get("advertised_model") != args.served_model:
            raise RuntimeError("endpoint preflight receipt model does not match this run")
        if native_required and prior.get("native_consumption_probe") != "passed":
            raise RuntimeError("endpoint preflight receipt lacks native consumption proof")
        if expected_mode is not None and prior.get("gateway_mode") != expected_mode:
            raise RuntimeError("endpoint preflight receipt gateway mode does not match this run")
        if (
            prior.get("chat_template_digest") is not None
            and prior.get("chat_template_digest") != chat_template_digest
        ):
            raise RuntimeError(
                "endpoint preflight chat template digest changed after restart"
            )
        return {
            **prior,
            "url": root,
            "prefix_cache_mode": prefix_mode,
            "prefix_cache_supported": prefix_supported,
            "prefix_cache_enabled": prefix_active,
            "replayed_from": str(Path(prior_receipt_path).resolve()),
            "post_restart_health_rechecked": True,
            "post_restart_model_rechecked": True,
            "chat_template_profile": chat_template_profile,
            "chat_template_digest": chat_template_digest,
            "generation_probe_replayed": True,
        }
    probe: dict[str, Any] = {
        "model": args.served_model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0,
        "top_p": float(getattr(args, "top_p", 1.0)),
        "seed": int(getattr(args, "sampling_seed", 0)),
        "chat_template_no_thinking": bool(
            getattr(args, "chat_template_no_thinking", False)
        ),
        "prefix_caching": bool(getattr(args, "prefix_caching", False)),
        "max_tokens": 1,
        "stream": False,
    }
    if expected_mode == "G10" or native_required:
        probe["pra"] = {
            "tenant_id": "paper4-5-preflight",
            # Session DELETE creates a deliberate tombstone.  Reusing a fixed
            # readiness-probe identity after a clean retry therefore produces
            # HTTP 410 and can block a valid campaign before its first task.
            "session_id": f"selected-context-consumption-probe-{uuid.uuid4().hex}",
            "resources": [{
                "resource_id": "preflight-resource",
                "uri": "pra://preflight/selected-context",
                "record_type": "preflight_fact",
                "text": "PRA selected-context consumption probe.",
                "version": "v1",
                "source_fingerprint": hashlib.sha256(
                    b"PRA selected-context consumption probe."
                ).hexdigest(),
                "authorization_scope": "paper4-5-preflight",
                "metadata": {"purpose": "live_consumption_probe"},
            }],
            "budget": {"max_resources": 1, "max_selected_tokens": 8},
            "allow_text_fallback": not native_required,
            "required_capabilities": ["logical_refs", "native_kv"] if native_required else [],
            "pra_policy": {"profile": "swebench-balanced-v1"},
            "metadata": {
                "requested_mode": "native-memory" if native_required else "selected-context",
                # A readiness probe must not alter the K/V occupancy seen by
                # the subsequent experiment.  The engine wrapper releases
                # this session after constructing the response.
                "ephemeral_session": True,
            },
        }
    probe_payload = json.dumps(probe).encode("utf-8")
    probe_request = urllib.request.Request(
        f"{root}/v1/chat/completions",
        data=probe_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(probe_request, timeout=120) as response:
            completion = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"gateway generation probe failed for {root}/v1/chat/completions: {error}"
        ) from error
    choices = completion.get("choices") if isinstance(completion, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "gateway generation probe returned no OpenAI-compatible choices"
        )
    selected_context_probe = None
    if expected_mode == "G10":
        selected_ids = (completion.get("pra") or {}).get("selected_resource_ids")
        if selected_ids != ["preflight-resource"]:
            raise RuntimeError(
                "G10 consumption probe did not acknowledge the selected resource: "
                f"observed {selected_ids!r}"
            )
        selected_context_probe = "passed"
    native_consumption_probe = None
    if native_required:
        pra = completion.get("pra") or {}
        native_trace = completion.get("pra_trace") or ()
        attached = any(
            row.get("stage") in {"llama_cpp_native_attach", "native_attach"}
            for row in native_trace if isinstance(row, dict)
        )
        if pra.get("native_kv") is not True or not attached:
            raise RuntimeError(
                "native PRA consumption probe did not prove physical native attachment"
            )
        native_consumption_probe = "passed"
    return {
        "url": root,
        "gateway_mode": expected_mode,
        "advertised_model": args.served_model,
        "protocol_version": health.get("protocol_version"),
        "generation_probe": "passed",
        "selected_context_probe": selected_context_probe,
        "native_consumption_probe": native_consumption_probe,
        "prefix_cache_mode": prefix_mode,
        "prefix_cache_supported": prefix_supported,
        "prefix_cache_enabled": prefix_active,
        "chat_template_profile": chat_template_profile,
        "chat_template_digest": chat_template_digest,
    }


def package_versions() -> dict[str, str | None]:
    """Return installed versions without converting missing packages to zero-like values."""

    versions: dict[str, str | None] = {}
    for package in EXPECTED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def package_record_hashes() -> dict[str, str | None]:
    """Fingerprint installed distribution manifests when the source gives only versions."""

    values: dict[str, str | None] = {}
    for package in EXPECTED_PACKAGES:
        try:
            record = importlib.metadata.distribution(package).read_text("RECORD")
        except importlib.metadata.PackageNotFoundError:
            record = None
        values[package] = hashlib.sha256(record.encode("utf-8")).hexdigest() if record else None
    return values


def preflight(args: argparse.Namespace, card: dict[str, Any]) -> dict[str, Any]:
    """Capture identity and host compatibility before expensive inference starts."""

    versions = package_versions()
    expected_packages = {
        "mini-swe-agent": args.harness_version,
        "swebench": args.grader_version,
    }
    if not args.local_calibration and args.engine == "vllm":
        expected_packages["vllm"] = args.engine_version
    differences = [
        f"{name}={versions[name]!r}; execution requires {expected!r}"
        for name, expected in expected_packages.items()
        if versions[name] != expected
    ]
    if args.model_revision == "NOT_REPORTED_BY_SOURCE":
        differences.append("execution model revision is not pinned")
    if args.tokenizer_revision == "NOT_REPORTED_BY_SOURCE":
        differences.append("execution tokenizer revision is not pinned")
    gpu = _nvidia_gpu()
    if not args.local_calibration and not _is_h100_80gb(gpu):
        differences.append(f"hardware={gpu or 'no NVIDIA GPU detected'}; source used one H100 80GB")
    current_dataset_revision = _dataset_revision()
    if current_dataset_revision != args.benchmark_revision:
        differences.append(
            f"SWE-bench dataset revision={current_dataset_revision!r}; "
            f"execution requires {args.benchmark_revision!r}"
        )
    selection_record = getattr(args, "selection_record", None)
    selection_replay = getattr(args, "selection_replay", None)
    selection_contract = (
        "frozen_replay" if selection_replay
        else "route_owned" if args.mode in {
            ContextTreatment.PRA_SELECTED_CONTEXT.value,
            ContextTreatment.DIRECT_NATIVE_PRA.value,
            ContextTreatment.GATEWAY_NATIVE_PRA.value,
        } else "not_applicable"
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_source_revision": card["source_revision"],
        "benchmark_execution_revision": args.benchmark_revision,
        "benchmark_ids_sha256": card["canonical_ids_sha256"],
        "benchmark_parent_ids_sha256": card.get("parent_cohort_ids_sha256"),
        "benchmark_stratum": card.get("stratum"),
        "benchmark_stratum_definition": card.get("stratum_definition"),
        "benchmark_stratum_reference": card.get("stratum_reference"),
        "instance_count": len(card["instance_ids"]),
        "model": args.model,
        "served_model": args.served_model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "engine_versions": versions,
        "package_record_sha256": package_record_hashes(),
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "context_limit": args.context_limit,
        "max_steps": args.max_steps,
        "max_completion_tokens": getattr(args, "max_completion_tokens", None),
        "docker_platform": getattr(args, "docker_platform", None),
        "prepull_images": bool(getattr(args, "prepull_images", False)),
        "temperature": 0,
        "top_p": float(getattr(args, "top_p", 1.0)),
        "sampling_seed": int(getattr(args, "sampling_seed", 0)),
        "campaign_mode": args.mode,
        "selection_contract": selection_contract,
        "selection_record_path": str(selection_record) if selection_record else None,
        "selection_replay_path": str(selection_replay) if selection_replay else None,
        "selection_replay_sha256": (
            hashlib.sha256(selection_replay.read_bytes()).hexdigest()
            if selection_replay else None
        ),
        **treatment_placement(args.mode),
        "context_budget_fraction": args.budget_fraction,
        "retention_policy": {
            "recent_completed_turns": getattr(args, "recent_completed_turns", 2),
            "recent_records_per_turn": getattr(args, "recent_records_per_turn", 2),
            "recent_source_turns": getattr(args, "recent_source_turns", 1),
            "recent_progress_turns": getattr(args, "recent_progress_turns", 1),
            "recent_mutation_turns": getattr(args, "recent_mutation_turns", 1),
            "recent_verification_turns": getattr(args, "recent_verification_turns", 1),
            "large_record_chunk_tokens": getattr(
                args, "large_record_chunk_tokens", 256,
            ),
            "max_records_per_turn_before_chunking": getattr(
                args, "max_records_per_turn_before_chunking", 8,
            ),
            "preserve_action_observation_pairs": getattr(
                args, "preserve_action_observation_pairs", True,
            ),
            "causal_bundle_round_up": getattr(args, "causal_bundle_round_up", True),
        },
        "harness": "mini-swe-agent",
        "harness_version": args.harness_version,
        "harness_config": args.scaffold,
        "consumption_policy": getattr(args, "consumption_policy", "standard"),
        "model_class": "litellm_textbased",
        "official_grader": args.grading,
        "grader_version": args.grader_version,
        "python": sys.version,
        "os": platform.platform(),
        "gpu": gpu,
        "configuration_differences": differences,
        "source_provenance_limitations": [] if args.local_calibration else [
            "source study did not publish an immutable model revision",
            "source study did not publish an immutable tokenizer revision",
            "source study did not publish package or grader-image hashes",
            "source study did not publish the SWE-bench dataset revision",
        ],
        "exact_environment": not differences,
    }


def run(args: argparse.Namespace) -> Path:
    """Execute all fixed IDs in resumable chunks and normalize official grading."""

    consumption_policy = getattr(args, "consumption_policy", "standard")
    if consumption_policy != "standard" and args.mode not in {
        ContextTreatment.PRA_SELECTED_CONTEXT.value,
        ContextTreatment.DIRECT_NATIVE_PRA.value,
        ContextTreatment.GATEWAY_NATIVE_PRA.value,
    }:
        raise ValueError("nonstandard consumption policies require a PRA treatment")
    selection_record = getattr(args, "selection_record", None)
    selection_replay = getattr(args, "selection_replay", None)
    if selection_record and selection_replay:
        raise ValueError("--selection-record and --selection-replay are mutually exclusive")
    if (selection_record or selection_replay) and args.mode not in {
        ContextTreatment.DIRECT_NATIVE_PRA.value,
        ContextTreatment.GATEWAY_NATIVE_PRA.value,
    }:
        raise ValueError("selection fixtures are restricted to native-PRA treatments")
    card = load_benchmark_card(args.benchmark_card)
    card = derive_task_card(
        card, getattr(args, "task_index", None), source=args.benchmark_card,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = preflight(args, card)
    receipt["interaction_history_path"] = str(output / "interaction_history.jsonl")
    receipt["interaction_history_contract"] = "logical-physical-response-v1"
    # Validate the actual configured gateway or native engine before inserting
    # the local treatment/telemetry proxy. Preflighting the temporary proxy
    # would only prove that the wrapper started and could mask a G00/G10 route
    # accidentally pointed at an ordinary OpenAI-compatible model endpoint.
    receipt["gateway_preflight"] = gateway_preflight(args)
    proxy = None
    agent_base_url = args.base_url.rstrip("/")
    if args.mode in {
        ContextTreatment.PASSTHROUGH.value,
        ContextTreatment.HEADROOM.value,
        ContextTreatment.TRUNCATION.value,
        ContextTreatment.PRA_SELECTED_CONTEXT.value,
        ContextTreatment.DIRECT_NATIVE_PRA.value,
        ContextTreatment.GATEWAY_NATIVE_PRA.value,
    }:
        proxy = TreatmentProxy(
            agent_base_url,
            mode=ContextTreatment(args.mode),
            budget_fraction=args.budget_fraction,
            trace_path=output / "request_telemetry.jsonl",
            interaction_trace_path=output / "interaction_history.jsonl",
            selection_record_path=getattr(args, "selection_record", None),
            selection_replay_path=getattr(args, "selection_replay", None),
            recent_completed_turns=getattr(args, "recent_completed_turns", 2),
            recent_records_per_turn=getattr(args, "recent_records_per_turn", 2),
            recent_source_turns=getattr(args, "recent_source_turns", 1),
            recent_progress_turns=getattr(args, "recent_progress_turns", 1),
            recent_mutation_turns=getattr(args, "recent_mutation_turns", 1),
            recent_verification_turns=getattr(args, "recent_verification_turns", 1),
            large_record_chunk_tokens=getattr(
                args, "large_record_chunk_tokens", 256,
            ),
            max_records_per_turn_before_chunking=getattr(
                args, "max_records_per_turn_before_chunking", 8,
            ),
            preserve_action_observation_pairs=getattr(
                args, "preserve_action_observation_pairs", True,
            ),
            causal_bundle_round_up=getattr(args, "causal_bundle_round_up", True),
            consumption_policy=getattr(args, "consumption_policy", "standard"),
            request_overrides={
                "prefix_caching": bool(getattr(args, "prefix_caching", False))
            },
        )
        agent_base_url = proxy.start()
    try:
        receipt["execution_fingerprint"] = _execution_fingerprint(receipt)
        _write_json(output / "run_manifest.json", receipt)
        if args.preflight_only:
            return output / "run_manifest.json"
        if receipt["configuration_differences"] and not args.allow_partial_reproduction:
            raise RuntimeError(
                "source-matched preflight failed; use --allow-partial-reproduction only for a "
                "diagnostic that must remain locked from PRA"
            )
        return _execute_chunks(args, card, output, agent_base_url, receipt)
    finally:
        if proxy is not None:
            proxy.close()


def _execute_chunks(
    args: argparse.Namespace,
    card: dict[str, Any],
    output: Path,
    agent_base_url: str,
    receipt: dict[str, Any],
) -> Path:
    """Run resumable agent/grader chunks against the direct endpoint or treatment proxy."""

    submitted: set[str] = set()
    resolved: set[str] = set()
    errors: set[str] = set()
    timeouts: set[str] = set()
    instance_ids = card["instance_ids"]
    for chunk_index in range(0, len(instance_ids), args.chunk_size):
        chunk_ids = instance_ids[chunk_index:chunk_index + args.chunk_size]
        chunk_number = chunk_index // args.chunk_size
        chunk_dir = output / f"chunk_{chunk_number:02d}"
        report_receipt = chunk_dir / "official_chunk_result.json"
        chunk_result = None
        if report_receipt.is_file():
            candidate = json.loads(report_receipt.read_text(encoding="utf-8"))
            # Legacy direct baselines do not cross mutable treatment code. A
            # treatment receipt, however, is reusable only under the exact
            # endpoint, policy, and implementation that produced it.
            if _chunk_receipt_reusable(args, candidate, receipt):
                chunk_result = candidate
        if chunk_result is None:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            if getattr(args, "prepull_images", False):
                _prepull_swebench_images(args, chunk_ids, output, chunk_number)
            pattern = "(" + "|".join(re.escape(item) for item in chunk_ids) + ")"
            predictions = chunk_dir / "preds.json"
            agent_command = [
                sys.executable, "-m", "minisweagent.run.benchmarks.swebench",
                "--subset", "verified", "--split", "test", "--filter", pattern,
                "-m", f"openai/{args.served_model}", "--model-class", "litellm_textbased",
                "-c", args.scaffold,
                "-c", f"model.model_kwargs.api_base={agent_base_url}",
                "-c", "model.model_kwargs.temperature=0",
                "-c", f"model.model_kwargs.top_p={float(getattr(args, 'top_p', 1.0))}",
                "-c", f"model.model_kwargs.seed={int(getattr(args, 'sampling_seed', 0))}",
            ]
            agent_command.extend(_completion_token_overrides(args))
            agent_command.extend([
                "-c", f"agent.step_limit={args.max_steps}", "-w", str(args.workers),
                "-o", str(chunk_dir),
            ])
            if getattr(args, "chat_template_no_thinking", False):
                agent_command.extend([
                    "-c",
                    'model.model_kwargs.extra_body={"chat_template_kwargs":{"enable_thinking":false}}',
                ])
            dataset_environment = _container_environment(args, output)
            timed_out = False
            if getattr(args, "recover_timeout_chunk", None) == chunk_number:
                _write_empty_predictions(predictions, chunk_ids, args.served_model)
                timed_out = True
            else:
                try:
                    _run(
                        agent_command, output / f"chunk_{chunk_number:02d}.agent.log",
                        args.timeout_seconds, extra_environment=dataset_environment,
                    )
                except subprocess.TimeoutExpired:
                    _write_empty_predictions(predictions, chunk_ids, args.served_model)
                    timed_out = True
            _raise_on_agent_infrastructure_error(chunk_dir, chunk_ids)
            if not predictions.is_file():
                raise RuntimeError(f"mini-swe-agent did not produce {predictions}")
            if timed_out:
                _write_json(chunk_dir / "agent_timeout_receipt.json", {
                    "instance_ids": chunk_ids,
                    "timeout_seconds": args.timeout_seconds,
                    "normalization": "empty patch submitted to official grader",
                })
            grade_command = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "-d", card["dataset"], "-s", card["split"],
                "-p", str(predictions), "--instance_ids", *chunk_ids,
                "--run_id", f"{args.run_id}_c{chunk_number}",
                "--max_workers", str(args.grader_workers), "--cache_level", "base",
                "--clean", "True", "--report_dir", str(chunk_dir),
            ]
            grader_wall_time_s = _run(
                grade_command, output / f"chunk_{chunk_number:02d}.grader.log",
                args.timeout_seconds, extra_environment=dataset_environment,
                cwd=chunk_dir,
            )
            raw_report = _find_report(chunk_dir, f"{args.run_id}_c{chunk_number}")
            chunk_result = _normalize_report(raw_report, chunk_ids)
            chunk_result["grader_wall_time_s"] = grader_wall_time_s
            chunk_result["agent_timeout_ids"] = chunk_ids if timed_out else []
            if receipt.get("execution_fingerprint"):
                chunk_result["execution_fingerprint"] = receipt["execution_fingerprint"]
            _write_json(report_receipt, chunk_result)
        submitted.update(chunk_result["submitted_ids"])
        resolved.update(chunk_result["resolved_ids"])
        errors.update(chunk_result["error_ids"])
        timeouts.update(chunk_result.get("agent_timeout_ids") or ())

    expected = set(instance_ids)
    if submitted != expected:
        missing = sorted(expected - submitted)
        extra = sorted(submitted - expected)
        raise RuntimeError(f"official cohort mismatch; missing={missing}, extra={extra}")
    ordered_resolved = [item for item in instance_ids if item in resolved]
    result = {
        "official_grader": True,
        "score": len(ordered_resolved) / len(instance_ids),
        "resolved": len(ordered_resolved),
        "total": len(instance_ids),
        "timeouts": len(timeouts),
        "task_ids": instance_ids,
        "configuration_differences": receipt["configuration_differences"],
        "grader_artifact": str(output / "official_aggregate.json"),
        "execution_identity": {
            "cohort_sha256": card["canonical_ids_sha256"],
            "parent_cohort_sha256": card.get("parent_cohort_ids_sha256"),
            "stratum": card.get("stratum"),
            "stratum_reference": card.get("stratum_reference"),
            "benchmark_revision": args.benchmark_revision,
            "harness": "mini-swe-agent",
            "harness_version": args.harness_version,
            "model": args.model,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
            "engine": args.engine,
            "engine_version": args.engine_version,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "scaffold": args.scaffold,
            "consumption_policy": getattr(args, "consumption_policy", "standard"),
            "context_limit": args.context_limit,
            "max_steps": args.max_steps,
            "max_completion_tokens": getattr(args, "max_completion_tokens", None),
            "temperature": 0.0,
            "top_p": float(getattr(args, "top_p", 1.0)),
            "sampling_seed": int(getattr(args, "sampling_seed", 0)),
            "chat_template_no_thinking": bool(
                getattr(args, "chat_template_no_thinking", False)
            ),
            "function_calling": False,
            "prefix_caching": bool(getattr(args, "prefix_caching", False)),
            "container_platform": getattr(args, "docker_platform", None),
            "images_prepulled": bool(getattr(args, "prepull_images", False)),
            "grading": args.grading,
        },
    }
    _write_json(output / "official_aggregate.json", {
        "submitted_ids": instance_ids,
        "resolved_ids": ordered_resolved,
        "error_ids": [item for item in instance_ids if item in errors],
        "timeout_ids": [item for item in instance_ids if item in timeouts],
    })
    _write_json(output / "official_result.json", result)
    _write_task_rows(
        output / "results.jsonl", output, args, instance_ids, resolved, errors, timeouts,
    )
    return output / "official_result.json"


def _completion_token_overrides(args: argparse.Namespace) -> list[str]:
    """Bound pathological backend continuations without changing uncapped runners."""

    max_completion_tokens = getattr(args, "max_completion_tokens", None)
    if max_completion_tokens is None:
        return []
    if max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")
    return ["-c", f"model.model_kwargs.max_tokens={max_completion_tokens}"]


def _execution_fingerprint(receipt: dict[str, Any]) -> str:
    """Bind resumable treatment chunks to endpoint, policy, and implementation."""

    source_files = (Path(__file__), Path(__file__).parents[1] / "context_treatment.py")
    material = {
        "identity": {
            key: receipt.get(key)
            for key in (
                "benchmark_ids_sha256", "benchmark_execution_revision", "model",
                "model_revision", "engine", "engine_version", "campaign_mode",
                "context_budget_fraction", "context_limit", "max_steps",
                "retention_policy",
                "max_completion_tokens",
                "top_p",
                "sampling_seed",
                "consumption_policy",
                "chat_template_no_thinking",
                "prefix_caching",
                "docker_platform",
                "prepull_images",
            )
        },
        "gateway_preflight": receipt.get("gateway_preflight"),
        "selection_contract": receipt.get("selection_contract"),
        "selection_replay_sha256": receipt.get("selection_replay_sha256"),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chunk_receipt_reusable(
    args: argparse.Namespace, candidate: dict[str, Any], receipt: dict[str, Any],
) -> bool:
    """Allow legacy direct baselines, but require exact treatment provenance."""

    if getattr(args, "mode", "no-pra") == "no-pra":
        return True
    expected = receipt.get("execution_fingerprint")
    return bool(expected and candidate.get("execution_fingerprint") == expected)


def _nvidia_gpu() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _prepull_swebench_images(
    args: argparse.Namespace,
    instance_ids: list[str],
    output: Path,
    chunk_number: int,
) -> None:
    """Acquire official task images before mini-swe-agent's fixed startup timeout."""

    docker_config = output / ".docker-anonymous"
    docker_config.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        compatible_id = instance_id.replace("__", "_1776_").lower()
        image = f"docker.io/swebench/sweb.eval.x86_64.{compatible_id}:latest"
        command = ["docker", "pull"]
        if args.docker_platform:
            command.extend(["--platform", args.docker_platform])
        command.append(image)
        log = output / f"chunk_{chunk_number:02d}.{compatible_id}.image.log"
        wall_time_s = _run(
            command,
            log,
            args.image_pull_timeout_seconds,
            extra_environment={"DOCKER_CONFIG": str(docker_config)},
        )
        receipts.append({
            "instance_id": instance_id,
            "image": image,
            "platform": args.docker_platform,
            "wall_time_s": wall_time_s,
        })
    _write_json(output / f"chunk_{chunk_number:02d}.image_receipt.json", {
        "schema_version": 1,
        "images": receipts,
        "registry_auth": "anonymous_public_pull",
        "excluded_from_model_and_grader_wall_time": True,
    })


def _container_environment(
    args: argparse.Namespace, output: Path,
) -> dict[str, str]:
    """Pin dataset cache and evaluator platform across agent and grader."""

    environment = {"HF_DATASETS_CACHE": str(output / "hf_datasets_cache")}
    # The official evaluator images are x86-only.  The pre-pull happens before
    # the agent, whose cleanup may remove that image; the grader must retain
    # the same platform contract for its own fallback pull on Apple Silicon.
    if getattr(args, "docker_platform", None):
        environment["DOCKER_DEFAULT_PLATFORM"] = str(args.docker_platform)
    return environment


def _dataset_revision() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info("princeton-nlp/SWE-bench_Verified").sha
    except Exception:  # The receipt records an unavailable identity as a blocking difference.
        return None


def _is_h100_80gb(gpu: str | None) -> bool:
    """Accept nvidia-smi's MiB form while rejecting smaller H100 variants."""

    if not gpu or "H100" not in gpu.upper():
        return False
    memory_values = [int(value) for value in re.findall(r"(\d+)\s*MiB", gpu, re.IGNORECASE)]
    return bool(memory_values) and max(memory_values) >= 79_000


def _run(
    command: list[str], log: Path, timeout_seconds: int,
    *, extra_environment: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> float:
    environment = os.environ.copy()
    environment.setdefault("OPENAI_API_KEY", "dummy")
    environment.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    path_entries = environment.get("PATH", "").split(os.pathsep)
    for candidate in (
        "/Applications/Docker.app/Contents/Resources/bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
    ):
        if Path(candidate).is_dir() and candidate not in path_entries:
            path_entries.insert(0, candidate)
    environment["PATH"] = os.pathsep.join(path_entries)
    environment.update(extra_environment or {})
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, env=environment,
            cwd=cwd, timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        cleaned = _cleanup_owned_containers(stdout + "\n" + stderr)
        log.write_text(
            stdout + "\n--- STDERR ---\n" + stderr + "\n--- TIMEOUT ---\n"
            + "cleaned_containers=" + json.dumps(cleaned) + "\n",
            encoding="utf-8",
        )
        raise
    elapsed = time.perf_counter() - started
    log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}; see {log}")
    return elapsed


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _cleanup_owned_containers(process_output: str) -> list[str]:
    """Remove only task containers named in the timed-out subprocess output."""

    names = sorted(set(re.findall(
        r"\b(?:minisweagent-[a-z0-9]+|sweb\.eval\.[A-Za-z0-9_.-]+)\b",
        process_output,
    )))
    cleaned = []
    for name in names:
        try:
            completed = subprocess.run(
                ["docker", "rm", "-f", name], capture_output=True, text=True,
                timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            cleaned.append(name)
    return cleaned


def _raise_on_agent_infrastructure_error(
    chunk_dir: Path, instance_ids: Sequence[str],
) -> None:
    """Reject mini-swe-agent's zero-exit empty-patch normalization of run errors.

    The benchmark command can return success after logging an exception for an
    instance and emitting an empty patch.  Such rows are infrastructure/model
    execution failures, not legitimate unsuccessful solutions, and must not be
    passed to the official grader as scientific observations.
    """

    log = chunk_dir / "minisweagent.log"
    if not log.is_file():
        return
    text = log.read_text(encoding="utf-8", errors="replace")
    failed = [
        instance_id for instance_id in instance_ids
        if f"Error processing instance {instance_id}:" in text
    ]
    if not failed:
        return
    receipt = {
        "schema_version": 1,
        "classification": "agent_execution_failure",
        "instance_ids": failed,
        "source_log": str(log),
        "admitted_as_benchmark_result": False,
    }
    _write_json(chunk_dir / "infrastructure_failure.json", receipt)
    raise RuntimeError(
        "mini-swe-agent reported execution failure for "
        f"{failed}; refusing to grade an infrastructure-generated empty patch"
    )


def _write_empty_predictions(
    destination: Path, instance_ids: list[str], served_model: str,
) -> None:
    """Represent an agent timeout as an empty patch for canonical grading."""

    payload = {
        instance_id: {
            "model_name_or_path": f"openai/{served_model}",
            "instance_id": instance_id,
            "model_patch": "",
        }
        for instance_id in instance_ids
    }
    _write_json(destination, payload)


def _find_report(directory: Path, run_id: str) -> Path:
    matches = sorted(directory.glob(f"*.{run_id}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one official report for {run_id}, found {len(matches)}")
    return matches[0]


def _normalize_report(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    submitted = set(payload.get("submitted_ids") or ())
    if submitted != set(expected_ids):
        raise RuntimeError(f"grader report {path} does not match its frozen chunk")
    return {
        "submitted_ids": expected_ids,
        "resolved_ids": [item for item in expected_ids if item in set(payload.get("resolved_ids") or ())],
        "error_ids": [item for item in expected_ids if item in set(payload.get("error_ids") or ())],
    }


def _write_task_rows(
    destination: Path, output: Path, args: argparse.Namespace, instance_ids: list[str],
    resolved: set[str], errors: set[str], timeouts: set[str],
) -> None:
    """Join agent-visible trajectories and proxy traces with official outcomes."""

    traces_by_session = _load_treatment_traces(output / "request_telemetry.jsonl")
    unavailable = {
        "wall_time_s": None, "ttft_s": None, "prefill_time_s": None,
        "decode_time_s": None, "tool_time_s": None, "pra_route_time_s": None,
        "pra_materialize_time_s": None, "gateway_overhead_s": None,
        "peak_memory_bytes": None, "kv_bytes": None, "pra_memory_bytes": None,
    }
    rows = []
    placement = treatment_placement(args.mode)
    for task_index, instance_id in enumerate(instance_ids):
        trajectory_path = _find_trajectory(output, instance_id)
        trajectory = _trajectory_metrics(trajectory_path) if trajectory_path else {}
        chunk_number = task_index // args.chunk_size
        chunk_receipt = _read_json(output / f"chunk_{chunk_number:02d}" / "official_chunk_result.json")
        grader_error_type = _grader_error_type(
            output / f"chunk_{chunk_number:02d}.grader.log", instance_id
        ) if instance_id in errors else None
        task_traces = traces_by_session.get(trajectory.get("session_id"), [])
        trace = _aggregate_traces(task_traces)
        prompt_tokens = trajectory.get("cumulative_prompt_tokens")
        logical_tokens = prompt_tokens if args.mode == "no-pra" else None
        physical_tokens = prompt_tokens
        patch = str(trajectory.get("patch") or "")
        rows.append({
            "run_id": args.run_id, "benchmark": "SWE-bench Verified",
            "instance_id": instance_id, "harness": "mini-swe-agent",
            "harness_version": args.harness_version, "model": args.model,
            "model_revision": args.model_revision, "engine": args.engine,
            "engine_version": args.engine_version, "quantization": args.quantization,
            "dtype": args.dtype, "mode": args.mode,
            **placement,
            "selection_contract": (
                "frozen_replay" if getattr(args, "selection_replay", None)
                else "route_owned" if args.mode in {
                    ContextTreatment.PRA_SELECTED_CONTEXT.value,
                    ContextTreatment.DIRECT_NATIVE_PRA.value,
                    ContextTreatment.GATEWAY_NATIVE_PRA.value,
                } else "not_applicable"
            ),
            "context_budget_fraction": args.budget_fraction,
            "pra_config_id": (
                "swebench-balanced-v1"
                if args.mode in {
                    ContextTreatment.PRA_SELECTED_CONTEXT.value,
                    ContextTreatment.DIRECT_NATIVE_PRA.value,
                    ContextTreatment.GATEWAY_NATIVE_PRA.value,
                } else None
            ),
            "context_budget": args.context_limit,
            "seed": int(getattr(args, "sampling_seed", 0)),
            "top_p": float(getattr(args, "top_p", 1.0)),
            "resolved": instance_id in resolved,
            "benchmark_score": 1.0 if instance_id in resolved else 0.0,
            "logical_input_tokens": logical_tokens,
            "physical_input_tokens": physical_tokens,
            "cumulative_prompt_tokens": prompt_tokens,
            "unique_context_tokens_estimate": trajectory.get("max_prompt_tokens"),
            "repeated_context_tokens_estimate": trajectory.get("repeated_context_tokens_estimate"),
            "repeated_context_fraction_estimate": trajectory.get("repeated_context_fraction_estimate"),
            "context_estimate_semantics": "max_prompt_under_accumulating_minisweagent_trajectory",
            "max_prompt_tokens": trajectory.get("max_prompt_tokens"),
            "logical_input_tokens_estimate": trace.get("logical_input_tokens_estimate"),
            "physical_input_tokens_estimate": trace.get("physical_input_tokens_estimate"),
            "selected_tokens_estimate": trace.get("selected_tokens_estimate"),
            "tokens_avoided_estimate": trace.get("tokens_avoided_estimate"),
            "token_saving_fraction_estimate": trace.get("token_saving_fraction_estimate"),
            "token_estimator": trace.get("token_estimator"),
            "selected_resource_digests": trace.get("selected_resource_digests"),
            "prefix_caching": bool(getattr(args, "prefix_caching", False)),
            "prefix_cached_tokens": trace.get("prefix_cached_tokens"),
            "engine_cached_tokens_total": trace.get("engine_cached_tokens_total"),
            "prefix_cache_observed_requests": trace.get("prefix_cache_observed_requests"),
            "native_tokens": trace.get("native_tokens"),
            "wire_tokens": trace.get("wire_tokens"),
            "physical_kv_copy_observed": trace.get("physical_kv_copy_observed"),
            "physical_kv_copy_bytes": trace.get("physical_kv_copy_bytes"),
            "total_kv_copy_bytes": trace.get("total_kv_copy_bytes"),
            "canonical_suffix_graft_d2d_bytes": trace.get(
                "canonical_suffix_graft_d2d_bytes"
            ),
            "host_to_device_bytes": trace.get("host_to_device_bytes"),
            "selected_kv_tokens": trace.get("selected_kv_tokens"),
            "selected_text_reencoded_tokens": trace.get(
                "selected_text_reencoded_tokens"
            ),
            "selected_history_reencoded_tokens": trace.get(
                "selected_history_reencoded_tokens"
            ),
            "realized_retention_fraction": trace.get(
                "realized_retention_fraction"
            ),
            "realized_retention_fraction_min": trace.get(
                "realized_retention_fraction_min"
            ),
            "realized_retention_fraction_max": trace.get(
                "realized_retention_fraction_max"
            ),
            "engine_reported_history_kv_retention_fraction_min": trace.get(
                "engine_reported_history_kv_retention_fraction_min"
            ),
            "engine_reported_history_kv_retention_fraction_max": trace.get(
                "engine_reported_history_kv_retention_fraction_max"
            ),
            "consumer_temporary_bytes": trace.get("consumer_temporary_bytes"),
            "consumer_temporary_peak_bytes": trace.get(
                "consumer_temporary_peak_bytes"
            ),
            "fused_attention_calls": trace.get("fused_attention_calls"),
            "full_retention_requests": trace.get("full_retention_requests"),
            "sparse_kv_requests": trace.get("sparse_kv_requests"),
            "resource_update_counts": trace.get("resource_update_counts"),
            "resource_prefix_cached_tokens": trace.get("resource_prefix_cached_tokens"),
            "resource_evaluated_tokens": trace.get("resource_evaluated_tokens"),
            "resource_total_tokens": trace.get("resource_total_tokens"),
            "output_tokens": trajectory.get("output_tokens"),
            "materialized_tokens": None,
            "selected_tokens": None,
            "token_saving_fraction": 0.0 if args.mode == "no-pra" and prompt_tokens is not None else None,
            "request_count": trajectory.get("model_call_count"),
            "model_call_count": trajectory.get("model_call_count"),
            "step_count": trajectory.get("model_call_count"),
            "trajectory_length": trajectory.get("model_call_count"),
            "tool_call_count": trajectory.get("tool_call_count"),
            "commands_executed": trajectory.get("commands_executed"),
            "files_inspected": trajectory.get("files_inspected"),
            "unique_files_inspected": trajectory.get("unique_files_inspected"),
            "files_modified": trajectory.get("files_modified"),
            "modified_file_paths": trajectory.get("modified_file_paths"),
            **unavailable,
            "wall_time_s": trajectory.get("wall_time_s"),
            "tool_time_s": trajectory.get("tool_time_s"),
            "grader_time_s": chunk_receipt.get("grader_wall_time_s"),
            "pra_route_time_s": trace.get("route_time_s"),
            "termination_reason": trajectory.get("termination_reason"),
            "timed_out": instance_id in timeouts,
            "error_type": grader_error_type,
            "grader_outcome": (
                "error" if instance_id in errors
                else "resolved" if instance_id in resolved
                else "unresolved"
            ),
            "empty_patch": not bool(patch.strip()) if trajectory_path else None,
            "invalid_patch": grader_error_type == "patch_apply_failed",
            "patch_bytes": len(patch.encode("utf-8")) if trajectory_path else None,
            "patch_lines": len(patch.splitlines()) if trajectory_path else None,
            "trajectory_path": trajectory_path.as_posix() if trajectory_path else None,
            "patch_path": trajectory_path.as_posix() if trajectory_path else None,
        })
    destination.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _find_trajectory(output: Path, instance_id: str) -> Path | None:
    matches = sorted(output.glob(f"chunk_*/*/{instance_id}.traj.json"))
    if not matches:
        matches = sorted(output.rglob(f"{instance_id}.traj.json"))
    return matches[0] if matches else None


def _trajectory_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("messages") or []
    calls = [
        row for row in messages
        if row.get("role") == "assistant" and (row.get("extra") or {}).get("response")
    ]
    prompt_per_call = [
        int((((row.get("extra") or {}).get("response") or {}).get("usage") or {}).get("prompt_tokens") or 0)
        for row in calls
    ]
    output_tokens = sum(
        int((((row.get("extra") or {}).get("response") or {}).get("usage") or {}).get("completion_tokens") or 0)
        for row in calls
    )
    cumulative = sum(prompt_per_call)
    unique = max(prompt_per_call, default=0)
    repeated = max(0, cumulative - unique)
    timestamps = [
        float((row.get("extra") or {}).get("timestamp"))
        for row in messages if (row.get("extra") or {}).get("timestamp") is not None
    ]
    tool_time = 0.0
    previous_assistant_time: float | None = None
    for row in messages:
        stamp = (row.get("extra") or {}).get("timestamp")
        if row.get("role") == "assistant" and stamp is not None:
            previous_assistant_time = float(stamp)
        elif row.get("role") == "user" and stamp is not None and previous_assistant_time is not None:
            tool_time += max(0.0, float(stamp) - previous_assistant_time)
            previous_assistant_time = None
    task_messages = [row for row in messages if row.get("role") != "exit"]
    commands = [
        str(action.get("command") or "")
        for row in calls for action in ((row.get("extra") or {}).get("actions") or ())
    ]
    mentioned_paths = sorted({
        match.group(0).lstrip("./")
        for command in commands
        for match in re.finditer(r"(?:\.?\.?/)?(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]+", command)
        if match.group(0) != "patch.txt"
    })
    patch = str((payload.get("info") or {}).get("submission") or "")
    modified_paths = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)))
    return {
        "session_id": session_id_for_messages(task_messages) if task_messages else None,
        "cumulative_prompt_tokens": cumulative,
        "max_prompt_tokens": unique,
        "repeated_context_tokens_estimate": repeated,
        "repeated_context_fraction_estimate": repeated / cumulative if cumulative else 0.0,
        "output_tokens": output_tokens,
        "model_call_count": len(calls),
        "tool_call_count": sum(len(((row.get("extra") or {}).get("actions") or ())) for row in calls),
        "commands_executed": len(commands),
        "files_inspected": len(mentioned_paths),
        "unique_files_inspected": mentioned_paths,
        "files_modified": len(modified_paths),
        "modified_file_paths": modified_paths,
        "wall_time_s": max(timestamps) - min(timestamps) if len(timestamps) > 1 else None,
        "tool_time_s": tool_time if timestamps else None,
        "termination_reason": (payload.get("info") or {}).get("exit_status"),
        "patch": patch,
    }


def _load_treatment_traces(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if not path.is_file():
        return grouped
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row.get("session_id")), []).append(row)
    return grouped


def _aggregate_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    logical = sum(int(row.get("logical_input_tokens_estimate") or 0) for row in rows)
    physical = sum(int(row.get("physical_input_tokens_estimate") or 0) for row in rows)
    update_counts: dict[str, int] = {}
    for row in rows:
        mode = row.get("resource_update_mode")
        if mode:
            update_counts[str(mode)] = update_counts.get(str(mode), 0) + 1
    context_retention = [
        int(row.get("physical_input_tokens_estimate") or 0)
        / int(row.get("logical_input_tokens_estimate") or 1)
        for row in rows if int(row.get("logical_input_tokens_estimate") or 0) > 0
    ]
    history_kv_retention = [
        float(
            row.get("engine_reported_history_kv_retention_fraction")
            if row.get("engine_reported_history_kv_retention_fraction") is not None
            else row["realized_retention_fraction"]
        )
        for row in rows if (
            row.get("engine_reported_history_kv_retention_fraction") is not None
            or row.get("realized_retention_fraction") is not None
        )
    ]

    def sum_reported(key: str) -> int | None:
        values = [int(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) if values else None

    def max_reported(key: str) -> int | None:
        values = [int(row[key]) for row in rows if row.get(key) is not None]
        return max(values) if values else None

    def sum_reported_alias(primary: str, fallback: str) -> int | None:
        values = [
            int(
                row[primary]
                if row.get(primary) is not None else row[fallback]
            )
            for row in rows
            if row.get(primary) is not None or row.get(fallback) is not None
        ]
        return sum(values) if values else None
    return {
        "logical_input_tokens_estimate": logical,
        "physical_input_tokens_estimate": physical,
        "selected_tokens_estimate": sum(int(row.get("selected_tokens_estimate") or 0) for row in rows),
        "tokens_avoided_estimate": max(0, logical - physical),
        "token_saving_fraction_estimate": max(0, logical - physical) / logical if logical else 0.0,
        "route_time_s": sum(float(row.get("route_time_s") or 0) for row in rows),
        "token_estimator": rows[0].get("token_estimator"),
        "selected_resource_digests": [
            row["selected_resource_digest"]
            for row in rows if row.get("selected_resource_digest")
        ],
        "prefix_cached_tokens": sum(int(row.get("prefix_cached_tokens") or 0) for row in rows),
        "engine_cached_tokens_total": sum(
            int(row.get("engine_cached_tokens_total") or 0) for row in rows
        ),
        "prefix_cache_observed_requests": sum(
            int(row.get("prefix_cache_observed") is True) for row in rows
        ),
        "native_tokens": sum(int(row.get("native_tokens") or 0) for row in rows),
        "wire_tokens": sum(int(row.get("wire_tokens") or 0) for row in rows),
        "physical_kv_copy_observed": any(
            row.get("physical_kv_copy") is True for row in rows
        ),
        "physical_kv_copy_bytes": sum_reported("physical_kv_copy_bytes"),
        "total_kv_copy_bytes": sum_reported("total_kv_copy_bytes"),
        "canonical_suffix_graft_d2d_bytes": sum_reported(
            "canonical_suffix_graft_d2d_bytes"
        ),
        "host_to_device_bytes": sum_reported("host_to_device_bytes"),
        "selected_kv_tokens": sum(
            int(row.get("selected_kv_tokens") or 0) for row in rows
        ),
        "selected_text_reencoded_tokens": sum(
            int(row.get("selected_text_reencoded_tokens") or 0) for row in rows
        ),
        "selected_history_reencoded_tokens": sum_reported_alias(
            "selected_history_reencoded_tokens", "selected_text_reencoded_tokens"
        ),
        "realized_retention_fraction": (
            physical / logical if logical else None
        ),
        "realized_retention_fraction_min": (
            min(context_retention) if context_retention else None
        ),
        "realized_retention_fraction_max": (
            max(context_retention) if context_retention else None
        ),
        "engine_reported_history_kv_retention_fraction_min": (
            min(history_kv_retention) if history_kv_retention else None
        ),
        "engine_reported_history_kv_retention_fraction_max": (
            max(history_kv_retention) if history_kv_retention else None
        ),
        "consumer_temporary_bytes": sum_reported("consumer_temporary_bytes"),
        "consumer_temporary_peak_bytes": max_reported(
            "consumer_temporary_peak_bytes"
        ),
        "fused_attention_calls": sum_reported("fused_attention_calls"),
        "full_retention_requests": sum(
            int(row.get("full_retention") is True) for row in rows
        ),
        "sparse_kv_requests": sum(
            int(row.get("full_retention") is False) for row in rows
        ),
        "resource_update_counts": update_counts,
        "resource_prefix_cached_tokens": sum(
            int(row.get("resource_prefix_cached_tokens") or 0) for row in rows
        ),
        "resource_evaluated_tokens": sum(
            int(row.get("resource_evaluated_tokens") or 0) for row in rows
        ),
        "resource_total_tokens": sum(
            int(row.get("resource_total_tokens") or 0) for row in rows
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _grader_error_type(log: Path, instance_id: str) -> str:
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    if instance_id in text and "Patch Apply Failed" in text:
        return "patch_apply_failed"
    return "official_grader_error"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-card", type=Path, required=True)
    parser.add_argument(
        "--task-index", type=int,
        help=(
            "Run one 1-based task from the locked benchmark card without "
            "resampling or creating an outcome-dependent cohort."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-revision", default="NOT_REPORTED_BY_SOURCE")
    parser.add_argument("--tokenizer-revision", default="NOT_REPORTED_BY_SOURCE")
    parser.add_argument("--benchmark-revision", default=PINNED_DATASET_REVISION)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--engine-version", default=EXPECTED_PACKAGES["vllm"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--harness-version", default=EXPECTED_PACKAGES["mini-swe-agent"])
    parser.add_argument("--grader-version", default=EXPECTED_PACKAGES["swebench"])
    parser.add_argument("--scaffold", default="swebench_backticks.yaml")
    parser.add_argument(
        "--chat-template-no-thinking",
        action="store_true",
        help="Disable model-specific thinking through the OpenAI extra-body contract.",
    )
    parser.add_argument(
        "--prefix-caching",
        action="store_true",
        help="Record that the serving endpoint has ordinary prefix caching enabled.",
    )
    parser.add_argument("--grading", default="SWE-bench 4.1.0 official Docker harness")
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        help="Per-turn completion ceiling forwarded unchanged to mini-swe-agent.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode", choices=("no-pra", *[mode.value for mode in ContextTreatment]),
        default="no-pra",
    )
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--recent-completed-turns", type=int, default=2)
    parser.add_argument("--recent-records-per-turn", type=int, default=2)
    parser.add_argument("--recent-source-turns", type=int, default=1)
    parser.add_argument("--recent-progress-turns", type=int, default=1)
    parser.add_argument("--recent-mutation-turns", type=int, default=1)
    parser.add_argument("--recent-verification-turns", type=int, default=1)
    parser.add_argument("--large-record-chunk-tokens", type=int, default=256)
    parser.add_argument(
        "--max-records-per-turn-before-chunking", type=int, default=8,
    )
    parser.add_argument(
        "--preserve-action-observation-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--causal-bundle-round-up",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--consumption-policy",
        choices=tuple(CONSUMPTION_POLICIES),
        default="standard",
        help="Vary the agent consumption contract without changing PRA retention settings.",
    )
    parser.add_argument(
        "--selection-record", type=Path,
        help="Record exact ordered selected resources for a later transport-equivalence replay.",
    )
    parser.add_argument(
        "--selection-replay", type=Path,
        help="Replay an exact direct-run selection fixture; fail if request content diverges.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--grader-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--prepull-images",
        action="store_true",
        help="Pull each official task image immediately before mini-swe-agent starts it.",
    )
    parser.add_argument(
        "--docker-platform",
        help="Optional platform for task-image acquisition, for example linux/amd64 on Apple Silicon.",
    )
    parser.add_argument("--image-pull-timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--recover-timeout-chunk", type=int,
        help="Normalize a previously observed timed-out chunk without rerunning its agent.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--require-endpoint-preflight",
        action="store_true",
        help=(
            "Require health/model/template admission even for a direct plain "
            "arm, so cross-arm engine configuration can be frozen."
        ),
    )
    parser.add_argument(
        "--endpoint-preflight-receipt",
        type=Path,
        help="Reuse a passed endpoint generation probe after a clean restart.",
    )
    parser.add_argument("--allow-partial-reproduction", action="store_true")
    parser.add_argument(
        "--local-calibration",
        action="store_true",
        help="Validate a pinned local configuration without imposing the H100 source host.",
    )
    result = run(parser.parse_args())
    print(result)


if __name__ == "__main__":
    main()
