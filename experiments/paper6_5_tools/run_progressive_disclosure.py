"""Run M9 two-phase tool and declarative-skill disclosure on a frozen model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog
from data.declarative_skills import declarative_skill_catalog, skill_semantic_hard_queries
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_hf.agent_execution import (
    ExecutionAuthorization,
    SafeToolExecutor,
    ToolCall,
    parse_tool_call,
    resource_tool_schema,
)
from pra_hf.context_records import serialize_record, tool_definition_record
from pra_hf.progressive_disclosure import CapabilityTransition, transition_selected_capability


OUTPUT = ROOT / "docs/papers/shared/results/paper6_5_tools/progressive_disclosure"
TOOL_CONDITIONS = ("T0_full_all", "T1_selection_only", "T2_selection_to_full", "T3_oracle_full")
SKILL_CONDITIONS = ("S0_full_all", "S1_selection_only", "S2_selection_to_full", "S3_oracle_full")
CAPABILITY_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)
SKILL_PATTERN = re.compile(r"SKILL_APPLIED\s*:\s*([a-z0-9_]+)", re.IGNORECASE)
MAX_RUNTIME_PROMPT_TOKENS = 8192


class _FirstTokenTimer(BaseStreamer):
    """Capture TTFT from the synchronous generation stream."""

    def __init__(self, started: float) -> None:
        self.started = started
        self.prompt_seen = False
        self.first_token_seconds: float | None = None

    def put(self, value) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first_token_seconds is None:
            self.first_token_seconds = time.perf_counter() - self.started

    def end(self) -> None:
        return None


class FrozenCapabilityModel:
    """Deterministic local Qwen wrapper for plain and provider-tool phases."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def native_kv_bytes_per_token(self) -> int:
        config = self.model.config
        layers = int(config.num_hidden_layers)
        kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        bytes_per_value = next(self.model.parameters()).element_size()
        return 2 * layers * kv_heads * head_dim * bytes_per_value

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        prompt: str,
        *,
        resources: Sequence = (),
        max_new_tokens: int = 96,
    ) -> tuple[str, dict[str, object]]:
        messages = [{"role": "user", "content": prompt}]
        template = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if resources:
            template["tools"] = [resource_tool_schema(resource) for resource in resources]
        rendered = self.tokenizer.apply_chat_template(messages, **template)
        inputs = self.tokenizer(rendered, return_tensors="pt")
        prompt_tokens = int(inputs.input_ids.shape[1])
        if prompt_tokens > MAX_RUNTIME_PROMPT_TOKENS:
            return "", {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": 0,
                "ttft_seconds": 0.0,
                "generation_seconds": 0.0,
                "context_fit": 0,
                "generation_error": "prompt_exceeds_runtime_limit",
            }
        inputs = inputs.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        streamer = _FirstTokenTimer(started)
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )
        except torch.OutOfMemoryError:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return "", {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": 0,
                "ttft_seconds": 0.0,
                "generation_seconds": time.perf_counter() - started,
                "context_fit": 0,
                "generation_error": "cuda_out_of_memory",
            }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - started
        generated = output[0, inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, {
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "generated_tokens": int(generated.shape[0]),
            "ttft_seconds": streamer.first_token_seconds or wall,
            "generation_seconds": wall,
            "context_fit": 1,
            "generation_error": "",
        }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _parse_capability(text: str) -> str | None:
    for match in CAPABILITY_PATTERN.findall(text):
        try:
            value = json.loads(match)
        except json.JSONDecodeError:
            continue
        for key in ("capability", "tool", "skill"):
            if isinstance(value.get(key), str):
                return value[key]
    return None


def _parse_tool_proposal(text: str) -> ToolCall | None:
    """Accept provider-wrapped or provider-neutral structured tool-call JSON."""

    wrapped = parse_tool_call(text)
    if wrapped is not None:
        return wrapped
    for match in CAPABILITY_PATTERN.findall(text):
        try:
            payload = json.loads(match)
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            return ToolCall(name=name, arguments=arguments, raw_text=match)
    return None


def _choice_prompt(query: str, payload: str, capability_type: str) -> str:
    return (
        f"Choose the single {capability_type} that best fits the request. The records below are "
        "complete selection views; do not invent another name. Return only JSON in the form "
        f'{{"capability":"name"}}.\n\nREQUEST:\n{query}\n\nCANDIDATES:\n{payload}'
    )


def _selection_tool_prompt(query: str, payload: str) -> str:
    return (
        "Choose and call exactly one candidate using only the compact records below. Return only "
        '<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>. '
        f"Do not ask a question.\n\nREQUEST:\n{query}\n\nCANDIDATES:\n{payload}"
    )


def _tool_prompt(query: str) -> str:
    return "Use exactly one available tool and supply every required argument. " + query


def _skill_prompt(query: str, payload: str) -> str:
    return (
        "Apply the one skill that best fits the request. Obey its visible instructions exactly, "
        "including any required output contract. Do not discuss skill selection.\n\n"
        f"REQUEST:\n{query}\n\nAVAILABLE SKILL RECORDS:\n{payload}"
    )


def _normalize(value: object) -> object:
    return value.strip().casefold() if isinstance(value, str) else value


def _argument_metrics(call: ToolCall | None, expected: Mapping[str, object]) -> tuple[bool, bool]:
    if call is None:
        return False, False
    coverage = all(name in call.arguments for name in expected)
    semantic = coverage and all(
        _normalize(call.arguments[name]) == _normalize(value) for name, value in expected.items()
    )
    return coverage, semantic


def _materialize(records, view: str, model: FrozenCapabilityModel) -> tuple[str, int, float]:
    started = time.perf_counter()
    payload = "\n".join(serialize_record(record, view=view) for record in records)
    elapsed = time.perf_counter() - started
    return payload, model.count(payload), elapsed


def _run_tool_condition(model, condition, case, candidates, resources_by_name, records_by_name, executor):
    candidate_resources = tuple(resources_by_name[name] for name in candidates)
    candidate_records = tuple(records_by_name[name] for name in candidates)
    target = resources_by_name[case["target_name"]]
    phase_a_text = phase_b_text = ""
    phase_a_tokens = phase_b_tokens = 0
    phase_a_materialization = phase_b_materialization = 0.0
    phase_a_cost = {"prompt_tokens": 0, "generated_tokens": 0, "ttft_seconds": 0.0, "generation_seconds": 0.0, "context_fit": 1, "generation_error": ""}
    phase_b_cost = dict(phase_a_cost)
    selected_name = None
    generated = ""
    invocation_count = 1

    if condition == "T0_full_all":
        phase_a_text, phase_a_tokens, phase_a_materialization = _materialize(candidate_records, "full", model)
        generated, phase_a_cost = model.generate(_tool_prompt(case["query"]), resources=candidate_resources)
        call = _parse_tool_proposal(generated)
        selected_name = call.name if call else None
    elif condition == "T1_selection_only":
        phase_a_text, phase_a_tokens, phase_a_materialization = _materialize(candidate_records, "selection", model)
        generated, phase_a_cost = model.generate(_selection_tool_prompt(case["query"], phase_a_text))
        call = _parse_tool_proposal(generated)
        selected_name = call.name if call else None
    elif condition == "T2_selection_to_full":
        phase_a_text, phase_a_tokens, phase_a_materialization = _materialize(candidate_records, "selection", model)
        choice_text, phase_a_cost = model.generate(_choice_prompt(case["query"], phase_a_text, "tool"), max_new_tokens=32)
        selected_name = _parse_capability(choice_text)
        call = None
        if selected_name in records_by_name and selected_name in candidates:
            started = time.perf_counter()
            phase_b = transition_selected_capability(
                candidate_records,
                CapabilityTransition(records_by_name[selected_name].record_id),
                token_counter=model.count,
                native_kv_bytes_per_token=model.native_kv_bytes_per_token,
            )
            phase_b_materialization = time.perf_counter() - started
            phase_b_text = phase_b.serialized_payload
            phase_b_tokens = phase_b.serialized_tokens
            generated, phase_b_cost = model.generate(
                _tool_prompt(case["query"]), resources=(resources_by_name[selected_name],)
            )
            call = _parse_tool_proposal(generated)
            invocation_count = 2
    elif condition == "T3_oracle_full":
        target_record = records_by_name[target.name]
        phase_a_text, phase_a_tokens, phase_a_materialization = _materialize((target_record,), "full", model)
        generated, phase_a_cost = model.generate(_tool_prompt(case["query"]), resources=(target,))
        call = _parse_tool_proposal(generated)
        selected_name = target.name
    else:
        raise ValueError(condition)

    required_coverage, semantic_correct = _argument_metrics(call, case["expected_arguments"])
    selected_resource = resources_by_name.get(selected_name or "")
    selected_uris = () if selected_resource is None else (selected_resource.uri,)
    authorization = ExecutionAuthorization(
        frozenset(selected_uris), allow_writes=True, allow_destructive=False
    )
    result = executor.execute(
        call,
        selected_uris=selected_uris,
        authorization=authorization,
        call_id=f"m9-{case['query_id']}-{condition}",
    )
    choice_correct = selected_name == target.name
    call_correct = call is not None and call.name == target.name
    unsafe_choice = bool(selected_resource and selected_resource.side_effect_class.value == "destructive")
    return {
        "selected_name": selected_name or "",
        "generated_name": "" if call is None else call.name,
        "capability_choice_correct": int(choice_correct),
        "wrong_tool_choice": int(bool(selected_name) and not choice_correct),
        "unsafe_tool_choice": int(unsafe_choice),
        "call_parse_valid": int(call is not None),
        "required_argument_coverage": int(required_coverage),
        "argument_semantic_correct": int(semantic_correct),
        "enum_type_valid": int(result.reason not in {"argument_type_mismatch", "invalid_resource_schema"}),
        "schema_valid_call": int(result.reason not in {"malformed_call", "unknown_tool", "missing_required_argument", "unknown_argument", "argument_type_mismatch", "invalid_resource_schema"}),
        "execution_acceptance": int(result.accepted),
        "task_success": int(choice_correct and call_correct and semantic_correct and result.executed),
        "wrong_tool_execution_attempt": int(call is not None and call.name != target.name),
        "host_rejection": int(not result.accepted),
        "execution_reason": result.reason,
        "retry_count": 0,
        "model_invocations": invocation_count,
        "phase_a_capability_tokens": phase_a_tokens,
        "phase_b_capability_tokens": phase_b_tokens,
        "total_capability_tokens": phase_a_tokens + phase_b_tokens,
        "native_kv_bytes": (phase_a_tokens + phase_b_tokens) * model.native_kv_bytes_per_token,
        "phase_a_ttft_seconds": phase_a_cost["ttft_seconds"],
        "phase_b_ttft_seconds": phase_b_cost["ttft_seconds"],
        "total_ttft_seconds": phase_a_cost["ttft_seconds"] + phase_b_cost["ttft_seconds"],
        "phase_transition_overhead_seconds": phase_b_materialization,
        "materialization_seconds": phase_a_materialization + phase_b_materialization,
        "wall_clock_seconds": phase_a_cost["generation_seconds"] + phase_b_cost["generation_seconds"] + phase_a_materialization + phase_b_materialization,
        "prompt_tokens": phase_a_cost["prompt_tokens"] + phase_b_cost["prompt_tokens"],
        "generated_tokens": phase_a_cost["generated_tokens"] + phase_b_cost["generated_tokens"],
        "context_fit": int(phase_a_cost["context_fit"] and phase_b_cost["context_fit"]),
        "generation_error": "|".join(filter(None, (phase_a_cost["generation_error"], phase_b_cost["generation_error"]))),
        "generated_text": generated.replace("\r", "\\r").replace("\n", "\\n"),
    }


def _skill_metrics(text: str, target_name: str) -> dict[str, int | str]:
    marker = SKILL_PATTERN.search(text)
    selected = marker.group(1).casefold() if marker else ""
    normalized = text.casefold()
    headings = all(value in normalized for value in ("decision", "evidence", "next action"))
    external_claim = any(
        phrase in normalized
        for phrase in ("i have updated", "i have deleted", "i have sent", "was changed successfully")
    )
    choice_correct = selected == target_name
    following = bool(marker and headings and not external_claim)
    return {
        "selected_name": selected,
        "capability_choice_correct": int(choice_correct),
        "wrong_skill_use": int(bool(selected) and not choice_correct),
        "instruction_following_success": int(following),
        "constraint_violation": int(external_claim),
        "task_success": int(choice_correct and following),
    }


def _run_skill_condition(model, condition, query, candidates, skills_by_name, records_by_name):
    candidate_records = tuple(records_by_name[name] for name in candidates)
    target_name = query.target_skill
    phase_a_tokens = phase_b_tokens = 0
    phase_a_materialization = phase_b_materialization = 0.0
    zero = {"prompt_tokens": 0, "generated_tokens": 0, "ttft_seconds": 0.0, "generation_seconds": 0.0, "context_fit": 1, "generation_error": ""}
    phase_a_cost = dict(zero)
    phase_b_cost = dict(zero)
    invocation_count = 1
    selected_name = None

    if condition == "S0_full_all":
        payload, phase_a_tokens, phase_a_materialization = _materialize(candidate_records, "full", model)
        generated, phase_a_cost = model.generate(_skill_prompt(query.query, payload), max_new_tokens=112)
    elif condition == "S1_selection_only":
        payload, phase_a_tokens, phase_a_materialization = _materialize(candidate_records, "selection", model)
        generated, phase_a_cost = model.generate(_skill_prompt(query.query, payload), max_new_tokens=112)
    elif condition == "S2_selection_to_full":
        payload, phase_a_tokens, phase_a_materialization = _materialize(candidate_records, "selection", model)
        choice_text, phase_a_cost = model.generate(_choice_prompt(query.query, payload, "skill"), max_new_tokens=32)
        selected_name = _parse_capability(choice_text)
        generated = ""
        if selected_name in records_by_name and selected_name in candidates:
            started = time.perf_counter()
            phase_b = transition_selected_capability(
                candidate_records,
                CapabilityTransition(records_by_name[selected_name].record_id),
                token_counter=model.count,
                native_kv_bytes_per_token=model.native_kv_bytes_per_token,
            )
            phase_b_materialization = time.perf_counter() - started
            phase_b_tokens = phase_b.serialized_tokens
            generated, phase_b_cost = model.generate(_skill_prompt(query.query, phase_b.serialized_payload), max_new_tokens=112)
            invocation_count = 2
    elif condition == "S3_oracle_full":
        payload, phase_a_tokens, phase_a_materialization = _materialize((records_by_name[target_name],), "full", model)
        generated, phase_a_cost = model.generate(_skill_prompt(query.query, payload), max_new_tokens=112)
    else:
        raise ValueError(condition)
    metrics = _skill_metrics(generated, target_name)
    if condition == "S2_selection_to_full":
        metrics["phase_a_selected_name"] = selected_name or ""
        metrics["capability_choice_correct"] = int(selected_name == target_name)
        metrics["wrong_skill_use"] = int(bool(selected_name) and selected_name != target_name)
        metrics["task_success"] = int(selected_name == target_name and metrics["instruction_following_success"])
    else:
        metrics["phase_a_selected_name"] = ""
    return {
        **metrics,
        "model_invocations": invocation_count,
        "phase_a_capability_tokens": phase_a_tokens,
        "phase_b_capability_tokens": phase_b_tokens,
        "total_capability_tokens": phase_a_tokens + phase_b_tokens,
        "native_kv_bytes": (phase_a_tokens + phase_b_tokens) * model.native_kv_bytes_per_token,
        "phase_a_ttft_seconds": phase_a_cost["ttft_seconds"],
        "phase_b_ttft_seconds": phase_b_cost["ttft_seconds"],
        "total_ttft_seconds": phase_a_cost["ttft_seconds"] + phase_b_cost["ttft_seconds"],
        "phase_transition_overhead_seconds": phase_b_materialization,
        "materialization_seconds": phase_a_materialization + phase_b_materialization,
        "wall_clock_seconds": phase_a_cost["generation_seconds"] + phase_b_cost["generation_seconds"] + phase_a_materialization + phase_b_materialization,
        "prompt_tokens": phase_a_cost["prompt_tokens"] + phase_b_cost["prompt_tokens"],
        "generated_tokens": phase_a_cost["generated_tokens"] + phase_b_cost["generated_tokens"],
        "context_fit": int(phase_a_cost["context_fit"] and phase_b_cost["context_fit"]),
        "generation_error": "|".join(filter(None, (phase_a_cost["generation_error"], phase_b_cost["generation_error"]))),
        "generated_text": generated.replace("\r", "\\r").replace("\n", "\\n"),
    }


def run(args: argparse.Namespace) -> None:
    candidates_payload = json.loads((args.output / "progressive_candidate_sets.json").read_text(encoding="utf-8"))
    candidate_by = {
        (row["resource_type"], row["query_id"], int(row["max_candidates"])): row
        for row in candidates_payload["rows"]
    }
    tool_cases = json.loads((args.output / "tool_progressive_cases.json").read_text(encoding="utf-8"))["rows"][:args.max_tool_cases]
    skill_queries = [row for row in skill_semantic_hard_queries() if row.split == "test"][:args.max_skill_cases]
    requested_budgets = tuple(dict.fromkeys(args.budgets))
    tool_budgets = requested_budgets + ((len(realistic_tool_catalog()),) if args.include_all else ())
    skill_budgets = requested_budgets + ((len(declarative_skill_catalog()),) if args.include_all else ())
    protocol = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": args.device,
        "tool_cases": [row["query_id"] for row in tool_cases],
        "skill_cases": [row.query_id for row in skill_queries],
        "requested_budgets": list(requested_budgets),
        "tool_budgets": list(tool_budgets),
        "skill_budgets": list(skill_budgets),
        "include_all_catalog_stress": args.include_all,
        "tool_conditions": list(TOOL_CONDITIONS),
        "skill_conditions": list(SKILL_CONDITIONS),
        "temperature": 0,
        "callback_behavior": "not_implemented",
        "runtime_prompt_token_limit": MAX_RUNTIME_PROMPT_TOKENS,
    }
    checkpoint_path = args.output / "progressive_disclosure_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() and not args.fresh else {}
    checkpoint_protocol = dict(checkpoint.get("protocol", {}))
    checkpoint_protocol.setdefault("runtime_prompt_token_limit", MAX_RUNTIME_PROMPT_TOKENS)
    if checkpoint and checkpoint_protocol != protocol:
        existing_skill_cases = checkpoint_protocol.get("skill_cases", [])
        requested_skill_cases = protocol["skill_cases"]
        prefix_resume = existing_skill_cases[: len(requested_skill_cases)] == requested_skill_cases
        narrowed_protocol = {**checkpoint_protocol, "skill_cases": requested_skill_cases}
        if not prefix_resume or narrowed_protocol != protocol:
            raise ValueError("Existing progressive-disclosure checkpoint uses another protocol; pass --fresh.")
    tool_rows = list(checkpoint.get("tool_rows", ()))
    requested_skill_ids = set(protocol["skill_cases"])
    skill_rows = [row for row in checkpoint.get("skill_rows", ()) if row["query_id"] in requested_skill_ids]
    model = FrozenCapabilityModel(torch.device(args.device))

    tool_resources = realistic_tool_catalog()
    resources_by_name = {resource.name: resource for resource in tool_resources}
    records_by_name = {resource.name: tool_definition_record(resource) for resource in tool_resources}
    executor = SafeToolExecutor(
        tool_resources,
        {resource.uri: (lambda _arguments, _observations: {"ok": True}) for resource in tool_resources},
    )
    completed = {(row["query_id"], int(row["max_candidates"]), row["condition"]) for row in tool_rows}
    for case in tool_cases:
        for budget in tool_budgets:
            candidates = candidate_by[("tool", case["query_id"], budget)]["candidate_names"]
            for condition in TOOL_CONDITIONS:
                key = (case["query_id"], budget, condition)
                if key in completed:
                    continue
                metrics = _run_tool_condition(model, condition, case, candidates, resources_by_name, records_by_name, executor)
                tool_rows.append({
                    "query_id": case["query_id"],
                    "hardness_level": case["hardness_level"],
                    "max_candidates": budget,
                    "condition": condition,
                    "target_name": case["target_name"],
                    "required_tool_recall_at_k": int(case["target_name"] in candidates),
                    "candidate_names": "|".join(candidates),
                    **metrics,
                })
                _write_checkpoint(checkpoint_path, {"protocol": protocol, "tool_rows": tool_rows, "skill_rows": skill_rows})
                print(f"tool {case['query_id']} K={budget} {condition} success={metrics['task_success']}", flush=True)

    skills = declarative_skill_catalog()
    skill_by_name = {skill.name: skill for skill in skills}
    skill_records = {skill.name: skill.to_context_record() for skill in skills}
    completed = {(row["query_id"], int(row["max_candidates"]), row["condition"]) for row in skill_rows}
    for query in skill_queries:
        for budget in skill_budgets:
            candidates = candidate_by[("skill", query.query_id, budget)]["candidate_names"]
            for condition in SKILL_CONDITIONS:
                key = (query.query_id, budget, condition)
                if key in completed:
                    continue
                metrics = _run_skill_condition(model, condition, query, candidates, skill_by_name, skill_records)
                skill_rows.append({
                    "query_id": query.query_id,
                    "family": query.family,
                    "max_candidates": budget,
                    "condition": condition,
                    "target_name": query.target_skill,
                    "required_skill_recall_at_k": int(query.target_skill in candidates),
                    "candidate_names": "|".join(candidates),
                    **metrics,
                })
                _write_checkpoint(checkpoint_path, {"protocol": protocol, "tool_rows": tool_rows, "skill_rows": skill_rows})
                print(f"skill {query.query_id} K={budget} {condition} success={metrics['task_success']}", flush=True)

    _write_csv(args.output / "tool_progressive_disclosure_results.csv", tool_rows)
    _write_csv(args.output / "skill_progressive_disclosure_results.csv", skill_rows)
    manifest = {
        **protocol,
        "tool_rows": len(tool_rows),
        "skill_rows": len(skill_rows),
        "native_kv_bytes_per_token": model.native_kv_bytes_per_token,
        "runtime": runtime_metadata(),
    }
    (args.output / "progressive_disclosure_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--budgets", nargs="+", type=int, default=(2, 4, 6, 8))
    parser.add_argument(
        "--include-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add separate all-tool and all-skill stress budgets after the matched K sweep.",
    )
    parser.add_argument("--max-tool-cases", type=int, default=8)
    parser.add_argument("--max-skill-cases", type=int, default=8)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
