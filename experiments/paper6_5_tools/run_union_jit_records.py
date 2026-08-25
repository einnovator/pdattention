"""Run Paper 6.5 E4--E6 with frozen Qwen and pure host handlers.

E4 evaluates resolver-produced candidate palettes in a reactive loop. E5 and
E6 isolate record preservation from deliberately partial tool-schema controls.
No discovered tool can authorize itself and all handlers are in-memory fixtures.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import struct
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import WorkflowTask, realistic_tool_catalog, workflow_executor, workflow_tasks
from data.python_tool_ingestion_cases import PAPER6_5_TOOL_CALLABLES
from data.semantic_concepts import canonical_concept_map
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper6_5_tools.run_m2_m4_pretrained import FrozenToolModel, _prompt_variant, call_matches
from pra_hf.agent_disclosure import DisclosureMode, ToolCapabilityGraph, ToolDisclosurePolicy
from pra_hf.agent_execution import ExecutionAuthorization, parse_tool_call, resource_tool_schema
from pra_hf.context_records import (
    OverflowBehavior,
    materialize_authoritative_slice,
    tool_catalog_slice_records,
)
from pra_hf.semantic_resource_discovery import CompactEmbeddingEncoder, ExternalSemanticIndex, ToolSemanticCard
from pra_hf.tool_records import PythonTypeSchemaCache, tool_record_from_callable
from pra_hf.union_discovery import (
    CandidateSet,
    CandidateProvenance,
    ChannelHit,
    ToolDiscoveryMode,
    ToolDiscoveryPolicy,
    UnionStrategy,
    discover_candidate_set,
)


SEEDS = (11, 23, 37, 53, 71)
E4_CONDITIONS = (
    "top1_jit",
    "union_jit_k2",
    "union_jit_k4",
    "union_jit_k6",
    "union_jit_k8",
    "static_oracle",
    "static_graph",
    "all_tools",
)


STEP_INTENTS: Mapping[tuple[str, int], str] = {
    ("m4-user-3", 0): "Find the account associated with alice@example.com.",
    ("m4-user-3", 1): "Check whether the identified account may be changed.",
    ("m4-user-3", 2): "Set the identified account status to reviewed.",
    ("m4-doc-4", 0): "Find the document titled PRA Notes.",
    ("m4-doc-4", 1): "Read the identified document.",
    ("m4-doc-4", 2): "Export the identified document as PDF.",
    ("m4-doc-4", 3): "Create a report titled PRA digest from the exported artifact.",
    ("m4-repo-5", 0): "Find the source repository named pra-core.",
    ("m4-repo-5", 1): "Retrieve the identified repository details.",
    ("m4-repo-5", 2): "Create an issue titled Routing audit in the identified repository.",
    ("m4-repo-5", 3): "Set the identified issue status to open.",
    ("m4-repo-5", 4): "Notify user u17 that the Routing audit issue is open.",
}


class StreamLoadedFrozenToolModel(FrozenToolModel):
    """Load one safetensors tensor at a time for paging-constrained hosts."""

    def __init__(self, model_id: str, revision: str, device: torch.device) -> None:
        self.device = device
        local_path = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        config = AutoConfig.from_pretrained(local_path, local_files_only=True)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)
        weight_files = sorted(local_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors weights found under {local_path}")
        dtype_map = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F64": torch.float64,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        for weight_file in weight_files:
            with weight_file.open("rb") as stream:
                header_length = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_length))
                data_start = 8 + header_length
                for name, metadata in header.items():
                    if name == "__metadata__":
                        continue
                    start, end = metadata["data_offsets"]
                    stream.seek(data_start + start)
                    raw = bytearray(stream.read(end - start))
                    value = torch.frombuffer(raw, dtype=dtype_map[metadata["dtype"]]).reshape(metadata["shape"])
                    target_dtype = torch.float16 if value.is_floating_point() else value.dtype
                    set_module_tensor_to_device(self.model, name, device, value=value, dtype=target_dtype)
                    del value, raw
        self.model.tie_weights()
        unresolved = [name for name, value in self.model.named_parameters() if value.device.type == "meta"]
        if unresolved:
            raise RuntimeError(f"Unresolved meta parameters after streaming load: {unresolved[:5]}")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)


class OllamaToolModel:
    """Frozen local Qwen3 Q4 backend with the same resource-facing contract."""

    def __init__(self, model_name: str, revision: str) -> None:
        self.model_name = model_name
        self.revision = revision
        local_path = Path(snapshot_download(MODEL_ID, revision=revision, local_files_only=True))
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        self.config = AutoConfig.from_pretrained(local_path, local_files_only=True)
        self.device = torch.device("cpu")

    def generate(self, messages, resources, *, max_new_tokens: int = 48):
        formatted = []
        for message in messages:
            value = dict(message)
            if value.get("role") == "user" and "/no_think" not in value.get("content", ""):
                value["content"] = "/no_think " + value.get("content", "")
            if value.get("role") == "assistant":
                call = parse_tool_call(value.get("content", ""))
                if call is not None:
                    value = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": call.name, "arguments": dict(call.arguments)}}],
                    }
            formatted.append(value)
        tools = [resource_tool_schema(resource) for resource in resources]
        payload = {
            "model": self.model_name,
            "stream": False,
            # This host cannot retain the runner beside the CUDA-enabled
            # Python process within its committed-memory limit.
            "keep_alive": 0,
            "messages": formatted,
            "tools": tools,
            "options": {"temperature": 0, "num_predict": max_new_tokens, "num_gpu": 0},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read())
        message = result.get("message", {})
        calls = message.get("tool_calls", ())
        if calls:
            function = calls[0].get("function", {})
            text = "<tool_call>" + json.dumps({
                "name": function.get("name"),
                "arguments": function.get("arguments", {}),
            }, separators=(",", ":")) + "</tool_call>"
        else:
            text = str(message.get("content", ""))
        return text, {
            "prompt_tokens": int(result.get("prompt_eval_count", 0)),
            "generated_tokens": int(result.get("eval_count", 0)),
            "generation_seconds": time.perf_counter() - started,
            "disclosed_tools": len(resources),
            "tool_definition_tokens": sum(
                len(self.tokenizer.encode(json.dumps(resource_tool_schema(resource))))
                for resource in resources
            ),
        }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _append_unique(existing, incoming, key_names):
    values = {tuple(str(row.get(name, "")) for name in key_names): dict(row) for row in existing}
    for row in incoming:
        values[tuple(str(row.get(name, "")) for name in key_names)] = dict(row)
    return list(values.values())


def _auto_resources():
    concepts = canonical_concept_map()
    cache = PythonTypeSchemaCache()
    return tuple(
        tool_record_from_callable(
            function,
            namespace="paper6_5_auto",
            tenant_id="paper6_5",
            concept_map=concepts,
            type_cache=cache,
        ).to_agent_resource()
        for function in PAPER6_5_TOOL_CALLABLES
    )


def _resolver_scores(
    resources,
    intents: Sequence[str],
    *,
    model_root: Path,
    device: str,
) -> list[dict[str, dict[str, float]]]:
    encoder = CompactEmbeddingEncoder(
        str(model_root / "bge-small-en-v1.5"),
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        device=device,
        query_prefix="Represent this sentence for searching relevant passages: ",
        pooling="cls",
    )
    cards = tuple(ToolSemanticCard.from_resource(row) for row in resources)
    tool_vectors = encoder.encode([row.structured_text for row in cards])
    query_vectors = encoder.encode(list(intents), query=True)
    external = ExternalSemanticIndex(resources, canonical_concept_map())
    graph = ToolCapabilityGraph(resources)
    output = []
    for query_index, intent in enumerate(intents):
        scored = external.score(intent)
        channels = {
            "lexical": {row.uri: max(row.token, row.bm25) for row in scored},
            "dictionary": {row.uri: row.dictionary for row in scored},
            "tags": {row.uri: row.tags for row in scored},
            "embedding": {
                resource.uri: float(((query_vectors[query_index] @ tool_vectors[index]) + 1.0) / 2.0)
                for index, resource in enumerate(resources)
            },
        }
        fused = {
            resource.uri: 0.75 * channels["tags"][resource.uri] + 0.25 * channels["embedding"][resource.uri]
            for resource in resources
        }
        roots = sorted(fused, key=lambda uri: (-fused[uri], uri))[:2]
        graph_scores: dict[str, float] = {}
        for root in roots:
            for edge in graph.outgoing.get(root, ()):
                graph_scores[edge.target_uri] = max(graph_scores.get(edge.target_uri, 0.0), float(edge.weight))
        channels["graph"] = graph_scores
        output.append(channels)
    del encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def _candidate_tables(resources, model_root: Path, device: str):
    tasks = tuple(task for task in workflow_tasks() if len(task.steps) > 1)
    keys = tuple((task.task_id, index) for task in tasks for index in range(len(task.steps)))
    intents = tuple(STEP_INTENTS[key] for key in keys)
    scores = _resolver_scores(resources, intents, model_root=model_root, device=device)
    by_uri = {row.uri: row for row in resources}
    by_name = {row.name: row for row in resources}
    graph = ToolCapabilityGraph(resources)
    output = {}

    def static_candidates(base, uris, source):
        values = tuple(uris)
        return replace(
            base,
            candidate_uris=values,
            provenance=tuple(
                CandidateProvenance(uri, (ChannelHit(source, rank, 1.0),), source, rank)
                for rank, uri in enumerate(values, start=1)
            ),
            max_candidates=len(values),
        )

    for key, intent, channels in zip(keys, intents, scores):
        task = next(row for row in tasks if row.task_id == key[0])
        output[(key, "top1_jit")] = discover_candidate_set(
            intent,
            resources,
            channels,
            ToolDiscoveryPolicy(
                mode=ToolDiscoveryMode.TOP_K,
                strategy=UnionStrategy.FUSED_SCORE,
                max_candidates=1,
                graph=True,
            ),
        )
        for budget in (2, 4, 6, 8):
            output[(key, f"union_jit_k{budget}")] = discover_candidate_set(
                intent,
                resources,
                channels,
                ToolDiscoveryPolicy(
                    mode=ToolDiscoveryMode.UNION,
                    strategy=UnionStrategy.DIVERSITY_UNION,
                    max_candidates=budget,
                    graph=True,
                ),
            )
        required = tuple(by_name[name].uri for name in dict.fromkeys(task.required_tools))
        output[(key, "static_oracle")] = static_candidates(
            output[(key, "top1_jit")], required, "static_oracle"
        )
        graph_trace = graph.disclose(
            (required[0],),
            ToolDisclosurePolicy(mode=DisclosureMode.PLANNING, max_tools=8),
        )
        output[(key, "static_graph")] = static_candidates(
            output[(key, "top1_jit")], graph_trace.disclosed_uris, "static_graph"
        )
        safe_all = tuple(row.uri for row in resources if row.side_effect_class.value != "destructive")
        output[(key, "all_tools")] = static_candidates(
            output[(key, "top1_jit")], safe_all, "all_tools"
        )
    return tasks, output, by_uri


def _save_candidate_tables(path: Path, tables) -> None:
    rows = []
    for ((task_id, step), condition), candidates in sorted(tables.items()):
        rows.append({
            "task_id": task_id,
            "step": step,
            "condition": condition,
            "mode": candidates.mode.value,
            "strategy": candidates.strategy.value,
            "candidate_uris": list(candidates.candidate_uris),
            "max_candidates": candidates.max_candidates,
            "explicit_resolution": candidates.explicit_resolution,
            "provenance": [asdict(row) for row in candidates.provenance],
        })
    path.write_text(json.dumps({"schema_version": "1.0", "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")


def _load_candidate_tables(path: Path, resources):
    payload = json.loads(path.read_text(encoding="utf-8"))
    tables = {}
    for row in payload["rows"]:
        provenance = tuple(
            CandidateProvenance(
                value["uri"],
                tuple(ChannelHit(**hit) for hit in value["sources"]),
                value["admission_source"],
                value["admission_rank"],
            )
            for value in row["provenance"]
        )
        tables[((row["task_id"], int(row["step"])), row["condition"])] = CandidateSet(
            mode=ToolDiscoveryMode(row["mode"]),
            strategy=UnionStrategy(row["strategy"]),
            candidate_uris=tuple(row["candidate_uris"]),
            provenance=provenance,
            max_candidates=int(row["max_candidates"]),
            explicit_resolution=bool(row["explicit_resolution"]),
        )
    tasks = tuple(task for task in workflow_tasks() if len(task.steps) > 1)
    return tasks, tables, {row.uri: row for row in resources}


def _kv_bytes_per_token(model) -> int:
    config = model.model.config if hasattr(model, "model") else model.config
    layers = int(config.num_hidden_layers)
    heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    element_size = next(model.model.parameters()).element_size() if hasattr(model, "model") else 2
    return 2 * layers * heads * head_dim * element_size


def _materialization_row(model, candidates, resources, *, row_id: str, budget_bytes: int | None = None):
    parent, children = tool_catalog_slice_records(
        candidates, resources, slice_id=f"slice:{row_id}", child_view="full"
    )
    required = sum(row.size_bytes for row in children) + max(len(children) - 1, 0)
    result = materialize_authoritative_slice(
        parent,
        children,
        max_bytes=required if budget_bytes is None else budget_bytes,
        overflow=OverflowBehavior.REQUEST_NARROW,
        token_counter=lambda text: len(model.tokenizer.encode(text)),
        native_kv_bytes_per_token=_kv_bytes_per_token(model),
    )
    return result


def run_e4(model, resources, tasks, tables, by_uri, seeds):
    rows = []
    materialization_rows = []
    for seed in seeds:
        for task in tasks:
            for condition in E4_CONDITIONS:
                executor = workflow_executor(resources, task)
                messages = [{"role": "user", "content": _prompt_variant(task.query, seed) + " Execute one step at a time."}]
                observations = []
                completed = wrong_calls = unsafe_calls = revisions = 0
                prompt_tokens = definition_tokens = total_candidates = 0
                generation_seconds = 0.0
                failure = ""
                disclosed_unique: set[str] = set()
                for step_index, step in enumerate(task.steps):
                    key = ((task.task_id, step_index), condition)
                    candidates = tables[key]
                    disclosed = [by_uri[uri] for uri in candidates.candidate_uris]
                    random.Random(seed * 100 + step_index).shuffle(disclosed)
                    disclosed_unique.update(row.uri for row in disclosed)
                    total_candidates += len(disclosed)
                    materialized = _materialization_row(
                        model, candidates, resources,
                        row_id=f"e4-{seed}-{task.task_id}-{condition}-{step_index}",
                    )
                    materialization_rows.append({
                        "experiment": "E4",
                        "seed": seed,
                        "task_id": task.task_id,
                        "condition": condition,
                        "step": step_index + 1,
                        **asdict(materialized),
                        "selected_record_ids": "|".join(materialized.selected_record_ids),
                        "materialized_record_ids": "|".join(materialized.materialized_record_ids),
                    })
                    text, cost = model.generate(messages, disclosed, max_new_tokens=48)
                    call = parse_tool_call(text)
                    tool_ok, arguments_ok = call_matches(call, step)
                    wrong_calls += int(not tool_ok)
                    unsafe_calls += int(call is not None and call.name in task.unsafe_tools)
                    prompt_tokens += int(cost["prompt_tokens"])
                    definition_tokens += int(cost["tool_definition_tokens"])
                    generation_seconds += float(cost["generation_seconds"])
                    authorization = ExecutionAuthorization(
                        frozenset(row.uri for row in disclosed),
                        allow_writes=True,
                        allow_destructive=False,
                    )
                    result = executor.execute(
                        call,
                        selected_uris=tuple(row.uri for row in disclosed),
                        authorization=authorization,
                        prior_observations=observations,
                        call_id=f"e4-{seed}-{task.task_id}-{condition}-{step_index + 1}",
                    )
                    rows.append({
                        "row_type": "step",
                        "seed": seed,
                        "task_id": task.task_id,
                        "condition": condition,
                        "step": step_index + 1,
                        "expected_tool": step.tool_name,
                        "generated_tool": "" if call is None else call.name,
                        "tool_correct": int(tool_ok),
                        "arguments_correct": int(arguments_ok),
                        "executed": int(result.executed),
                        "failure_reason": "" if result.executed else result.reason,
                        "candidate_count": len(disclosed),
                        "candidate_names": "|".join(row.name for row in disclosed),
                        "prompt_tokens": cost["prompt_tokens"],
                        "definition_tokens": cost["tool_definition_tokens"],
                        "generation_seconds": cost["generation_seconds"],
                    })
                    if not arguments_ok or not result.executed or result.observation is None:
                        failure = result.reason if not result.executed else "wrong_call"
                        break
                    completed += 1
                    observations.append(result.observation)
                    messages.extend((
                        {"role": "assistant", "content": text},
                        {"role": "tool", "content": result.observation.content},
                        {"role": "user", "content": "Continue the workflow with exactly one next tool call."},
                    ))
                rows.append({
                    "row_type": "summary",
                    "seed": seed,
                    "task_id": task.task_id,
                    "condition": condition,
                    "step": 0,
                    "expected_tool": "",
                    "generated_tool": "",
                    "tool_correct": int(completed == len(task.steps)),
                    "arguments_correct": int(completed == len(task.steps)),
                    "executed": int(completed == len(task.steps)),
                    "failure_reason": failure,
                    "candidate_count": total_candidates / max(completed + int(bool(failure)), 1),
                    "candidate_names": "",
                    "prompt_tokens": prompt_tokens,
                    "definition_tokens": definition_tokens,
                    "generation_seconds": generation_seconds,
                    "task_success": int(completed == len(task.steps)),
                    "wrong_tool_calls": wrong_calls,
                    "unsafe_calls": unsafe_calls,
                    "plan_revisions": revisions,
                    "jit_steps": completed + int(bool(failure)),
                    "total_disclosed_tools": len(disclosed_unique),
                })
                print(f"[E4 {seed} {task.task_id} {condition}] {completed}/{len(task.steps)}", flush=True)
    return rows, materialization_rows


def _partial_resource(resource, condition: str):
    schema = json.loads(resource.content)
    if condition == "description_only":
        schema["parameters"] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        return replace(resource, content=json.dumps(schema, sort_keys=True))
    if condition == "drop_required_field":
        parameters = dict(schema["parameters"])
        properties = dict(parameters.get("properties", {}))
        required = list(parameters.get("required", ()))
        if required:
            removed = required[-1]
            required.remove(removed)
            properties.pop(removed, None)
        parameters.update({"properties": properties, "required": required})
        schema["parameters"] = parameters
        return replace(resource, content=json.dumps(schema, sort_keys=True))
    if condition == "parameters_only":
        return replace(resource, description="")
    return resource


def _flat_message(task: WorkflowTask, resource) -> str:
    schema = json.dumps(resource_tool_schema(resource), sort_keys=True)
    return (
        f"{task.query}\nThe available function is serialized below as ordinary text. "
        "Reply with <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>.\n"
        f"{schema}"
    )


def run_e5_e6(model, resources, seeds):
    tasks = tuple(task for task in workflow_tasks() if len(task.steps) == 1)
    by_name = {row.name: row for row in resources}
    atomic_rows = []
    materialization_rows = []
    conditions = ("record_full", "flat_serialization", "parameters_only", "description_only", "drop_required_field")
    for seed in seeds:
        for task in tasks:
            step = task.steps[0]
            target = by_name[step.tool_name]
            for condition in conditions:
                visible = _partial_resource(target, condition)
                if condition == "flat_serialization":
                    messages = [{"role": "user", "content": _flat_message(task, target)}]
                    disclosed = ()
                else:
                    messages = [{"role": "user", "content": _prompt_variant(task.query, seed)}]
                    disclosed = (visible,)
                text, cost = model.generate(messages, disclosed, max_new_tokens=48)
                call = parse_tool_call(text)
                tool_ok, arguments_ok = call_matches(call, step)
                executor = workflow_executor(resources, task)
                result = executor.execute(
                    call,
                    selected_uris=(target.uri,),
                    authorization=ExecutionAuthorization(frozenset((target.uri,)), allow_writes=True),
                    call_id=f"atomic-{seed}-{task.task_id}-{condition}",
                )
                required = set(json.loads(target.content)["parameters"].get("required", ()))
                actual = set() if call is None else set(call.arguments)
                omitted = required - actual
                atomic_rows.append({
                    "seed": seed,
                    "task_id": task.task_id,
                    "condition": condition,
                    "record_aware": int(condition == "record_full"),
                    "partial_tool": int(condition in {"description_only", "drop_required_field"}),
                    "call_valid": int(call is not None),
                    "tool_correct": int(tool_ok),
                    "arguments_correct": int(arguments_ok),
                    "execution_accepted": int(result.accepted),
                    "execution_reason": result.reason,
                    "required_argument_omission": int(bool(omitted)),
                    "omitted_arguments": "|".join(sorted(omitted)),
                    "schema_error": int(result.reason in {"invalid_resource_schema", "missing_required_argument", "unknown_argument", "argument_type_mismatch"}),
                    "incorrect_call": int(not arguments_ok),
                    "unsafe_call": int(call is not None and call.name in task.unsafe_tools),
                    "prompt_tokens": cost["prompt_tokens"],
                    "definition_tokens": cost["tool_definition_tokens"],
                    "generation_seconds": cost["generation_seconds"],
                })
                if condition == "record_full":
                    candidates = discover_candidate_set(
                        target.name,
                        resources,
                        {"lexical": {target.uri: 1.0}},
                        ToolDiscoveryPolicy(mode=ToolDiscoveryMode.RESOLVE, max_candidates=1, embedding=False, dictionary=False, tags=False),
                        explicit_reference_uris=(target.uri,),
                    )
                    full = _materialization_row(model, candidates, resources, row_id=f"e5-{seed}-{task.task_id}")
                    parent, children = tool_catalog_slice_records(
                        candidates,
                        resources,
                        slice_id=f"slice:e5-overflow-{seed}-{task.task_id}",
                        child_view="full",
                    )
                    narrow = materialize_authoritative_slice(
                        parent,
                        children,
                        max_bytes=max(children[0].size_bytes - 1, 0),
                        overflow=OverflowBehavior.REQUEST_NARROW,
                        token_counter=lambda value: len(model.tokenizer.encode(value)),
                        native_kv_bytes_per_token=_kv_bytes_per_token(model),
                    )
                    for overflow_case, value in (("fits", full), ("narrow_required", narrow)):
                        materialization_rows.append({
                            "experiment": "E5",
                            "seed": seed,
                            "task_id": task.task_id,
                            "condition": "record_full",
                            "overflow_case": overflow_case,
                            **asdict(value),
                            "selected_record_ids": "|".join(value.selected_record_ids),
                            "materialized_record_ids": "|".join(value.materialized_record_ids),
                        })
    return atomic_rows, materialization_rows


def _summaries(e4_rows, atomic_rows):
    e4 = []
    for condition in E4_CONDITIONS:
        subset = [row for row in e4_rows if row["row_type"] == "summary" and row["condition"] == condition]
        steps = [row for row in e4_rows if row["row_type"] == "step" and row["condition"] == condition]
        candidate_recall = sum(
            str(row["expected_tool"]) in str(row["candidate_names"]).split("|")
            for row in steps
        ) / max(len(steps), 1)
        e4.append({
            "condition": condition,
            "workflows": len(subset),
            "task_success": sum(int(row["task_success"]) for row in subset) / len(subset),
            "candidate_required_recall": candidate_recall,
            "wrong_tool_calls": sum(int(row["wrong_tool_calls"]) for row in subset) / len(subset),
            "unsafe_calls": sum(int(row["unsafe_calls"]) for row in subset) / len(subset),
            "mean_jit_steps": sum(int(row["jit_steps"]) for row in subset) / len(subset),
            "mean_context_tokens": sum(int(row["definition_tokens"]) for row in subset) / len(subset),
            "mean_disclosed_tools": sum(int(row["total_disclosed_tools"]) for row in subset) / len(subset),
        })
    atomic = []
    for condition in sorted({row["condition"] for row in atomic_rows}):
        subset = [row for row in atomic_rows if row["condition"] == condition]
        atomic.append({
            "condition": condition,
            "calls": len(subset),
            "call_validity": sum(int(row["call_valid"]) for row in subset) / len(subset),
            "argument_accuracy": sum(int(row["arguments_correct"]) for row in subset) / len(subset),
            "execution_acceptance": sum(int(row["execution_accepted"]) for row in subset) / len(subset),
            "required_argument_omission": sum(int(row["required_argument_omission"]) for row in subset) / len(subset),
            "schema_error": sum(int(row["schema_error"]) for row in subset) / len(subset),
        })
    return e4, atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / ".hf_models")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-candidates", action="store_true")
    parser.add_argument("--append", action="store_true", help="Append completed seeds without replacing prior rows.")
    parser.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    parser.add_argument("--ollama-model", default="qwen3:0.6b")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/auto_union_records",
    )
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = _auto_resources()
    candidate_path = args.output_dir / "union_jit_candidate_sets.json"
    if args.prepare_only or args.rebuild_candidates:
        tasks, tables, by_uri = _candidate_tables(resources, args.model_root, args.device)
        _save_candidate_tables(candidate_path, tables)
        if args.prepare_only:
            print(json.dumps({"candidate_sets": len(tables), "path": str(candidate_path)}, indent=2))
            return
    elif not candidate_path.exists():
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--prepare-only",
            "--device", args.device,
            "--model-root", str(args.model_root),
            "--output-dir", str(args.output_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    tasks, tables, by_uri = _load_candidate_tables(candidate_path, resources)
    model = (
        OllamaToolModel(args.ollama_model, MODEL_REVISION)
        if args.backend == "ollama"
        else StreamLoadedFrozenToolModel(MODEL_ID, MODEL_REVISION, torch.device(args.device))
    )
    e4_rows, e4_materialization = run_e4(model, resources, tasks, tables, by_uri, seeds)
    atomic_rows, e5_materialization = run_e5_e6(model, resources, seeds)
    if args.append:
        e4_rows = _append_unique(
            _read_csv(args.output_dir / "union_jit_results.csv"), e4_rows,
            ("row_type", "seed", "task_id", "condition", "step"),
        )
        atomic_rows = _append_unique(
            _read_csv(args.output_dir / "tool_atomicity_controls.csv"), atomic_rows,
            ("seed", "task_id", "condition"),
        )
        prior_materialization = _read_csv(args.output_dir / "record_materialization_results.csv")
        e4_materialization = _append_unique(
            prior_materialization, e4_materialization + e5_materialization,
            ("experiment", "seed", "task_id", "condition", "step", "overflow_case"),
        )
        e5_materialization = []
    e4_summary, atomic_summary = _summaries(e4_rows, atomic_rows)
    _write_csv(args.output_dir / "union_jit_results.csv", e4_rows)
    _write_csv(args.output_dir / "union_jit_summary.csv", e4_summary)
    _write_csv(args.output_dir / "record_materialization_results.csv", e4_materialization + e5_materialization)
    _write_csv(args.output_dir / "tool_atomicity_controls.csv", atomic_rows)
    _write_csv(args.output_dir / "tool_atomicity_summary.csv", atomic_summary)
    union_rows = [row for row in e4_summary if row["condition"].startswith("union_jit")]
    best_union = max(union_rows, key=lambda row: (row["task_success"], -row["mean_context_tokens"]))
    top1 = next(row for row in e4_summary if row["condition"] == "top1_jit")
    full = next(row for row in atomic_summary if row["condition"] == "record_full")
    partial = [row for row in atomic_summary if row["condition"] in {"description_only", "drop_required_field"}]
    frontier_rows = _read_csv(args.output_dir / "union_recall_frontier.csv")
    best_union_k = int(best_union["condition"].rsplit("k", 1)[1])
    diversity_recall = float(next(
        row["required_recall"] for row in frontier_rows
        if row["strategy"] == "diversity_union" and int(row["max_candidates"]) == best_union_k
    ))
    fused_recall = float(next(
        row["required_recall"] for row in frontier_rows
        if row["strategy"] == "fused_score" and int(row["max_candidates"]) == best_union_k
    ))
    completed_seeds = sorted({int(row["seed"]) for row in e4_rows if row["row_type"] == "summary"})
    findings = {
        "schema_version": "1.0",
        "model_id": MODEL_ID if args.backend == "transformers" else args.ollama_model,
        "model_revision": MODEL_REVISION,
        "model_frozen": True,
        "generation_backend": args.backend,
        "weight_precision": "fp16" if args.backend == "transformers" else "Q4_K_M",
        "device": args.device,
        "seeds": completed_seeds,
        "e4": e4_summary,
        "e5_e6": atomic_summary,
        "gates": {
            "union_default": bool(
                diversity_recall > fused_recall
                and best_union["task_success"] >= top1["task_success"]
                and best_union["unsafe_calls"] == 0
            ),
            "union_recall_dominates_fusion": diversity_recall > fused_recall,
            "record_atomic_default": bool(full["execution_acceptance"] >= max(row["execution_acceptance"] for row in partial)),
            "paper7_progressive_detail": "not_implemented",
        },
        "runtime": runtime_metadata(),
        "elapsed_seconds": time.perf_counter() - started,
        "execution_boundary": "registered in-memory pure handlers only; discovery never authorizes execution",
    }
    (args.output_dir / "union_record_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
