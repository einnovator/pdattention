"""Run Paper 6.5 M2--M4 with a frozen pretrained tool-capable model.

M2 tests whether a selected schema is enough for structured call generation.
M3 adds host authorization, pure execution, and typed observations. M4 tests
sequential reactive disclosure. No model weights are trained and no external
side effect is reachable from this runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import (
    WorkflowStep,
    WorkflowTask,
    realistic_tool_catalog,
    workflow_executor,
    workflow_tasks,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_hf.agent_execution import (
    ExecutionAuthorization,
    ToolCall,
    parse_tool_call,
    resource_tool_schema,
)


SEEDS = (11, 23, 37, 53, 71)
M2_CONDITIONS = ("selected", "oracle", "shuffled", "irrelevant", "empty", "eager")
M4_CONDITIONS = ("reactive_jit", "eager_required", "no_refresh")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalize(value: object) -> object:
    if isinstance(value, str):
        return value.strip().casefold()
    return value


def call_matches(call: ToolCall | None, step: WorkflowStep) -> tuple[bool, bool]:
    """Return tool-name and exact-argument correctness separately."""

    if call is None:
        return False, False
    tool_ok = call.name == step.tool_name
    expected = {name: _normalize(value) for name, value in step.arguments.items()}
    actual = {name: _normalize(value) for name, value in call.arguments.items()}
    return tool_ok, tool_ok and actual == expected


def _observation_grounded(text: str, output: Mapping[str, object]) -> bool:
    normalized = text.casefold().replace(" ", "")
    values = [str(value).casefold().replace(" ", "") for value in output.values()]
    return bool(values) and all(value in normalized for value in values)


class FrozenToolModel:
    """Minimal deterministic chat wrapper that records prompt and generation cost."""

    def __init__(self, model_id: str, revision: str, device: torch.device) -> None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        resources: Sequence,
        *,
        max_new_tokens: int = 80,
    ) -> tuple[str, dict[str, object]]:
        tools = [resource_tool_schema(resource) for resource in resources]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools:
            kwargs["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        generated = output[0, inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, {
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "generated_tokens": int(generated.shape[0]),
            "generation_seconds": seconds,
            "disclosed_tools": len(resources),
            "tool_definition_tokens": sum(
                len(self.tokenizer.encode(json.dumps(resource_tool_schema(resource))))
                for resource in resources
            ),
        }


def _prompt_variant(query: str, seed: int) -> str:
    prefixes = (
        "Use exactly one available tool. ",
        "Call the matching function with all required arguments. ",
        "Complete this request with one function call. ",
        "Choose the correct available function and call it. ",
        "Return a single tool call for this request. ",
    )
    return prefixes[SEEDS.index(seed) % len(prefixes)] + query


def _wrong_resource(resources, target, seed: int, *, same_family: bool):
    candidates = [resource for resource in resources if resource.uri != target.uri]
    if same_family:
        object_types = set(target.metadata.get("object_types", ()))
        related = [
            resource
            for resource in candidates
            if object_types & set(resource.metadata.get("object_types", ()))
        ]
        if related:
            candidates = related
    return candidates[random.Random(seed).randrange(len(candidates))]


def _m2_disclosure(condition, resources, target, seed):
    if condition in {"selected", "oracle"}:
        return (target,)
    if condition == "shuffled":
        return (_wrong_resource(resources, target, seed, same_family=True),)
    if condition == "irrelevant":
        return (_wrong_resource(resources, target, seed + 1000, same_family=False),)
    if condition == "empty":
        return ()
    if condition == "eager":
        values = list(resources)
        random.Random(seed).shuffle(values)
        return tuple(values)
    raise ValueError(condition)


def run_m2(model: FrozenToolModel, seeds: Sequence[int], max_tasks: int | None) -> list[dict]:
    resources = realistic_tool_catalog()
    by_name = {resource.name: resource for resource in resources}
    tasks = [task for task in workflow_tasks() if len(task.steps) == 1]
    tasks = tasks[:max_tasks] if max_tasks is not None else tasks
    rows = []
    for seed in seeds:
        for task in tasks:
            step = task.steps[0]
            target = by_name[step.tool_name]
            for condition in M2_CONDITIONS:
                disclosed = _m2_disclosure(condition, resources, target, seed)
                text, cost = model.generate(
                    ({"role": "user", "content": _prompt_variant(task.query, seed)},),
                    disclosed,
                )
                call = parse_tool_call(text)
                tool_ok, arguments_ok = call_matches(call, step)
                rows.append({
                    "milestone": "M2",
                    "seed": seed,
                    "task_id": task.task_id,
                    "family": task.family,
                    "condition": condition,
                    "target_tool": step.tool_name,
                    "generated_tool": "" if call is None else call.name,
                    "parse_valid": call is not None,
                    "tool_correct": tool_ok,
                    "arguments_correct": arguments_ok,
                    "end_to_end_success": arguments_ok,
                    "generated_text": text.replace("\r", "\\r").replace("\n", "\\n"),
                    **cost,
                })
                print(f"[M2 seed={seed} task={task.task_id} {condition}] success={arguments_ok}", flush=True)
    return rows


def _final_messages(query: str, call_text: str, observation) -> tuple[dict[str, str], ...]:
    return (
        {"role": "user", "content": query},
        {"role": "assistant", "content": call_text},
        {"role": "tool", "content": observation.content},
        {
            "role": "user",
            "content": "Report the latest tool result. Include every returned value and do not call another tool.",
        },
    )


def run_m3(model: FrozenToolModel, seeds: Sequence[int], max_tasks: int | None) -> list[dict]:
    resources = realistic_tool_catalog()
    by_name = {resource.name: resource for resource in resources}
    tasks = [task for task in workflow_tasks() if len(task.steps) == 1]
    tasks = tasks[:max_tasks] if max_tasks is not None else tasks
    rows = []
    for seed in seeds:
        for task in tasks:
            step = task.steps[0]
            target = by_name[step.tool_name]
            query = _prompt_variant(task.query, seed)
            call_text, call_cost = model.generate(({"role": "user", "content": query},), (target,))
            call = parse_tool_call(call_text)
            executor = workflow_executor(resources, task)
            result = executor.execute(
                call,
                selected_uris=(target.uri,),
                authorization=ExecutionAuthorization(
                    frozenset((target.uri,)), allow_writes=True, allow_destructive=False
                ),
                call_id=f"{task.task_id}-seed{seed}",
            )
            final_text = ""
            final_cost = {"prompt_tokens": 0, "generated_tokens": 0, "generation_seconds": 0.0}
            grounded = False
            if result.observation is not None:
                final_text, final_cost = model.generate(
                    _final_messages(query, call_text, result.observation), (), max_new_tokens=48
                )
                grounded = _observation_grounded(final_text, result.output)
            rows.append({
                "milestone": "M3",
                "seed": seed,
                "task_id": task.task_id,
                "family": task.family,
                "call_valid": call is not None,
                "execution_accepted": result.accepted,
                "execution_reason": result.reason,
                "typed_observation": result.observation is not None,
                "observation_grounded": grounded,
                "end_to_end_success": bool(result.executed and grounded),
                "call_prompt_tokens": call_cost["prompt_tokens"],
                "final_prompt_tokens": final_cost["prompt_tokens"],
                "generation_seconds": float(call_cost["generation_seconds"]) + float(final_cost["generation_seconds"]),
                "observation_uri": "" if result.observation is None else result.observation.uri,
                "final_text": final_text.replace("\r", "\\r").replace("\n", "\\n"),
            })
            print(f"[M3 seed={seed} task={task.task_id}] executed={result.executed} grounded={grounded}", flush=True)
    return rows


def _resources_for_step(condition, resources, task, step_index, seed):
    by_name = {resource.name: resource for resource in resources}
    if condition == "reactive_jit":
        return (by_name[task.steps[step_index].tool_name],)
    if condition == "eager_required":
        values = [by_name[name] for name in dict.fromkeys(task.required_tools)]
        random.Random(seed).shuffle(values)
        return tuple(values)
    if condition == "no_refresh":
        return (by_name[task.steps[0].tool_name],)
    raise ValueError(condition)


def run_m4(model: FrozenToolModel, seeds: Sequence[int], max_tasks: int | None) -> list[dict]:
    resources = realistic_tool_catalog()
    tasks = [task for task in workflow_tasks() if len(task.steps) > 1]
    tasks = tasks[:max_tasks] if max_tasks is not None else tasks
    rows = []
    for seed in seeds:
        for task in tasks:
            for condition in M4_CONDITIONS:
                executor = workflow_executor(resources, task)
                messages: list[dict[str, str]] = [
                    {"role": "user", "content": _prompt_variant(task.query, seed) + " Execute one step at a time."}
                ]
                observations = []
                completed = 0
                retrievals = 0
                prompt_tokens = definition_tokens = generated_tokens = 0
                generation_seconds = 0.0
                failure = ""
                for step_index, step in enumerate(task.steps):
                    disclosed = _resources_for_step(condition, resources, task, step_index, seed)
                    if condition == "reactive_jit" and step_index > 0:
                        retrievals += 1
                    text, cost = model.generate(messages, disclosed)
                    call = parse_tool_call(text)
                    tool_ok, arguments_ok = call_matches(call, step)
                    prompt_tokens += int(cost["prompt_tokens"])
                    definition_tokens += int(cost["tool_definition_tokens"])
                    generated_tokens += int(cost["generated_tokens"])
                    generation_seconds += float(cost["generation_seconds"])
                    authorization = ExecutionAuthorization(
                        frozenset(resource.uri for resource in disclosed),
                        allow_writes=True,
                        allow_destructive=False,
                    )
                    result = executor.execute(
                        call,
                        selected_uris=tuple(resource.uri for resource in disclosed),
                        authorization=authorization,
                        prior_observations=observations,
                        call_id=f"{task.task_id}-{condition}-seed{seed}-step{step_index + 1}",
                    )
                    rows.append({
                        "milestone": "M4",
                        "seed": seed,
                        "task_id": task.task_id,
                        "family": task.family,
                        "plan_horizon": len(task.steps),
                        "condition": condition,
                        "step": step_index + 1,
                        "expected_tool": step.tool_name,
                        "generated_tool": "" if call is None else call.name,
                        "tool_correct": tool_ok,
                        "arguments_correct": arguments_ok,
                        "executed": result.executed,
                        "failure_reason": "" if result.executed else result.reason,
                        "disclosed_tools": len(disclosed),
                        "mid_execution_retrievals": retrievals,
                        "cumulative_definition_tokens": definition_tokens,
                        "cumulative_prompt_tokens": prompt_tokens,
                        "cumulative_generated_tokens": generated_tokens,
                        "cumulative_generation_seconds": generation_seconds,
                    })
                    if not arguments_ok:
                        failure = "wrong_call"
                        break
                    if not result.executed or result.observation is None:
                        failure = result.reason
                        break
                    completed += 1
                    observations.append(result.observation)
                    messages.extend((
                        {"role": "assistant", "content": text},
                        {"role": "tool", "content": result.observation.content},
                        {"role": "user", "content": "Continue the workflow with exactly one next tool call."},
                    ))
                rows.append({
                    "milestone": "M4-summary",
                    "seed": seed,
                    "task_id": task.task_id,
                    "family": task.family,
                    "plan_horizon": len(task.steps),
                    "condition": condition,
                    "step": 0,
                    "expected_tool": "",
                    "generated_tool": "",
                    "tool_correct": completed == len(task.steps),
                    "arguments_correct": completed == len(task.steps),
                    "executed": completed == len(task.steps),
                    "failure_reason": failure,
                    "disclosed_tools": len(set(task.required_tools)) if condition == "eager_required" else 1,
                    "mid_execution_retrievals": retrievals,
                    "cumulative_definition_tokens": definition_tokens,
                    "cumulative_prompt_tokens": prompt_tokens,
                    "cumulative_generated_tokens": generated_tokens,
                    "cumulative_generation_seconds": generation_seconds,
                })
                print(f"[M4 seed={seed} task={task.task_id} {condition}] {completed}/{len(task.steps)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--stages", default="m2,m3,m4")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/pretrained_bridge",
    )
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    stages = {value.strip().casefold() for value in args.stages.split(",")}
    device = torch.device(args.device)
    started = time.perf_counter()
    model = FrozenToolModel(args.model_id, args.revision, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    if "m2" in stages:
        rows = run_m2(model, seeds, args.max_tasks)
        _write_csv(args.output_dir / "m2_rows.csv", rows)
        counts["m2_rows"] = len(rows)
    if "m3" in stages:
        rows = run_m3(model, seeds, args.max_tasks)
        _write_csv(args.output_dir / "m3_rows.csv", rows)
        counts["m3_rows"] = len(rows)
    if "m4" in stages:
        rows = run_m4(model, seeds, args.max_tasks)
        _write_csv(args.output_dir / "m4_rows.csv", rows)
        counts["m4_rows"] = len(rows)
    manifest = {
        "schema_version": "1.0",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "model_frozen": True,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seeds": list(seeds),
        "stages": sorted(stages),
        "counts": counts,
        "runtime": runtime_metadata(),
        "elapsed_seconds": time.perf_counter() - started,
        "execution_boundary": "registered in-memory pure handlers only; no external side effects",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
