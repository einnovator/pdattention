"""Validate TEXT/PRA-FULL/PRA-DELTA behavior through a model-backed G10 gateway."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper4_5_runtime.run_agent_transport_protocol import _record
from pra_hf.agent_transport import AgentTurnContext, NegotiatedRemoteBackend
from pra_hf.context_records import RecordType
from pra_hf.deployment import PRAEngineCapabilities, PRAEngineResult
from pra_hf.gateway import PRAGateway, create_gateway_server


DEFAULT_OUTPUT = Path(
    "docs/papers/shared/results/paper4_5_runtime/agent_transport_model_backed"
)


class _FullResourceGateway(PRAGateway):
    """Advertise typed full-resource transport while using the same G10 model."""

    def capabilities(self) -> dict[str, Any]:
        value = super().capabilities()
        value["gateway"]["resource_delta"] = False
        value["effective_capabilities"]["resource_delta"] = False
        return value


class MLXTextModelAdapter:
    """Ordinary E0 adapter that records model-side timing and parity evidence."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str,
        gold_answers: Mapping[str, str],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        # Gold answers stay in the evaluator and never enter the wire request.
        self.gold_answers = dict(gold_answers)
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(
            adapter="mlx_model_backed_g10",
            integration_level="E0",
            logical_refs=False,
            typed_records=False,
            text_fallback=True,
            streaming=False,
        )

    def prepare_session(self, request):
        return request.session_id

    def _prompt(self, request) -> tuple[str, list[int]]:
        options = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if request.tools:
            options["tools"] = list(request.tools)
        try:
            prompt = self.tokenizer.apply_chat_template(list(request.messages), **options)
        except TypeError:
            options.pop("enable_thinking", None)
            prompt = self.tokenizer.apply_chat_template(list(request.messages), **options)
        prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        return prompt, prompt_ids

    def _gold_log_probability(self, prompt_ids: list[int], answer: str) -> tuple[float | None, float]:
        import mlx.core as mx

        gold_ids = list(self.tokenizer.encode(answer, add_special_tokens=False))
        if not prompt_ids or not gold_ids:
            return None, 0.0
        started = time.perf_counter()
        logits = self.model(mx.array([prompt_ids + gold_ids], dtype=mx.int32))
        if isinstance(logits, tuple):
            logits = logits[0]
        mx.eval(logits)
        values = []
        for offset, token_id in enumerate(gold_ids):
            row = logits[0, len(prompt_ids) + offset - 1]
            values.append(float((row[token_id] - mx.logsumexp(row)).item()))
        return sum(values), (time.perf_counter() - started) * 1000.0

    def generate(self, request) -> PRAEngineResult:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        with self._lock:
            prompt, prompt_ids = self._prompt(request)
            started = time.perf_counter()
            arrivals: list[float] = []
            pieces: list[str] = []
            generation_tokens = 0
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=request.resolved_max_new_tokens,
                sampler=make_sampler(temp=0),
            ):
                pieces.append(str(response.text))
                arrivals.append((time.perf_counter() - started) * 1000.0)
                generation_tokens = int(response.generation_tokens)
            completion_ms = (time.perf_counter() - started) * 1000.0
            intervals = [right - left for left, right in zip(arrivals, arrivals[1:])]
            example_id = str(request.task_id or request.metadata.get("example_id", ""))
            gold_log_probability, scoring_ms = self._gold_log_probability(
                prompt_ids, self.gold_answers.get(example_id, "")
            )
            output = "".join(pieces)
            row = {
                "condition": request.metadata.get("condition"),
                "example_id": example_id,
                "messages_sha256": hashlib.sha256(
                    json.dumps(list(request.messages), sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "visible_prompt_tokens": len(prompt_ids),
                "generation_tokens": generation_tokens,
                "ttft_ms": arrivals[0] if arrivals else None,
                "itl_ms": mean(intervals) if intervals else None,
                "completion_ms": completion_ms,
                "gold_log_probability": gold_log_probability,
                "scoring_ms_excluded_from_serving": scoring_ms,
                "task_status": dict(request.metadata.get("task_metadata", {})).get("status"),
                "tool_schema_preserved": bool(request.tools),
                "output": output,
            }
            self.rows.append(row)
            return PRAEngineResult(output, row)

    def stream(self, request):
        raise NotImplementedError

    def close_session(self, session_id):
        return None


def _turn(example, *, condition: str, tokenizer: Any, max_source_tokens: int) -> AgentTurnContext:
    source_tokens = list(tokenizer.encode(example.selected_source, add_special_tokens=False))
    source = tokenizer.decode(source_tokens[:max_source_tokens])
    document = _record(
        f"doc:{example.example_id}",
        RecordType.GENERIC_DOCUMENT,
        {"uri": f"dataset://{example.dataset}/{example.example_id}", "body": source},
        version=example.selected_source_sha256,
        task_status="active",
    )
    task = _record(
        f"task:{example.example_id}",
        RecordType.TASK_STATE,
        {"task_id": example.example_id, "status": "active", "description": "Answer from selected evidence"},
        version="v1",
        task_status="active",
    )
    tool = _record(
        "tool:evidence_lookup",
        RecordType.TOOL_RECORD,
        {"uri": "tool://evidence_lookup", "schema": {"name": "evidence_lookup", "arguments": ["query"]}},
        version="v1",
        task_status="active",
    )
    result = _record(
        f"result:{example.example_id}",
        RecordType.TOOL_RESPONSE,
        {"uri": f"result://{example.example_id}", "compact": "Frozen selector result"},
        version="v1",
        task_status="active",
    )
    skill = _record(
        "skill:qa",
        RecordType.SKILL_RECORD,
        {"uri": "skill://qa", "instructions": "Answer only from selected evidence."},
        version="v1",
        task_status="active",
    )
    return AgentTurnContext(
        messages=(
            {"role": "system", "content": "Answer the question concisely from the supplied evidence."},
            {"role": "user", "content": example.question},
        ),
        records=(document, task, result),
        tool_records=(tool,),
        skill_records=(skill,),
        tools=({
            "type": "function",
            "function": {
                "name": "evidence_lookup",
                "description": "Return the already selected evidence.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        },),
        task_id=example.example_id,
        task_metadata=dict(task.payload),
        selected_record_ids=(document.record_id, task.record_id, result.record_id, tool.record_id, skill.record_id),
        metadata={
            "condition": condition,
            "example_id": example.example_id,
        },
    )


def _serve(gateway: PRAGateway):
    server = create_gateway_server(gateway, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def run(args) -> dict[str, Any]:
    from mlx_lm import load

    _manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    examples = examples[: args.max_examples]
    model, tokenizer = load(args.model, revision=args.revision)
    adapter = MLXTextModelAdapter(
        model,
        tokenizer,
        model_id=args.model,
        gold_answers={example.example_id: example.answer for example in examples},
    )
    delta_server, delta_thread, delta_endpoint = _serve(PRAGateway(adapter, mode="G10"))
    full_server, full_thread, full_endpoint = _serve(_FullResourceGateway(adapter, mode="G10"))
    rows: list[dict[str, Any]] = []
    try:
        for example in examples:
            for condition, endpoint, transport in (
                ("TEXT", delta_endpoint, "text"),
                ("PRA_FULL", full_endpoint, "pra"),
                ("PRA_DELTA", delta_endpoint, "pra"),
            ):
                backend = NegotiatedRemoteBackend(endpoint, args.model, transport=transport)
                turn = _turn(
                    example,
                    condition=condition,
                    tokenizer=tokenizer,
                    max_source_tokens=args.max_source_tokens,
                )
                before = len(adapter.rows)
                output = backend.generate_turn(
                    turn,
                    tenant_id="benchmark",
                    session_id=f"{condition.lower()}-{example.example_id}",
                    max_new_tokens=args.max_new_tokens,
                )
                model_row = dict(adapter.rows[before])
                trace = dict(backend.inspect()["transport"])
                rows.append({
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "condition": condition,
                    "answer": example.answer,
                    "output": output,
                    **{key: value for key, value in model_row.items() if key not in {"condition", "example_id", "output"}},
                    **trace,
                })
    finally:
        for server, thread in (
            (delta_server, delta_thread),
            (full_server, full_thread),
        ):
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)

    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(str(row["example_id"]), []).append(row)
    parity = {
        example_id: (
            len({row["output"] for row in example_rows}) == 1
            and len({row["messages_sha256"] for row in example_rows}) == 1
        )
        for example_id, example_rows in by_example.items()
    }
    summary = {
        "schema_version": "1.0",
        "experiment": "paper4_5_model_backed_agent_transport_v1",
        "evidence_tier": "NATURAL_QA_MODEL_BACKED_G10",
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "examples": len(examples),
        "conditions": ["TEXT", "PRA_FULL", "PRA_DELTA"],
        "exact_output_and_execution_message_parity": sum(parity.values()),
        "parity_pairs": len(parity),
        "task_status_preserved": all(row["task_status"] == "active" for row in rows),
        "tool_schema_preserved": all(row["tool_schema_preserved"] for row in rows),
        "mean_gold_log_probability": {
            condition: mean(
                float(row["gold_log_probability"])
                for row in rows
                if row["condition"] == condition and row["gold_log_probability"] is not None
            )
            for condition in ("TEXT", "PRA_FULL", "PRA_DELTA")
        },
        "mean_wire_bytes": {
            condition: mean(float(row["wire_bytes"]) for row in rows if row["condition"] == condition)
            for condition in ("TEXT", "PRA_FULL", "PRA_DELTA")
        },
        "mean_ttft_ms": {
            condition: mean(float(row["ttft_ms"]) for row in rows if row["condition"] == condition and row["ttft_ms"] is not None)
            for condition in ("TEXT", "PRA_FULL", "PRA_DELTA")
        },
        "rows": rows,
        "scope": "first-turn G10 representation parity; native G11 consumption is evaluated separately",
    }
    return summary


def write_results(payload: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "agent_transport_model_backed.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows = list(payload["rows"])
    with (output / "agent_transport_model_backed_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "generated_agent_transport_model_results.tex").write_text(
        "\n".join((
            f"\\newcommand{{\\AgentModelExamples}}{{{payload['examples']}}}",
            f"\\newcommand{{\\AgentModelParity}}{{{payload['exact_output_and_execution_message_parity']}/{payload['parity_pairs']}}}",
            f"\\newcommand{{\\AgentModelTextWire}}{{{payload['mean_wire_bytes']['TEXT']:.0f}}}",
            f"\\newcommand{{\\AgentModelFullWire}}{{{payload['mean_wire_bytes']['PRA_FULL']:.0f}}}",
            f"\\newcommand{{\\AgentModelDeltaWire}}{{{payload['mean_wire_bytes']['PRA_DELTA']:.0f}}}",
        )) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest_expanded.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--max-source-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args)
    write_results(payload, args.output)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
