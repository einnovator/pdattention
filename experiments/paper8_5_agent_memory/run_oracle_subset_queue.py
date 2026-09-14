"""Execute a bounded oracle pair/beam queue against one frozen decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .materialization import MaterializationMode
from .negative_receipts import NegativeRealizationMode
from .run_frozen_replay import _token_counter, replay


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_queue(
    queue: Mapping[str, Any], trajectory: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    if queue.get("study") != "paper8_5_oracle_headroom_subset_queue":
        raise ValueError("not a Paper 8.5 oracle headroom subset queue")
    if str(queue.get("instance_id")) != str(trajectory.get("instance_id")):
        raise ValueError("queue instance does not match trajectory")
    if queue.get("reference_replay_digest") != _json_digest(reference):
        raise ValueError("queue FULL-reference digest does not match")
    trials = queue.get("trials")
    if not isinstance(trials, list):
        raise ValueError("queue has no trial list")
    expected_depth = int(queue.get("target_depth") or 0)
    seen: set[tuple[str, ...]] = set()
    for trial in trials:
        groups = tuple(str(value) for value in trial.get(
            "omitted_causal_group_ids", ()
        ))
        if len(groups) != expected_depth or len(groups) != len(set(groups)):
            raise ValueError("queue trial has invalid omission depth")
        if groups in seen:
            raise ValueError("queue repeats an omission subset")
        seen.add(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reference-replay", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--tail", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_replay.read_text(encoding="utf-8"))
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    validate_queue(queue, trajectory, reference)
    counter, tokenizer_identity = _token_counter(str(args.tokenizer))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    completed = 0
    for trial in queue["trials"]:
        trial_id = str(trial["trial_id"])
        output = args.output_directory / f"subset_{trial_id}.json"
        result = replay(
            trajectory=trajectory,
            model=args.model,
            base_url=args.base_url,
            policy="full",
            head=args.head,
            tail=args.tail,
            budget_fraction=1.0,
            count_tokens=counter,
            tokenizer_identity=tokenizer_identity,
            materialization_mode=MaterializationMode.WHOLE_RECORD,
            materialization_threshold_tokens=512,
            max_decisions=1,
            seed=args.seed,
            max_output_tokens=args.max_output_tokens,
            api_key=os.environ.get(args.api_key_env),
            timeout=args.timeout,
            negative_realization=NegativeRealizationMode.DROP,
            reference_replay=reference,
            progress_path=output,
            min_decision=int(queue["decision"]),
            oracle_omit_causal_group_ids=trial["omitted_causal_group_ids"],
        )
        completed += int(result.get("completed_decisions") == 1)
        print(json.dumps({
            "trial_id": trial_id,
            "exact_command_rate": result.get("exact_command_rate"),
            "output": str(output),
        }))
    print(json.dumps({"trials": len(queue["trials"]), "completed": completed}))


if __name__ == "__main__":
    main()
