"""Gate the causal-versus-bidirectional representation pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import git_metadata, write_json  # noqa: E402


def run(args):
    natural = json.loads(args.natural_findings.read_text(encoding="utf-8"))
    eligible = bool(natural.get("gate3_statistical_pass", False))
    result = {
        "schema_version": "1.0",
        "git": git_metadata(),
        "gate5_eligible": eligible,
        "run_performed": False,
        "causal_representation": "frozen pretrained decoder-only hidden states",
        "unmasked_pretrained_decoder_run": False,
        "reason": (
            "A separately trained encoder-mode model is unavailable."
            if eligible
            else "Natural paired intervals include zero; the graph method was not frozen for Gate 5."
        ),
    }
    write_json(args.output, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    base = ROOT / "docs/papers/shared/results/paper2_7_query_graph"
    parser.add_argument(
        "--natural-findings",
        type=Path,
        default=base / "natural/natural_findings.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "encoding_mode/encoding_mode_gate.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
