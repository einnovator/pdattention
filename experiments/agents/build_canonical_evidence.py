"""Build canonical three-condition evidence from normalized agent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import baseline_promotion_gate, canonical_agent_evidence
from .runner import load_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, nargs="+", help="No-PRA runs.jsonl files.")
    parser.add_argument("--pra-no-adaptor", type=Path, nargs="*", default=())
    parser.add_argument("--pra-adaptor-bundle", type=Path, nargs="*", default=())
    parser.add_argument("--bundle-id")
    parser.add_argument("--bundle-revision")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--mode", default="agent-gateway")
    parser.add_argument("--date", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    baseline = [row for path in args.baseline for row in load_runs(path)]
    no_adaptor = [row for path in args.pra_no_adaptor for row in load_runs(path)]
    bundle = [row for path in args.pra_adaptor_bundle for row in load_runs(path)]
    record = canonical_agent_evidence(
        baseline,
        pra_no_adaptor=no_adaptor,
        pra_adaptor_bundle=bundle,
        profile=args.profile,
        mode=args.mode,
        bundle_id=args.bundle_id,
        bundle_revision=args.bundle_revision,
        date=args.date,
        commit=args.commit,
    )
    payload = record.serialize_for_control_plane()
    payload["agent_admission_gate"] = baseline_promotion_gate(baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": payload["agent_admission_gate"]}, indent=2))


if __name__ == "__main__":
    main()
