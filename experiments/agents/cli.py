"""Command line for catalog audit, planning, smoke runs, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import FixtureAgentAdapter, command_adapter
from .analysis import summarize
from .catalog import audit_local_agents, load_catalog
from .runner import external_plan, load_runs, run_manifest
from .schema import BenchmarkManifest, PRAProfile, PRAMode


ROOT = Path(__file__).parent


def _manifest(value: str) -> Path:
    path = Path(value)
    return path if path.is_file() else ROOT / "manifests" / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit catalog and local installations.")
    audit.add_argument("--output", type=Path)
    plan = commands.add_parser("plan", help="Emit an official-harness execution plan.")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--agent", required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--condition", default="no-pra")
    plan.add_argument("--output", type=Path)
    run = commands.add_parser("run", help="Run the deterministic harness-validation cohort.")
    run.add_argument("--manifest", default="fixture_smoke.yaml")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--pra-mode", choices=[value.value for value in PRAMode], default="none")
    run.add_argument("--profile", choices=[value.value for value in PRAProfile], default="none")
    smoke = commands.add_parser("smoke-agent", help="Run pinned external CLI fixture tasks in isolated workspaces.")
    smoke.add_argument("--agent", required=True)
    smoke.add_argument("--executable", help="Pinned executable override, normally a project-local install.")
    smoke.add_argument("--manifest", default="fixture_smoke.yaml")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--engine", default="commercial-native")
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--connection", choices=["gateway", "direct", "commercial-native"], default="commercial-native")
    smoke.add_argument("--protocol", default="agent-native")
    smoke.add_argument(
        "--agent-arg", action="append", default=[],
        help="Additional agent argv item; repeat for each item (use --agent-arg=-c for leading dashes).",
    )
    smoke.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", help="Provider environment value; never copied to result artifacts.")
    analyze = commands.add_parser("analyze", help="Summarize normalized JSONL results.")
    analyze.add_argument("results", type=Path)
    analyze.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        value = {
            "catalog": load_catalog().model_dump(mode="json"),
            "local_installations": audit_local_agents(),
        }
    elif args.command == "plan":
        value = external_plan(
            BenchmarkManifest.load(_manifest(args.manifest)), agent=args.agent,
            model=args.model, condition=args.condition,
        )
    elif args.command in {"run", "smoke-agent"}:
        fixture_adapter = (
            FixtureAgentAdapter()
            if args.command == "run"
            else command_adapter(
                args.agent,
                executable=args.executable,
                extra_args=tuple(args.agent_arg),
            )
        )
        if args.command == "smoke-agent":
            environment = {}
            for item in args.env:
                if "=" not in item:
                    raise SystemExit(f"--env requires NAME=VALUE, got {item!r}")
                name, value = item.split("=", 1)
                environment[name] = value
            fixture_adapter.configure_provider(model=args.model, environment=environment)
        rows = run_manifest(
            BenchmarkManifest.load(_manifest(args.manifest)), fixture_adapter,
            output=args.output,
            agent="fixture-agent" if args.command == "run" else args.agent,
            engine="fixture" if args.command == "run" else args.engine,
            model="fixture-model" if args.command == "run" else args.model,
            pra_mode=PRAMode(args.pra_mode) if args.command == "run" else PRAMode.NONE,
            pra_profile=PRAProfile(args.profile) if args.command == "run" else PRAProfile.NONE,
            connection="fixture" if args.command == "run" else args.connection,
            protocol="fixture" if args.command == "run" else args.protocol,
        )
        value = {"runs": len(rows), "output": str(args.output / "runs.jsonl"), "summary": summarize(rows)}
    else:
        value = summarize(load_runs(args.results))
    rendered = json.dumps(value, indent=2)
    if getattr(args, "output", None) and args.command in {"audit", "plan", "analyze"}:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
