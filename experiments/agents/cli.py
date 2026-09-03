"""Command line for catalog audit, planning, smoke runs, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import FixtureAgentAdapter, command_adapter
from .analysis import baseline_promotion_gate, summarize
from .catalog import audit_local_agents, load_catalog
from .runner import external_plan, import_harbor_job, load_runs, run_manifest
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
    analyze.add_argument("results", type=Path, nargs="+")
    analyze.add_argument("--output", type=Path)
    screen = commands.add_parser("screen", help="Apply the no-PRA admission gate before a PRA comparison.")
    screen.add_argument("results", type=Path, nargs="+")
    screen.add_argument("--minimum-success-rate", type=float, default=0.30)
    screen.add_argument("--maximum-success-rate", type=float, default=0.80)
    screen.add_argument("--minimum-runs", type=int, default=3)
    screen.add_argument("--output", type=Path)
    harbor = commands.add_parser("import-harbor", help="Normalize an official Harbor job without re-grading it.")
    harbor.add_argument("job", type=Path)
    harbor.add_argument("--manifest", required=True)
    harbor.add_argument("--output", type=Path, required=True)
    harbor.add_argument("--engine", required=True)
    harbor.add_argument("--engine-version")
    harbor.add_argument("--host", required=True)
    harbor.add_argument(
        "--hardware", action="append", default=[], metavar="NAME=JSON_VALUE",
        help="Hardware fact stored with each row; repeat for multiple facts.",
    )
    harbor.add_argument("--model", required=True)
    harbor.add_argument("--model-revision")
    harbor.add_argument("--quantization")
    harbor.add_argument("--connection", choices=["gateway", "direct", "commercial-native", "fixture"], required=True)
    harbor.add_argument("--protocol", required=True)
    harbor.add_argument("--pra-mode", choices=[value.value for value in PRAMode], default="none")
    harbor.add_argument("--profile", choices=[value.value for value in PRAProfile], default="none")
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
    elif args.command == "import-harbor":
        hardware = {}
        for item in args.hardware:
            if "=" not in item:
                raise SystemExit(f"--hardware requires NAME=JSON_VALUE, got {item!r}")
            name, raw = item.split("=", 1)
            try:
                hardware[name] = json.loads(raw)
            except json.JSONDecodeError:
                hardware[name] = raw
        rows = import_harbor_job(
            args.job, BenchmarkManifest.load(_manifest(args.manifest)), output=args.output,
            engine=args.engine, engine_version=args.engine_version,
            host=args.host, hardware=hardware, model=args.model,
            model_revision=args.model_revision, quantization=args.quantization,
            pra_mode=PRAMode(args.pra_mode), pra_profile=PRAProfile(args.profile),
            connection=args.connection, protocol=args.protocol,
        )
        value = {"runs": len(rows), "output": str(args.output / "runs.jsonl"), "summary": summarize(rows)}
    elif args.command == "screen":
        rows = [row for path in args.results for row in load_runs(path)]
        value = {
            "summary": summarize(rows),
            "admission_gate": baseline_promotion_gate(
                rows,
                minimum_success_rate=args.minimum_success_rate,
                maximum_success_rate=args.maximum_success_rate,
                minimum_runs=args.minimum_runs,
            ),
        }
    else:
        value = summarize(
            row for path in args.results for row in load_runs(path)
        )
    rendered = json.dumps(value, indent=2)
    if getattr(args, "output", None) and args.command in {"audit", "plan", "analyze", "screen"}:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
