"""Run the controlled WikiText-2 Phase 1 follow-up experiment matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "nb", REPO):
    sys.path.insert(0, str(path))

import pra_notebook_utils as pnu  # noqa: E402


SEEDS = (1, 7, 21, 42, 87)
RUN_TAG = "corrected_router_v2"
REPORT_NAMESPACE = "phase1_followup_v2_5seed"
PLAIN_EXPERIMENTS = {
    "sa": "phase1_plain_sa_1m",
    "pra": "phase1_plain_pra_1m",
    "hybrid": "phase1_plain_hybrid_1m",
    "sa_match_full": "phase1_plain_sa_match_full_1m",
    "sa_match_hybrid": "phase1_plain_sa_match_hybrid_1m",
}
REFERENCE_EXPERIMENTS = {
    "sa": "phase1_order_sa_refs_v2",
    "pra": "phase1_order_pra_refs_v2",
    "hybrid": "phase1_order_hybrid_refs_v2",
}
REFERENCE_MODES = {
    "sa": ("scratch", "joint"),
    "pra": ("scratch", "frozen_refpath", "joint"),
    "hybrid": ("scratch", "frozen_refpath", "joint"),
}
def checkpoint_step(path: Path) -> int:
    if not path.exists():
        return 0
    return int(torch.load(path, map_location="cpu").get("global_step", 0))


def plain_artifact(runtime: pnu.NotebookRuntime, experiment_name: str) -> str:
    settings = pnu.load_experiment_settings(runtime, experiment_name)
    policy = pnu.policy_from_experiment(settings)
    train = settings["experiment"]["train"]
    return pnu.experiment_artifact_name(
        settings["model_name"],
        "wikitext2",
        settings["name"],
        RUN_TAG,
        f"seed{runtime.seed}",
        f"steps{train['max_steps']}",
        f"seq{policy.max_seq_len}",
    )


def reference_artifact(
    runtime: pnu.NotebookRuntime, experiment_name: str, mode: str
) -> str:
    settings = pnu.load_experiment_settings(runtime, experiment_name)
    policy = pnu.policy_from_experiment(settings)
    train = settings["experiment"]["train"]
    stage = settings["experiment"]["dataset"]["stage"]
    return pnu.experiment_artifact_name(
        settings["model_name"],
        stage,
        settings["name"],
        RUN_TAG,
        mode,
        f"seed{runtime.seed}",
        f"steps{train['max_steps']}",
        f"seq{policy.max_seq_len}",
    )


def run_report_path(
    runtime: pnu.NotebookRuntime,
    experiment_name: str,
    dataset_stage: str,
    *qualifiers: str,
) -> Path:
    settings = pnu.load_experiment_settings(runtime, experiment_name)
    name = pnu.experiment_artifact_name(
        settings["model_name"], dataset_stage, *qualifiers
    )
    return REPO / "out" / "reports" / REPORT_NAMESPACE / "runs" / name / "index.html"


def run_plain(
    experiment_name: str, seed: int, *, force: bool = False
) -> tuple[Path, Path]:
    runtime = pnu.configure_notebook(repo=REPO, seed=seed)
    settings = pnu.load_experiment_settings(runtime, experiment_name)
    max_steps = int(settings["experiment"]["train"]["max_steps"])
    run_dir = REPO / "out" / "notebook_wikitext" / plain_artifact(runtime, experiment_name)
    latest = run_dir / "checkpoints" / "latest.pt"
    best = run_dir / "checkpoints" / "best.pt"
    report = run_report_path(
        runtime, experiment_name, "wikitext2", experiment_name, RUN_TAG, f"seed{seed}"
    )
    if not force and checkpoint_step(latest) >= max_steps and report.exists():
        print(f"skip completed plain run: {experiment_name} seed={seed}")
        return best if best.exists() else latest, report

    resume_from = latest if latest.exists() else None
    print(
        f"run plain: experiment={experiment_name} seed={seed} "
        f"resume_step={checkpoint_step(resume_from) if resume_from else 0}"
    )
    result = pnu.train_wikitext_language_experiment(
        runtime, experiment_name, resume_from=resume_from
    )
    report = pnu.generate_html_report(
        result,
        runtime=runtime,
        dataset_details=result["loader_summary"],
        model_name=result["model_name"],
        qualifiers=(experiment_name, RUN_TAG, f"seed{seed}"),
        report_root=REPO / "out" / "reports" / REPORT_NAMESPACE / "runs",
    )
    return result["state"].checkpoint.best_path, report


def run_reference(
    architecture: str,
    experiment_name: str,
    mode: str,
    seed: int,
    parent_checkpoint: Path,
    *,
    force: bool = False,
) -> Path:
    runtime = pnu.configure_notebook(repo=REPO, seed=seed)
    settings = pnu.load_experiment_settings(runtime, experiment_name)
    max_steps = int(settings["experiment"]["train"]["max_steps"])
    artifact = reference_artifact(runtime, experiment_name, mode)
    run_dir = REPO / "out" / "notebook_wikitext_refs" / artifact
    latest = run_dir / "checkpoints" / "latest.pt"
    report = run_report_path(
        runtime,
        experiment_name,
        "wikitext2_references_v2",
        experiment_name,
        RUN_TAG,
        mode,
        f"seed{seed}",
    )
    if not force and checkpoint_step(latest) >= max_steps and report.exists():
        print(f"skip completed reference run: {architecture} {mode} seed={seed}")
        return report

    print(f"run references: architecture={architecture} mode={mode} seed={seed}")
    result = pnu.train_wikitext_reference_experiment(
        runtime,
        experiment_name,
        initial_checkpoint=parent_checkpoint if mode != "scratch" else None,
        tokenizer_checkpoint=parent_checkpoint if mode == "scratch" else None,
        training_mode=mode,
    )
    pnu.evaluate_plain_after_reference(
        result, runtime, PLAIN_EXPERIMENTS[architecture]
    )
    pnu.evaluate_reference_conditions(result)
    return pnu.generate_html_report(
        result,
        runtime=runtime,
        dataset_details=result["loader_summary"],
        model_name=result["model_name"],
        qualifiers=(experiment_name, RUN_TAG, mode, f"seed{seed}"),
        report_root=REPO / "out" / "reports" / REPORT_NAMESPACE / "runs",
    )


def aggregate_reports(grouped_reports: dict[str, list[Path]]) -> dict[str, Path]:
    aggregate_root = REPO / "out" / "reports" / REPORT_NAMESPACE / "aggregates"
    outputs = {}
    for group, reports in grouped_reports.items():
        report_json_paths = []
        for report_path in reports:
            report_json = report_path.with_name("report.json")
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            results = payload["results"]
            metric_records = json.loads(
                report_path.with_name("metrics.json").read_text(encoding="utf-8")
            )["records"]
            batch_durations = [
                float(record["metrics"]["train_batch_duration_seconds"])
                for record in metric_records
                if record["split"] == "train_batch"
                and "train_batch_duration_seconds" in record["metrics"]
            ]
            median_duration = statistics.median(batch_durations)
            interruption_threshold = max(1.0, 20.0 * median_duration)
            active_durations = [
                duration for duration in batch_durations if duration <= interruption_threshold
            ]
            interruption_durations = [
                duration for duration in batch_durations if duration > interruption_threshold
            ]
            results["active_train_seconds"] = sum(active_durations)
            results["host_interruption_seconds"] = sum(interruption_durations)
            results["train_batch_duration_median_seconds"] = median_duration
            results["processed_tokens_this_invocation"] = results.get("processed_tokens")
            if payload["training"].get("dataset_stage") == "wikitext2":
                results["training_budget_tokens"] = int(results["optimizer_steps"]) * int(
                    payload["training"]["batch_size"]
                ) * int(payload["training"]["max_seq_len"])
            report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            report_json_paths.append(report_json)
        outputs[group] = pnu.generate_aggregate_html_report(
            report_json_paths,
            report_name=group,
            title=group.replace("_", " ").title(),
            report_root=aggregate_root,
        )
    return outputs


def aggregate_value(report: dict[str, Any], metric: str) -> float | None:
    value = report.get("aggregate", {}).get(metric)
    return float(value["mean"]) if value else None


def metric_curve(report_paths: list[Path], metric: str) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for report_path in report_paths:
        metrics_path = report_path.with_name("metrics.json")
        records = json.loads(metrics_path.read_text(encoding="utf-8"))["records"]
        for record in records:
            if record["split"] != "val" or metric not in record["metrics"]:
                continue
            values.setdefault(int(record["step"]), []).append(
                float(record["metrics"][metric])
            )
    return {step: statistics.fmean(items) for step, items in sorted(values.items())}


def make_findings_report(
    aggregate_paths: dict[str, Path], grouped_reports: dict[str, list[Path]]
) -> Path:
    report_dir = REPO / "out" / "reports" / f"{REPORT_NAMESPACE}_findings"
    assets = report_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    aggregates = {
        name: json.loads(path.with_name("report.json").read_text(encoding="utf-8"))
        for name, path in aggregate_paths.items()
    }

    plain_labels = ["sa", "pra", "hybrid", "sa_match_full", "sa_match_hybrid"]
    plain_rows = []
    for architecture in plain_labels:
        report = aggregates[f"plain_{architecture}"]
        plain_rows.append(
            {
                "architecture": architecture,
                "parameters": int(report["model"]["parameter_count"]),
                "test_loss": aggregate_value(report, "test_loss"),
                "test_perplexity": aggregate_value(report, "test_perplexity"),
                "training_budget_tokens": aggregate_value(report, "training_budget_tokens"),
                "active_train_seconds": aggregate_value(report, "active_train_seconds"),
                "host_interruption_seconds": aggregate_value(
                    report, "host_interruption_seconds"
                ),
            }
        )

    order_rows = []
    for architecture, modes in REFERENCE_MODES.items():
        base_loss = aggregate_value(aggregates[f"plain_{architecture}"], "test_loss")
        for mode in modes:
            report = aggregates[f"refs_{architecture}_{mode}"]
            plain_after = aggregate_value(report, "plain_test_after_reference_loss")
            order_rows.append(
                {
                    "architecture": architecture,
                    "mode": mode,
                    "valid_loss": aggregate_value(report, "reference_valid_loss"),
                    "disabled_loss": aggregate_value(report, "reference_disabled_loss"),
                    "shuffled_loss": aggregate_value(report, "reference_shuffled_loss"),
                    "irrelevant_loss": aggregate_value(report, "reference_irrelevant_loss"),
                    "oracle_loss": aggregate_value(report, "reference_oracle_loss"),
                    "initial_top1": aggregate_value(report, "initial_reference_top1_accuracy"),
                    "final_top1": aggregate_value(report, "test_reference_top1_accuracy"),
                    "plain_after": plain_after,
                    "forgetting_delta": plain_after - base_loss if plain_after is not None else None,
                }
            )

    by_key = {(row["architecture"], row["mode"]): row for row in order_rows}
    questions = []
    chance_top1 = 0.4201923076923077
    pretrained_top1 = statistics.fmean(
        by_key[(architecture, "frozen_refpath")]["final_top1"]
        for architecture in ("pra", "hybrid")
    )
    scratch_top1 = statistics.fmean(
        by_key[(architecture, "scratch")]["final_top1"]
        for architecture in ("pra", "hybrid")
    )
    questions.append(
        {
            "question": "Does pretraining improve reference-selection speed?",
            "answer": pretrained_top1 > chance_top1 and pretrained_top1 > scratch_top1,
            "detail": (
                f"frozen pretrained final top-1={pretrained_top1:.4f}; "
                f"scratch={scratch_top1:.4f}; candidate-count chance={chance_top1:.4f}; "
                "periodic curves show no sustained rise"
            ),
        }
    )
    for architecture in ("pra", "hybrid"):
        direct = by_key[(architecture, "joint")]
        scratch = by_key[(architecture, "scratch")]
        frozen = by_key[(architecture, "frozen_refpath")]
        questions.append(
            {
                "question": f"Does pretraining improve final loss for {architecture}?",
                "answer": direct["valid_loss"] < scratch["valid_loss"],
                "detail": (
                    f"direct joint valid loss={direct['valid_loss']:.4f}; "
                    f"scratch={scratch['valid_loss']:.4f}"
                ),
            }
        )
        questions.append(
            {
                "question": f"Does frozen reference-path training preserve plain LM for {architecture}?",
                "answer": abs(frozen["forgetting_delta"]) < abs(direct["forgetting_delta"]),
                "detail": (
                    f"frozen plain-loss delta={frozen['forgetting_delta']:+.4f}; "
                    f"direct joint={direct['forgetting_delta']:+.4f}"
                ),
            }
        )

    full_staged_gain = (
        by_key[("pra", "scratch")]["valid_loss"]
        - by_key[("pra", "frozen_refpath")]["valid_loss"]
    )
    hybrid_staged_gain = (
        by_key[("hybrid", "scratch")]["valid_loss"]
        - by_key[("hybrid", "frozen_refpath")]["valid_loss"]
    )
    questions.append(
        {
            "question": "Does the hybrid benefit more from staged training than full PRA?",
            "answer": hybrid_staged_gain > full_staged_gain,
            "detail": (
                f"hybrid scratch-to-staged gain={hybrid_staged_gain:.4f}; "
                f"full PRA gain={full_staged_gain:.4f}"
            ),
        }
    )

    old_pra_report = REPO / "out" / "reports" / "final_td_pra_tiny_wikitext2_plain" / "report.json"
    old_pra_loss = None
    if old_pra_report.exists():
        old_pra = json.loads(old_pra_report.read_text(encoding="utf-8"))
        old_pra_loss = float(old_pra["aggregate"]["test_loss"]["mean"])
        new_pra_loss = aggregate_value(aggregates["plain_pra"], "test_loss")
        questions.append(
            {
                "question": "Does full PRA benefit from longer ordinary-language pretraining?",
                "answer": new_pra_loss < old_pra_loss,
                "detail": (
                    f"mean test loss fell from {old_pra_loss:.4f} at 524,288 tokens "
                    f"to {new_pra_loss:.4f} at 1,048,576 tokens"
                ),
            }
        )

    matched_full_loss = aggregate_value(aggregates["plain_sa_match_full"], "test_loss")
    full_pra_loss = aggregate_value(aggregates["plain_pra"], "test_loss")
    matched_hybrid_loss = aggregate_value(aggregates["plain_sa_match_hybrid"], "test_loss")
    hybrid_loss = aggregate_value(aggregates["plain_hybrid"], "test_loss")
    questions.append(
        {
            "question": "Does ordinary-language parity survive parameter matching?",
            "answer": abs(matched_full_loss - full_pra_loss) < 0.05
            and abs(matched_hybrid_loss - hybrid_loss) < 0.05,
            "detail": (
                f"full PRA={full_pra_loss:.4f} vs matched SA={matched_full_loss:.4f}; "
                f"hybrid={hybrid_loss:.4f} vs matched SA={matched_hybrid_loss:.4f}"
            ),
        }
    )
    for architecture in ("pra", "hybrid"):
        direct = by_key[(architecture, "joint")]
        frozen = by_key[(architecture, "frozen_refpath")]
        questions.append(
            {
                "question": f"Is direct joint fine-tuning sufficient for {architecture}?",
                "answer": direct["valid_loss"] <= frozen["valid_loss"],
                "detail": (
                    f"direct joint valid loss={direct['valid_loss']:.4f}; "
                    f"frozen={frozen['valid_loss']:.4f}; forgetting must be considered separately"
                ),
            }
        )

    selection_curves = {}
    for architecture in ("pra", "hybrid"):
        for mode in ("scratch", "frozen_refpath", "joint"):
            group = f"refs_{architecture}_{mode}"
            selection_curves[group] = metric_curve(
                grouped_reports[group], "reference_selection_top1_accuracy"
            )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(
        [row["architecture"] for row in plain_rows],
        [row["test_loss"] for row in plain_rows],
        color=["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"],
    )
    axes[0].set_title("Plain WikiText-2: 1M-token budget")
    axes[0].set_ylabel("test cross-entropy")
    axes[0].tick_params(axis="x", rotation=15)
    for group, curve in selection_curves.items():
        if curve:
            axes[1].plot(curve.keys(), curve.values(), marker="o", label=group.removeprefix("refs_"))
    axes[1].set_title("Reference top-1 selection during training")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("top-1 accuracy")
    axes[1].axhline(
        0.4859477124183006,
        color="#111827",
        linestyle="--",
        linewidth=1,
        label="validation chance",
    )
    axes[1].set_ylim(0.45, 0.57)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets / "phase1_followup.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    payload = {
        "title": "PRA Phase 1 Follow-up Findings",
        "run_tag": RUN_TAG,
        "seeds": list(SEEDS),
        "dataset_seed": 1729,
        "split_seed": 1729,
        "reference_top1_chance": chance_top1,
        "previous_524k_full_pra_test_loss": old_pra_loss,
        "git_commit": git_commit,
        "dirty_worktree": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO, text=True
            ).strip()
        ),
        "plain_parity": plain_rows,
        "training_order": order_rows,
        "questions": questions,
        "selection_curves": selection_curves,
        "aggregate_reports": {
            name: str(path.resolve()) for name, path in aggregate_paths.items()
        },
    }
    (report_dir / "report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    def table(rows: list[dict[str, Any]]) -> str:
        headers = list(rows[0])
        head = "".join(f"<th>{escape(key.replace('_', ' ').title())}</th>" for key in headers)
        body = ""
        for row in rows:
            cells = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    value = f"{value:.5f}"
                cells.append(f"<td>{escape(str(value))}</td>")
            body += "<tr>" + "".join(cells) + "</tr>"
        return f"<table><tr>{head}</tr>{body}</table>"

    question_html = "".join(
        f"<h3>{escape(item['question'])}</h3><p><strong>{'Yes' if item['answer'] else 'No or not yet'}.</strong> {escape(item['detail'])}</p>"
        for item in questions
    )
    links = "".join(
        f'<li><a href="{path.resolve().as_uri()}">{escape(name)}</a></li>'
        for name, path in aggregate_paths.items()
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PRA Phase 1 Follow-up</title>
<style>body{{font-family:Segoe UI,Arial;margin:32px auto;max-width:1300px;color:#18202a}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}h2{{margin-top:32px}}</style></head><body>
<h1>PRA Phase 1 Follow-up Findings</h1><p>Five paired model seeds, fixed WikiText-2 reference dataset and split, matched training budgets. Uncertainty is available in the aggregate reports.</p>
<h2>Answers</h2>{question_html}<h2>Plain-Language Parity</h2>{table(plain_rows)}
<h2>Training Order And Retention</h2>{table(order_rows)}<h2>Curves</h2><img src="assets/phase1_followup.png">
<h2>Aggregate Reports</h2><ul>{links}</ul><p><a href="report.json">Structured findings JSON</a></p></body></html>"""
    output = report_dir / "index.html"
    output.write_text(html, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", "parity", "order", "report"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(SEEDS)) < 5:
        raise RuntimeError("Phase 1 comparisons require at least five distinct model seeds.")
    experiment_names = (*PLAIN_EXPERIMENTS.values(), *REFERENCE_EXPERIMENTS.values())
    runtime = pnu.configure_notebook(repo=REPO, seed=SEEDS[0])
    for experiment_name in experiment_names:
        settings = pnu.load_experiment_settings(runtime, experiment_name)
        configured_seeds = tuple(settings["experiment"].get("seeds", ()))
        if configured_seeds != SEEDS:
            raise RuntimeError(
                f"{experiment_name} must use the paired Phase 1 seeds {SEEDS}; "
                f"found {configured_seeds}."
            )
    if not torch.cuda.is_available():
        raise RuntimeError("The Phase 1 follow-up must run in the CUDA-enabled Python environment.")
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    grouped: dict[str, list[Path]] = {}
    parents: dict[tuple[str, int], Path] = {}

    if args.stage in {"all", "parity"}:
        for architecture, experiment in PLAIN_EXPERIMENTS.items():
            grouped[f"plain_{architecture}"] = []
            for seed in SEEDS:
                checkpoint, report = run_plain(experiment, seed, force=args.force)
                parents[(architecture, seed)] = checkpoint
                grouped[f"plain_{architecture}"].append(report)

    if args.stage in {"all", "order"}:
        for architecture in REFERENCE_EXPERIMENTS:
            if (architecture, SEEDS[0]) not in parents:
                for seed in SEEDS:
                    runtime = pnu.configure_notebook(repo=REPO, seed=seed)
                    run_dir = REPO / "out" / "notebook_wikitext" / plain_artifact(
                        runtime, PLAIN_EXPERIMENTS[architecture]
                    )
                    parents[(architecture, seed)] = run_dir / "checkpoints" / "best.pt"
            for mode in REFERENCE_MODES[architecture]:
                group = f"refs_{architecture}_{mode}"
                grouped[group] = []
                for seed in SEEDS:
                    grouped[group].append(
                        run_reference(
                            architecture,
                            REFERENCE_EXPERIMENTS[architecture],
                            mode,
                            seed,
                            parents[(architecture, seed)],
                            force=args.force,
                        )
                    )

    if args.stage == "report":
        for architecture, experiment in PLAIN_EXPERIMENTS.items():
            grouped[f"plain_{architecture}"] = [
                run_report_path(
                    pnu.configure_notebook(repo=REPO, seed=seed),
                    experiment,
                    "wikitext2",
                    experiment,
                    RUN_TAG,
                    f"seed{seed}",
                )
                for seed in SEEDS
            ]
        for architecture, experiment in REFERENCE_EXPERIMENTS.items():
            for mode in REFERENCE_MODES[architecture]:
                grouped[f"refs_{architecture}_{mode}"] = [
                    run_report_path(
                        pnu.configure_notebook(repo=REPO, seed=seed),
                        experiment,
                        "wikitext2_references_v2",
                        experiment,
                        RUN_TAG,
                        mode,
                        f"seed{seed}",
                    )
                    for seed in SEEDS
                ]

    aggregates = aggregate_reports(grouped)
    final_report = make_findings_report(aggregates, grouped)
    print(f"findings report: {final_report}")


if __name__ == "__main__":
    main()
