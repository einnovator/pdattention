"""Summarize the completed LoRA sweep and finalize SDK release metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _aggregate_lookup(artifact: dict[str, Any]) -> dict[tuple[str, str, str], dict]:
    return {
        (row["variant"], row["dataset"], row["condition"]): row
        for row in artifact["test_aggregates"]
    }


def paired_vs_frozen(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute five-seed paired delta-logP differences from frozen PRA."""

    lookup = {
        (row["seed"], row["variant"], row["dataset"], row["condition"]): row
        for row in artifact["test_seed_aggregates"]
    }
    variants = list(artifact["stage_c_finalists"])
    variants.extend(
        sorted(
            {
                row["variant"]
                for row in artifact["test_seed_aggregates"]
                if row["variant"].startswith("combo_")
            }
        )
    )
    seeds = tuple(int(seed) for seed in artifact["manifest"]["seeds"])
    output = []
    for variant in variants:
        for dataset in ("hotpotqa", "qasper"):
            for condition in ("oracle", "routed"):
                differences = [
                    lookup[(seed, variant, dataset, condition)][
                        "gold_sequence_logprob_delta_vs_none"
                    ]
                    - lookup[(0, "fixed", dataset, condition)][
                        "gold_sequence_logprob_delta_vs_none"
                    ]
                    for seed in seeds
                ]
                std = statistics.stdev(differences)
                output.append(
                    {
                        "variant": variant,
                        "comparator": "fixed",
                        "dataset": dataset,
                        "condition": condition,
                        "mean_delta_logp_difference": statistics.fmean(differences),
                        "std": std,
                        "ci95": 2.776 * std / math.sqrt(len(differences)),
                        "same_direction": all(value > 0 for value in differences)
                        or all(value < 0 for value in differences),
                        "paired_differences": differences,
                    }
                )
    return output


def _row(lookup, variant: str, dataset: str, condition: str) -> dict:
    return lookup[(variant, dataset, condition)]


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}%"


def _report(artifact: dict[str, Any], paired: list[dict[str, Any]]) -> str:
    winner = artifact["pareto_winner"]
    winner_id = winner["config_id"]
    lookup = _aggregate_lookup(artifact)
    fixed_hr = _row(lookup, "fixed", "hotpotqa", "routed")
    winner_ho = _row(lookup, winner_id, "hotpotqa", "oracle")
    winner_hr = _row(lookup, winner_id, "hotpotqa", "routed")
    winner_qr = _row(lookup, winner_id, "qasper", "routed")
    combo_id = next(
        row["variant"]
        for row in artifact["test_aggregates"]
        if row["variant"].startswith("combo_")
    )
    combo_hr = _row(lookup, combo_id, "hotpotqa", "routed")
    return f"""# Paper 2 Overnight Conditional-LoRA Sweep

## Decision

The validation-only Pareto rule selected `{winner_id}`: rank {winner['rank']},
{winner['steps']} updates ({winner['step_multiplier']:g}x the prior budget), learning rate
{winner['learning_rate']:.0e}, and {winner['memory_use_parameters']:,} trainable parameters
({winner['memory_use_parameter_percent']:.4f}% of Qwen3-0.6B). It is packaged for research use,
but it is **not the SDK default**.

The adapter improves clean oracle integration but is brittle to learned-routing errors. On the
untouched HotpotQA test identities, oracle recovery reaches {_percent(winner_ho['rho_direct_cohort'])}
of direct-text benefit and {_percent(winner_ho['rho_full_cohort'])} of full-context benefit;
oracle F1 is {winner_ho['f1_mean']:.3f}. Routed recovery is
{_percent(winner_hr['rho_direct_cohort'])}, routed delta-logP is
{winner_hr['gold_sequence_logprob_delta_vs_none_mean']:+.3f}, and F1 is
{winner_hr['f1_mean']:.3f}. Frozen PRA remains positive at
{_percent(fixed_hr['rho_direct_cohort'])} routed direct recovery.

QASPER routed delta-logP rises to {winner_qr['gold_sequence_logprob_delta_vs_none_mean']:+.3f}
and F1 to {winner_qr['f1_mean']:.3f}, but its direct-text denominator is negative, so no recovery
ratio is reported. EM is zero for every routed finalist.

Adding residual-32 does not resolve the mismatch. Its HotpotQA routed delta-logP is
{combo_hr['gold_sequence_logprob_delta_vs_none_mean']:+.3f}; paired combo-minus-LoRA effects
change direction across seeds for both datasets and both selection modes.

## Questions Answered

- **Did longer training help?** Only selectively. Rank 32 improves from the 32-update screen to
  64 updates, then slips at 128. Rank 4 and rank 8 become worse at 64 updates under the baseline
  learning rate. The prior result was not uniformly under-converged.
- **Did larger rank help?** Yes for oracle integration: five-seed validation rises from rank 16
  ({artifact['finalist_validation_ranking'][1]['combined_oracle_delta_logp']:.3f}) to rank 32
  ({winner['combined_oracle_delta_logp']:.3f}). It does not help routed HotpotQA.
- **Where did performance saturate?** The tested oracle frontier peaks at rank 32 and 64 updates;
  128 updates regress. Rank 64 was not expanded because the rank-32 routed safety result already
  failed the product gate, so saturation beyond rank 32 is not claimed.
- **Gold rank and F1?** Oracle HotpotQA mean gold rank improves to
  {winner_ho['gold_first_token_rank_mean']:.2f} and F1 to {winner_ho['f1_mean']:.3f}; routed rank
  worsens to {winner_hr['gold_first_token_rank_mean']:.2f} and F1 remains near zero.
- **PRA-off exactness?** Yes: all {artifact['exactness']['candidate_checks']} candidate checks are
  exact and native-limit violations are zero.
- **SDK default?** Keep frozen PRA plus the learned router. The packaged LoRA is opt-in for
  oracle/controlled studies, not general routed inference.

## Protocol

- Train: 12 HotpotQA identities, offsets 0--11.
- Validation: four identities per dataset at offset 12; screening and finalist selection use no
  test identities.
- Test: eight identities per dataset at offset 16, loaded only after finalist selection.
- Stage A: ranks 4/8/16/32, 32 and 64 updates, baseline learning rate.
- Stage B: best three ranks, 64/128 updates and 0.5x/1x/2x learning rate.
- Stage C: three rank-diverse finalists over seeds 11, 23, 37, 53, and 71.
- Combination: the selected rank-32 LoRA plus residual-32, once, over the same five seeds.
- Full-context greedy decoding is omitted because 2,048-token eager generation exceeds the
  4-GiB evaluation GPU. Full-context teacher-forced logP and recovery remain measured.

## Files

- `overnight_lora_manifest.json`: predeclared grid and separation rules.
- `validation_ranking.csv`: complete Stage-A/B screen.
- `finalist_validation.csv`: five-seed held-out finalists.
- `test_finalists.csv` and `test_five_seed.csv`: aggregate and per-seed test metrics.
- `paired_vs_frozen.csv` and `combo_paired.csv`: paired effects.
- `recovery_ratios.csv`: direct/full recovery where denominators are valid.
- `pra_off_exactness.csv`: hard retrofit gate.
- `lora_parameter_pareto.pdf`: validation quality against trainable parameter fraction.
- `overnight_lora_sweep.json`: complete raw rows and provenance.
"""


def validation_grid_tex(artifact: dict[str, Any]) -> str:
    """Render the complete Stage-A/B screen as a paper-ready LaTeX table."""

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Complete single-seed Stage-A/B conditional-LoRA screen, ranked by the equal-weight held-out oracle score. H and Q denote HotpotQA and QASPER. Stage C repeats three rank-diverse finalists over five seeds.}",
        r"\label{tab:overnight-lora-grid}",
        r"\begin{tabular}{crrrrrr}",
        r"\toprule",
        r"Stage & Rank & Updates & LR & H $\Delta\log p$ & Q $\Delta\log p$ & Mean \\",
        r"\midrule",
    ]
    rows = artifact.get("screen_ranking", artifact.get("validation_ranking"))
    if rows is None:
        raise KeyError("Artifact has no Stage-A/B screen ranking.")
    for row in rows:
        lines.append(
            f"{row['stage']} & {int(row['rank'])} & {int(row['steps'])} & "
            f"{float(row['learning_rate']):.1e} & "
            f"{float(row['hotpotqa_oracle_delta_logp']):+.3f} & "
            f"{float(row['qasper_oracle_delta_logp']):+.3f} & "
            f"{float(row['combined_oracle_delta_logp']):+.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [
        {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_sdk_metadata(
    artifact: dict[str, Any], router_dir: Path, memory_dir: Path
) -> None:
    """Mark the measured adapter opt-in and connect its compatible router."""

    winner = artifact["pareto_winner"]
    router_config_path = router_dir / "config.json"
    router = json.loads(router_config_path.read_text(encoding="utf-8"))
    router.update(
        {
            "compatible_memory_adapter": str(memory_dir),
            "memory_adapter_default": False,
            "recommended_memory_use": "frozen_pra",
            "overnight_sweep_artifact": (
                "docs/papers/shared/results/paper2_hf/overnight_lora_sweep/"
                "overnight_lora_sweep.json"
            ),
        }
    )
    router_config_path.write_text(
        json.dumps(router, indent=2, sort_keys=True), encoding="utf-8"
    )

    memory_config_path = memory_dir / "memory_adapter_config.json"
    memory = json.loads(memory_config_path.read_text(encoding="utf-8"))
    memory.update(
        {
            "release_status": "research_opt_in",
            "sdk_default": False,
            "recommended_default": "frozen_pra_plus_router",
            "selection_metric": (
                "equal-weight HotpotQA/QASPER validation oracle sequence delta-logP"
            ),
            "validation_oracle_delta_logp": winner["combined_oracle_delta_logp"],
            "validation_routed_delta_logp": winner["combined_routed_delta_logp"],
            "known_limitation": (
                "Improves oracle integration but degrades routed HotpotQA; use only for "
                "controlled research until routing-conditioned adaptation is validated."
            ),
            "compatible_router_config_sha256": _sha256(router_config_path),
        }
    )
    memory_config_path.write_text(
        json.dumps(memory, indent=2, sort_keys=True), encoding="utf-8"
    )
    (memory_dir / "README.md").write_text(
        f"""# PRA conditional memory-use adapter

**Status: research opt-in. This is not the SDK default.**

- Base model: `Qwen/Qwen3-0.6B`
- Type: conditional output-projection LoRA
- Rank / alpha: {winner['rank']} / {winner['rank']}
- PRA depth: last 14 layers (14--27)
- Parameters: {winner['memory_use_parameters']:,} ({winner['memory_use_parameter_percent']:.4f}% of base)
- Selected seed: {artifact['selected_artifact_seed']}

The adapter improves oracle memory integration but degrades learned-routing HotpotQA. Keep
frozen PRA plus the compatible router as the product default. This artifact is provided for
controlled oracle and adaptation studies. PRA-off execution structurally bypasses these weights.
See the sweep result directory for five-seed metrics and the full validation grid.
""",
        encoding="utf-8",
    )
    (router_dir / "README.md").write_text(
        """# PRA routing adapter

This is the learned Qwen3-0.6B router used by the Paper 2 last-14 LoRA sweep. It is compatible
with the public `pra_hf` router artifact API. The recommended default is frozen PRA plus this
router. The linked conditional memory-use adapter is a research-only opt-in because it improves
oracle integration but degrades routed HotpotQA.
""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    paired = paired_vs_frozen(artifact)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "paired_vs_frozen.csv", paired)
    (args.output_dir / "README.md").write_text(
        _report(artifact, paired), encoding="utf-8"
    )
    (args.output_dir / "validation_grid.tex").write_text(
        validation_grid_tex(artifact), encoding="utf-8"
    )
    finalize_sdk_metadata(artifact, args.router_dir, args.memory_adapter_dir)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    results = root / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=results / "overnight_lora_sweep" / "overnight_lora_sweep.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=results / "overnight_lora_sweep"
    )
    parser.add_argument(
        "--router-dir",
        type=Path,
        default=root / "artifacts" / "pra_hf" / "routers" / "qwen3-0.6b-joint-d128",
    )
    parser.add_argument(
        "--memory-adapter-dir",
        type=Path,
        default=root / "artifacts" / "pra_hf" / "memory_adapters" / "qwen3-0.6b-last14-lora",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
