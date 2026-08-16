"""Pure analysis helpers for the Paper 5 scaling study.

The helpers intentionally avoid model-specific dependencies.  Experiment rows
can therefore distinguish measured routing data, inherited model results, and
analytical baselines while sharing one fit and Pareto-frontier implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FitResult:
    """One candidate fit and diagnostics over the observed x/y points."""

    family: str
    parameters: dict[str, float]
    predictions: tuple[float, ...]
    residuals: tuple[float, ...]
    rmse: float
    r_squared: float
    aic: float
    n: int
    parameter_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "parameters": self.parameters,
            "predictions": list(self.predictions),
            "residuals": list(self.residuals),
            "rmse": self.rmse,
            "r_squared": self.r_squared,
            "aic": self.aic,
            "n": self.n,
            "parameter_count": self.parameter_count,
        }


def _finish_fit(
    family: str,
    parameters: Mapping[str, float],
    observed: np.ndarray,
    predicted: np.ndarray,
) -> FitResult:
    residuals = observed - predicted
    rss = float(np.square(residuals).sum())
    tss = float(np.square(observed - observed.mean()).sum())
    n = len(observed)
    parameter_count = len(parameters)
    rmse = math.sqrt(rss / max(n, 1))
    r_squared = 1.0 - rss / tss if tss > 0 else float(rss <= 1e-18)
    aic = n * math.log(max(rss / max(n, 1), 1e-18)) + 2 * parameter_count
    return FitResult(
        family,
        {key: float(value) for key, value in parameters.items()},
        tuple(float(value) for value in predicted),
        tuple(float(value) for value in residuals),
        rmse,
        r_squared,
        aic,
        n,
        parameter_count,
    )


def fit_candidate_laws(x: Sequence[float], y: Sequence[float]) -> list[FitResult]:
    """Fit constant, logarithmic, power, linear, and saturating candidates.

    These are descriptive low-sample fits, not asymptotic-law estimators.  The
    saturating family performs a deterministic grid search over the scale term
    and solves the remaining linear coefficients exactly.
    """

    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xv.ndim != 1 or yv.ndim != 1 or len(xv) != len(yv) or len(xv) < 3:
        raise ValueError("x and y must be equal one-dimensional arrays with at least three points")
    if np.any(xv <= 0) or not np.all(np.isfinite(xv)) or not np.all(np.isfinite(yv)):
        raise ValueError("fit inputs must be finite and x must be positive")

    results: list[FitResult] = []
    constant = np.full_like(yv, yv.mean())
    results.append(_finish_fit("constant", {"c": yv.mean()}, yv, constant))

    for family, transformed in (("linear", xv), ("logarithmic", np.log(xv))):
        design = np.column_stack([np.ones_like(transformed), transformed])
        coefficients, *_ = np.linalg.lstsq(design, yv, rcond=None)
        predicted = design @ coefficients
        results.append(
            _finish_fit(
                family,
                {"intercept": coefficients[0], "slope": coefficients[1]},
                yv,
                predicted,
            )
        )

    if np.all(yv > 0):
        design = np.column_stack([np.ones_like(xv), np.log(xv)])
        coefficients, *_ = np.linalg.lstsq(design, np.log(yv), rcond=None)
        predicted = np.exp(design @ coefficients)
        results.append(
            _finish_fit(
                "power",
                {"coefficient": math.exp(coefficients[0]), "exponent": coefficients[1]},
                yv,
                predicted,
            )
        )

    best_saturating: FitResult | None = None
    for scale in np.geomspace(float(xv.min()) / 4.0, float(xv.max()) * 4.0, 120):
        basis = 1.0 - np.exp(-xv / scale)
        design = np.column_stack([np.ones_like(basis), basis])
        coefficients, *_ = np.linalg.lstsq(design, yv, rcond=None)
        candidate = _finish_fit(
            "saturating_exponential",
            {"offset": coefficients[0], "amplitude": coefficients[1], "scale": scale},
            yv,
            design @ coefficients,
        )
        if best_saturating is None or candidate.aic < best_saturating.aic:
            best_saturating = candidate
    assert best_saturating is not None
    results.append(best_saturating)
    return sorted(results, key=lambda fit: (fit.aic, fit.rmse, fit.family))


def pareto_frontier(
    rows: Iterable[Mapping[str, object]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[dict[str, object]]:
    """Return nondominated rows for explicitly named objective directions."""

    values = [dict(row) for row in rows]
    frontier = []
    for index, candidate in enumerate(values):
        dominated = False
        for other_index, other in enumerate(values):
            if index == other_index:
                continue
            weakly_better = all(float(other[key]) >= float(candidate[key]) for key in maximize)
            weakly_better &= all(float(other[key]) <= float(candidate[key]) for key in minimize)
            strictly_better = any(float(other[key]) > float(candidate[key]) for key in maximize)
            strictly_better |= any(float(other[key]) < float(candidate[key]) for key in minimize)
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a SciPy dependency."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))
