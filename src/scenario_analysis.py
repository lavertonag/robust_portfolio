# src/scenario_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import cvxpy as cp



@dataclass(frozen=True)
class CrashScenario:
    name: str
    start: pd.Timestamp
    severity: float  # simple return shock, negative for drawdown (e.g. -0.25 => -25%)
    duration: int = 1  # trading days affected by the shock
    window_before: int = 60  # number of OOS observations to display before the crash start
    window_after: int = 40   # number of OOS observations to display after the crash window

    def shock_log_return(self) -> float:
        if self.severity <= -1:
            raise ValueError("Severity must be greater than -1 (cannot lose more than 100%).")
        return float(np.log1p(self.severity))


def apply_crash_shock(returns: pd.DataFrame, scenario: CrashScenario) -> Tuple[pd.DataFrame, int, int]:
    """
    Inject a synthetic crash by adding a constant log-return shock across all assets.

    Returns the shocked returns DataFrame and the (start, end) positional indices
    of the crash window in the DataFrame.
    """
    shocked = returns.copy()
    idx = shocked.index
    start = pd.Timestamp(scenario.start)
    if start < idx[0] or start > idx[-1]:
        raise ValueError(f"Crash start {scenario.start} outside data range.")
    try:
        start_loc = int(idx.get_loc(start))
    except KeyError:
        start_loc = int(idx.get_indexer([start], method='nearest')[0])
        start = idx[start_loc]

    end_loc = min(start_loc + scenario.duration, len(idx))
    crash_slice = idx[start_loc:end_loc]
    shock = scenario.shock_log_return()
    shocked.loc[crash_slice] = shocked.loc[crash_slice] + shock
    return shocked, start_loc, end_loc - 1


def compute_efficient_frontier(mu: np.ndarray, Sigma: np.ndarray, n_points: int = 40) -> np.ndarray:
    """Compute long-only efficient frontier as (vol, ret) pairs."""
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    if n == 0:
        return np.empty((0, 2))

    mu_min = float(mu.min())
    mu_max = float(mu.max())
    if np.isclose(mu_min, mu_max):
        span = abs(mu_min) if mu_min != 0 else 0.01
        targets = np.linspace(mu_min - span, mu_min + span, n_points)
    else:
        targets = np.linspace(mu_min, mu_max, n_points)

    w = cp.Variable(n)
    target_ret = cp.Parameter()
    constraints = [cp.sum(w) == 1, w >= 0, mu @ w >= target_ret]
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)

    frontier: List[Tuple[float, float]] = []
    for t in targets:
        target_ret.value = t
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                problem.solve(solver=cp.ECOS)
        except cp.SolverError:
            continue
        if w.value is None:
            continue
        weights = np.array(w.value, dtype=float).reshape(-1)
        ret = float(mu @ weights)
        vol = float(np.sqrt(weights.T @ Sigma @ weights))
        frontier.append((vol, ret))

    return np.array(frontier, dtype=float)


def compute_strategy_points(
    weights: Dict[str, pd.Series],
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    """Return annualized (vol, ret) for each strategy weight vector."""
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    out: Dict[str, Tuple[float, float]] = {}
    for name, series in weights.items():
        w = np.asarray(series.fillna(0.0), dtype=float)
        ret = float(mu @ w)
        vol = float(np.sqrt(w.T @ Sigma @ w))
        out[name] = (vol, ret)
    return out


def get_window_slice(index: pd.Index, center_pos: int, before: int, after: int) -> slice:
    """Helper to obtain positional slice bounds for plotting windows."""
    start = max(0, center_pos - before)
    stop = min(len(index), center_pos + after + 1)
    return slice(start, stop)
