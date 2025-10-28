# src/optimizers.py
from __future__ import annotations

from typing import Any, Iterable, Tuple
import warnings

import cvxpy as cp
import numpy as np


SolverSpec = Tuple[Any, dict[str, Any]]


def _solve_with_fallback(prob: cp.Problem, var: cp.Variable, solvers: Iterable[SolverSpec] | None = None, verbose: bool = False) -> np.ndarray:
    """
    Try a sequence of CVXPY solvers and return the associated variable value.
    Falls back to the best (possibly inaccurate) solution if no solver reaches OPTIMAL.
    """
    solver_chain = list(solvers) if solvers is not None else [
        (cp.ECOS, {}),
        (cp.OSQP, {"eps_abs": 1e-7, "eps_rel": 1e-7, "polish": True, "max_iter": 100000}),
        (cp.SCS, {"eps": 1e-6}),
    ]
    best_value: np.ndarray | None = None
    for solver, opts in solver_chain:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Solution may be inaccurate",
                    category=UserWarning,
                    module="cvxpy",
                )
                prob.solve(solver=solver, verbose=verbose, warm_start=True, **opts)
        except cp.SolverError:
            continue
        status = prob.status
        value = var.value
        if value is not None:
            value = np.array(value, dtype=float).reshape(-1)
        if status == cp.OPTIMAL and value is not None:
            return value
        if status == cp.OPTIMAL_INACCURATE and value is not None and best_value is None:
            best_value = value.copy()
    if best_value is not None:
        return best_value
    raise RuntimeError(f"Unable to solve problem; last status={prob.status}")


def markowitz(mu, Sigma, lambda_risk: float = 0.05, long_only: bool = True) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(w @ mu - lambda_risk * cp.quad_form(w, Sigma))
    constraints = []
    if long_only:
        constraints.append(w >= 0)
    constraints.append(cp.sum(w) == 1)
    prob = cp.Problem(objective, constraints)
    return _solve_with_fallback(prob, w)


def robust_box(mu, Sigma, delta, lambda_risk: float = 0.05, long_only: bool = True) -> np.ndarray:
    """Worst-case mean with box uncertainty: mu_i in [mu_i - delta_i, mu_i + delta_i] (long-only)."""
    n = len(mu)
    w = cp.Variable(n)
    penalty = cp.sum(cp.multiply(cp.abs(w), delta))  # long-only => w == abs(w)
    objective = cp.Maximize(w @ mu - penalty - lambda_risk * cp.quad_form(w, Sigma))
    constraints = []
    if long_only:
        constraints.append(w >= 0)
    constraints.append(cp.sum(w) == 1)
    prob = cp.Problem(objective, constraints)
    return _solve_with_fallback(prob, w)


def robust_ellipsoid(mu, Sigma, M, rho: float = 1.0, lambda_risk: float = 0.05, long_only: bool = True) -> np.ndarray:
    """Ellipsoidal mean uncertainty with error-shape matrix M (s.t. u = M^T w)."""
    n = len(mu)
    w = cp.Variable(n)
    penalty = rho * cp.norm(M.T @ w, 2)
    objective = cp.Maximize(w @ mu - penalty - lambda_risk * cp.quad_form(w, Sigma))
    constraints = []
    if long_only:
        constraints.append(w >= 0)
    constraints.append(cp.sum(w) == 1)
    prob = cp.Problem(objective, constraints)
    return _solve_with_fallback(prob, w)


def robust_budgeted(mu, Sigma, delta, Gamma: float = 2.0, lambda_risk: float = 0.05, long_only: bool = True) -> np.ndarray:
    """
    Bertsimas-Sim budgeted uncertainty (long-only simplification).
    """
    n = len(mu)
    w = cp.Variable(n)
    t = cp.Variable()
    s = cp.Variable(n)
    penalty = Gamma * t + cp.sum(s)
    objective = cp.Maximize(w @ mu - penalty - lambda_risk * cp.quad_form(w, Sigma))
    constraints = [s >= 0, s >= cp.multiply(delta, w) - t, t >= 0]
    if long_only:
        constraints.append(w >= 0)
    constraints.append(cp.sum(w) == 1)
    prob = cp.Problem(objective, constraints)
    return _solve_with_fallback(prob, w)


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.nan_to_num(w, nan=0.0)
    w[w < 0] = 0.0
    total = w.sum()
    if total <= 0:
        return np.ones(len(w)) / len(w)
    return w / total
