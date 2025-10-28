# src/backtest.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable, List, Tuple, Optional

from .estimators import estimate_mean_cov, garch_vol_forecast, covariance_of_mean
from .optimizers import markowitz, robust_box, robust_ellipsoid, robust_budgeted, normalize_weights
from .metrics import turnover

def rolling_backtest(
    returns: pd.DataFrame,
    T_train: int = 252,
    T_test: int = 21,
    lambda_risk: float = 0.05,
    delta_mult: float = 1.0,
    rho: float = 1.0,
    gamma: float = 2.0,
    use_shrinkage: bool = True,
    use_garch: bool = True,
    strategies: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    if strategies is None:
        strategies = ['markowitz','box','ellipsoid','budgeted']

    T, n = returns.shape
    idx = returns.index

    # Storage
    returns_oos = {name: [] for name in strategies}
    weights_hist = {name: [] for name in strategies}
    dates_hist = []

    w_prev = {name: np.ones(n)/n for name in strategies}
    t = T_train
    while t + T_test <= T:
        train = returns.iloc[t - T_train:t]
        test = returns.iloc[t:t+T_test]
        mu, Sigma = estimate_mean_cov(train, use_shrinkage=use_shrinkage)

        # Uncertainty components
        Cov_mu = covariance_of_mean(train, B=300, block_size=min(21, len(train)))  # shape (n,n)
        delta = np.sqrt(np.maximum(np.diag(Cov_mu), 0.0))
        if use_garch:
            sigma_scale = garch_vol_forecast(train)
            avg_sigma = np.nanmean(sigma_scale)
            if not np.isfinite(avg_sigma) or avg_sigma <= 0:
                rel_scale = np.ones_like(sigma_scale)
            else:
                rel_scale = np.nan_to_num(sigma_scale / avg_sigma, nan=1.0, posinf=1.0, neginf=1.0)
            # Scale the mean-uncertainty by relative volatility so high-vol assets remain conservative
            delta = delta * rel_scale
        delta = delta_mult * delta

        # Solve per strategy
        W = {}
        if 'markowitz' in strategies:
            W['markowitz'] = normalize_weights(markowitz(mu, Sigma, lambda_risk=lambda_risk))
        if 'box' in strategies:
            W['box'] = normalize_weights(robust_box(mu, Sigma, delta, lambda_risk=lambda_risk))
        if 'ellipsoid' in strategies:
            M = np.linalg.cholesky(Cov_mu)  # M s.t. Cov_mu = M M^T
            W['ellipsoid'] = normalize_weights(robust_ellipsoid(mu, Sigma, M, rho=rho, lambda_risk=lambda_risk))
        if 'budgeted' in strategies:
            W['budgeted'] = normalize_weights(robust_budgeted(mu, Sigma, delta, Gamma=gamma, lambda_risk=lambda_risk))

        # Apply on test window
        for name in strategies:
            w = W[name]
            # portfolio returns over T_test
            r_ptf = test.values @ w
            returns_oos[name].extend(r_ptf.tolist())
            weights_hist[name].extend([w.copy() for _ in range(len(test))])

        dates_hist.extend(list(test.index))
        t += T_test

    # Assemble outputs
    out = {}
    for name in strategies:
        out[name] = {
            'returns': pd.Series(returns_oos[name], index=pd.DatetimeIndex(dates_hist), name=name),
            'weights': pd.DataFrame(np.vstack(weights_hist[name]), index=pd.DatetimeIndex(dates_hist), columns=returns.columns)
        }
    return out
