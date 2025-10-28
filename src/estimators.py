# src/estimators.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def ensure_pd(Sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Make covariance positive definite by jittering the diagonal if needed."""
    Sigma = np.array(Sigma, dtype=float)
    # Try Cholesky; if fails, add eps on diagonal progressively
    for k in range(6):
        try:
            np.linalg.cholesky(Sigma)
            return Sigma
        except np.linalg.LinAlgError:
            Sigma = Sigma + (eps * (10**k)) * np.eye(Sigma.shape[0])
    # Last resort
    w, v = np.linalg.eigh(Sigma)
    w[w < eps] = eps
    return (v * w) @ v.T

def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit–Wolf shrinkage estimator via scikit-learn (if available)."""
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.values)
        return lw.covariance_
    except Exception:
        # Fallback to sample covariance
        return np.cov(returns.values, rowvar=False)

def estimate_mean_cov(returns: pd.DataFrame, use_shrinkage: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate mean vector and covariance matrix from returns DataFrame."""
    mu = returns.mean().values  # per-period mean
    if use_shrinkage:
        Sigma = ledoit_wolf_cov(returns)
    else:
        Sigma = np.cov(returns.values, rowvar=False)
    Sigma = ensure_pd(Sigma)
    return mu, Sigma

def garch_vol_forecast(returns: pd.DataFrame, scale: float = 1.0) -> np.ndarray:
    """Per-asset GARCH(1,1) forecasted sigma for next step. If arch missing, fallback to stdev."""
    sigmas = []
    try:
        from arch import arch_model
        for col in returns.columns:
            r = 1_000 * returns[col].dropna().values  # scale for stability
            if len(r) < 50:
                sigmas.append(returns[col].std())
                continue
            am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal', rescale=False)
            res = am.fit(disp='off')
            f = res.forecast(horizon=1, reindex=False)
            sigma_next = float(f.variance.values[-1, 0])**0.5 / 1_000.0
            sigmas.append(sigma_next)
    except Exception:
        sigmas = [returns[col].std() for col in returns.columns]
    return scale * np.array(sigmas, dtype=float)

def block_bootstrap_means(returns: pd.DataFrame, B: int = 500, block_size: int = 21, random_state: Optional[int]=42) -> np.ndarray:
    """Block bootstrap to get distribution of mean returns. Returns array shape (B, n_assets)."""
    rng = np.random.default_rng(random_state)
    X = returns.values
    T, n = X.shape
    n_blocks = int(np.ceil(T / block_size))
    means = np.zeros((B, n))
    for b in range(B):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, T - block_size + 1))
            idx.extend(list(range(start, min(start + block_size, T))))
        idx = np.array(idx[:T])
        means[b, :] = X[idx, :].mean(axis=0)
    return means

def covariance_of_mean(returns: pd.DataFrame, B: int = 500, block_size: int = 21) -> np.ndarray:
    samples = block_bootstrap_means(returns, B=B, block_size=block_size)
    cov_mu = np.cov(samples, rowvar=False)
    return ensure_pd(cov_mu)
