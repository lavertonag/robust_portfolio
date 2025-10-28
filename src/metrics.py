# src/metrics.py
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    cum = (1 + returns).prod()
    n = len(returns) / periods_per_year
    return cum ** (1/n) - 1 if n > 0 else np.nan

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)

def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    ex = returns - rf/periods_per_year
    mu = ex.mean() * periods_per_year
    sig = ex.std() * np.sqrt(periods_per_year)
    return mu / sig if sig > 0 else np.nan

def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    ex = returns - rf/periods_per_year
    downside = ex[ex < 0]
    dd = downside.std() * np.sqrt(periods_per_year)
    mu = ex.mean() * periods_per_year
    return mu / dd if dd > 0 else np.nan

def max_drawdown(returns: pd.Series) -> float:
    nav = (1 + returns).cumprod()
    peak = nav.cummax()
    dd = (nav / peak - 1).min()
    return dd

def cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    q = returns.quantile(1 - alpha)
    tail = returns[returns <= q]
    return tail.mean() if len(tail) > 0 else np.nan

def turnover(w_old, w_new) -> float:
    w_old = np.nan_to_num(w_old, nan=0.0)
    w_new = np.nan_to_num(w_new, nan=0.0)
    return np.abs(w_new - w_old).sum()

def hhi(weights) -> float:
    w = np.nan_to_num(weights, nan=0.0)
    return float((w**2).sum())


def compile_performance_table(
    results: Dict[str, Dict[str, pd.DataFrame]],
    base_strategy: Optional[str] = 'markowitz',
) -> pd.DataFrame:
    """
    Build a performance summary table with a paired t-test against the base strategy.
    """
    returns_df = pd.DataFrame({name: data['returns'] for name, data in results.items()})
    summary = {}
    for name, data in results.items():
        r = data['returns']
        w_last = data['weights'].iloc[-1].values if len(data['weights']) > 0 else np.array([])
        summary[name] = {
            'CAGR': cagr(r),
            'Vol': annualized_vol(r),
            'Sharpe': sharpe(r),
            'Sortino': sortino(r),
            'MaxDD': max_drawdown(r),
            'CVaR95': cvar(r, 0.95),
            'HHI (last)': hhi(w_last) if w_last.size > 0 else np.nan,
        }
    table = pd.DataFrame(summary).T

    base = base_strategy if base_strategy in returns_df.columns else None
    if base is not None:
        p_values = {}
        for name in returns_df.columns:
            aligned = returns_df[[name, base]].dropna()
            if len(aligned) < 2 or name == base:
                p_values[name] = np.nan
                continue
            _, p_val = stats.ttest_rel(aligned[name], aligned[base])
            p_values[name] = p_val
        table[f'p_value_vs_{base}'] = table.index.map(p_values.get)

    return table
