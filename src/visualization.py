# src/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Iterable, Tuple


def plot_cumulative_dual(nav_df: pd.DataFrame):
    """Plot cumulative returns with linear and log scales side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    nav_df.plot(ax=axes[0])
    axes[0].set_title('Cumulative Returns (Linear)')
    axes[0].grid(True)
    nav_df.plot(ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Cumulative Returns (Log)')
    axes[1].grid(True)
    plt.tight_layout()
    return fig, axes


def plot_rolling_sharpe_windows(returns_df: pd.DataFrame, windows=(30, 60, 90)):
    """Plot rolling Sharpe ratios for multiple windows (in trading days)."""
    n = len(windows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, window in zip(axes, windows):
        rolling_mean = returns_df.rolling(window).mean() * 252
        rolling_vol = returns_df.rolling(window).std() * np.sqrt(252)
        sharpe = rolling_mean / rolling_vol
        sharpe.plot(ax=ax, title=f'Rolling Sharpe ({window}d window)')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.grid(True)
    plt.tight_layout()
    return fig, axes


def plot_return_boxplots(returns_df: pd.DataFrame):
    """Display boxplots of out-of-sample returns per strategy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    returns_df.plot(kind='box', ax=ax)
    ax.set_ylabel('Return')
    ax.set_title('Out-of-sample Return Distribution per Strategy')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig, ax


def plot_return_volatility_scatter(returns_df: pd.DataFrame):
    """Scatter plot of annualized return versus volatility for each strategy."""
    annual_return = returns_df.mean() * 252
    annual_vol = returns_df.std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(annual_vol, annual_return, s=80)
    for name in returns_df.columns:
        ax.annotate(name, (annual_vol[name], annual_return[name]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Return vs Volatility (OOS)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig, ax


def plot_crash_nav(nav_df: pd.DataFrame, scenario_name: str, crash_start: pd.Timestamp | None = None):
    """Plot strategy NAVs over a crash window."""
    fig, ax = plt.subplots(figsize=(11, 5))
    nav_df.plot(ax=ax)
    if crash_start is not None and crash_start in nav_df.index:
        ax.axvline(crash_start, color='black', linestyle='--', linewidth=1.2, label='Crash start')
    ax.set_title(f'Scenario {scenario_name}: NAV during crash window')
    ax.set_ylabel('Cumulative NAV')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig, ax


def plot_strategy_weight_bars(
    weights: Dict[str, pd.Series],
    scenario_name: str,
    top_n: int = 8,
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    """
    Bar charts of portfolio weights (largest holdings) per strategy for a scenario.
    """
    strategies = list(weights.keys())
    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, strategies):
        series = weights[name].fillna(0.0).astype(float)
        ordered = series.sort_values(ascending=False)
        if top_n and len(ordered) > top_n:
            top = ordered.iloc[:top_n]
            remainder = ordered.iloc[top_n:].sum()
            if remainder > 0:
                top.loc['Other'] = remainder
            ordered = top
        ordered.plot(kind='bar', ax=ax)
        ax.set_title(name)
        ax.set_ylabel('Weight')
        ax.set_ylim(0, min(1.05, ordered.max() * 1.2 if len(ordered) else 1.0))
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')
    fig.suptitle(f'Scenario {scenario_name}: Allocation by strategy', y=1.02, fontsize=13)
    fig.tight_layout()
    return fig, axes


def plot_efficient_frontier(
    frontier: np.ndarray,
    strategy_points: Dict[str, Tuple[float, float]],
    scenario_name: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Markowitz efficient frontier with strategy markers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if frontier.size > 0:
        ax.plot(frontier[:, 0], frontier[:, 1], label='Efficient frontier', color='tab:blue')
    for name, (vol, ret) in strategy_points.items():
        ax.scatter(vol, ret, label=name, s=70)
        ax.annotate(name, (vol, ret), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Expected Return')
    ax.set_title(f'Scenario {scenario_name}: Efficient Frontier')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig, ax
