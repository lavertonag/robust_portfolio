# demo.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.config import DEFAULT_TICKERS, START_DATE, END_DATE, T_TRAIN, T_TEST, DELTA_MULT, RHO, GAMMA, LAMBDA_RISK
from src.data_loader import load_prices, to_log_returns
from src.backtest import rolling_backtest
from src.metrics import compile_performance_table
from src.visualization import (
    plot_cumulative_dual,
    plot_rolling_sharpe_windows,
    plot_return_volatility_scatter,
    plot_crash_nav,
    plot_strategy_weight_bars,
)


def main(args):
    tickers = args.tickers or DEFAULT_TICKERS
    prices = load_prices(tickers, start=args.start or START_DATE, end=args.end or END_DATE, source=args.source, csv_dir=args.csv_dir)
    rets = to_log_returns(prices)

    res = rolling_backtest(
        rets,
        T_train=args.train or T_TRAIN,
        T_test=args.test or T_TEST,
        lambda_risk=args.lambda_risk or LAMBDA_RISK,
        delta_mult=args.delta_mult or DELTA_MULT,
        rho=args.rho or RHO,
        gamma=args.gamma or GAMMA,
        use_shrinkage=not args.no_shrinkage,
        use_garch=not args.no_garch,
        strategies=['markowitz','box','ellipsoid','budgeted']
    )

    # Les resultatds du backtest stpockés dans un dataframe ou colonne = strategies + lignes = rendements 
    returns_df = pd.DataFrame({k: v['returns'] for k, v in res.items()})
    nav = (1 + returns_df).cumprod() #capitalisation cumulé (evolution du cpatial initial)

    # Visuals
    figures = []
    fig, _ = plot_cumulative_dual(nav) # courbe cumule 
    figures.append(fig)
    fig, _ = plot_rolling_sharpe_windows(returns_df, windows=(30,)) # sharpe rolling
    figures.append(fig)
    fig, _ = plot_return_volatility_scatter(returns_df)
    figures.append(fig)

    # Crash window viz (based on worst drawdown for baseline strategy)
    baseline_nav = nav['markowitz'] if 'markowitz' in nav else nav.iloc[:, 0]
    rolling_peak = baseline_nav.cummax()
    drawdowns = baseline_nav / rolling_peak - 1.0
    trough_date = drawdowns.idxmin()
    if pd.notna(trough_date):
        peak_date = baseline_nav.loc[:trough_date].idxmax()
        window_start = peak_date - pd.Timedelta(days=45)
        window_end = trough_date + pd.Timedelta(days=45)
        crash_slice = nav.loc[(nav.index >= window_start) & (nav.index <= window_end)]
        if not crash_slice.empty:
            crash_start = peak_date if peak_date in crash_slice.index else crash_slice.index[0]
            fig, _ = plot_crash_nav(crash_slice, scenario_name='Worst drawdown', crash_start=crash_start)
            figures.append(fig)

    # Strategy allocation bars (average monthly weights over OOS)
    averaged_weights = {}
    for name, data in res.items():
        weights_df = data['weights']
        if weights_df.empty:
            continue
        if isinstance(weights_df.index, pd.DatetimeIndex):
            monthly = weights_df.resample('ME').mean()
            if not monthly.empty:
                averaged_weights[name] = monthly.mean()
                continue
        averaged_weights[name] = weights_df.mean()

    if averaged_weights:
        fig, _ = plot_strategy_weight_bars(averaged_weights, scenario_name='Average allocation', top_n=8)
        figures.append(fig)

    # Summary table
    summary = compile_performance_table(res, base_strategy='markowitz' if 'markowitz' in res else None)
    print("\n=== Summary (annualized) ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))
    plt.show(block=False)
    try:
        input("\nPress Enter to close the figures...")
    finally:
        plt.close('all')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs='*', default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--source", type=str, choices=['yahoo','csv'], default='yahoo')
    p.add_argument("--csv_dir", type=str, default="./data")
    p.add_argument("--train", type=int, default=None)
    p.add_argument("--test", type=int, default=None)
    p.add_argument("--delta_mult", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lambda_risk", type=float, default=None)
    p.add_argument("--no_shrinkage", action="store_true")
    p.add_argument("--no_garch", action="store_true")
    args = p.parse_args()
    main(args)
