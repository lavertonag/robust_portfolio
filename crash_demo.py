# crash_demo.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    DEFAULT_TICKERS,
    START_DATE,
    END_DATE,
    T_TRAIN,
    T_TEST,
    DELTA_MULT,
    RHO,
    GAMMA,
    LAMBDA_RISK,
)
from src.data_loader import load_prices, to_log_returns
from src.backtest import rolling_backtest
from src.estimators import estimate_mean_cov
from src.scenario_analysis import (
    CrashScenario,
    apply_crash_shock,
    compute_efficient_frontier,
    compute_strategy_points,
    get_window_slice,
)
from src.visualization import (
    plot_crash_nav,
    plot_strategy_weight_bars,
    plot_efficient_frontier,
)


def parse_crash_arg(arg: str, default_before: int, default_after: int) -> CrashScenario:
    """
    Parse crash definition formatted as name:date:severity:duration.
    Severity is a simple return (e.g. -0.25 => -25%).
    """
    parts = arg.split(":")
    if len(parts) < 3:
        raise ValueError(f"Invalid crash specification '{arg}'. Expected name:date:severity[:duration].")
    name = parts[0]
    start = pd.Timestamp(parts[1])
    severity = float(parts[2])
    duration = int(parts[3]) if len(parts) > 3 else 1
    return CrashScenario(
        name=name,
        start=start,
        severity=severity,
        duration=duration,
        window_before=default_before,
        window_after=default_after,
    )


def build_crash_scenarios(args, returns_index: pd.Index) -> list[CrashScenario]:
    if args.crash:
        specs = args.crash
    else:
        specs = [
            "Flash2015:2015-08-24:-0.18:3",
            "Covid2020:2020-03-16:-0.30:5",
        ]
    scenarios = []
    for spec in specs:
        scenario = parse_crash_arg(spec, args.window_before, args.window_after)
        if scenario.start < returns_index[0] or scenario.start > returns_index[-1]:
            raise ValueError(f"Scenario {scenario.name} start {scenario.start} outside available data.")
        scenarios.append(scenario)
    return scenarios


def main(args):
    tickers = args.tickers or DEFAULT_TICKERS
    prices = load_prices(
        tickers,
        start=args.start or START_DATE,
        end=args.end or END_DATE,
        source=args.source,
        csv_dir=args.csv_dir,
    )
    returns = to_log_returns(prices)

    scenarios = build_crash_scenarios(args, returns.index)
    strategies = args.strategies or ['markowitz', 'box', 'ellipsoid', 'budgeted']

    for scenario in scenarios:
        shocked_returns, crash_start_pos, crash_end_pos = apply_crash_shock(returns, scenario)
        backtest = rolling_backtest(
            shocked_returns,
            T_train=args.train or T_TRAIN,
            T_test=args.test or T_TEST,
            lambda_risk=args.lambda_risk or LAMBDA_RISK,
            delta_mult=args.delta_mult or DELTA_MULT,
            rho=args.rho or RHO,
            gamma=args.gamma or GAMMA,
            use_shrinkage=not args.no_shrinkage,
            use_garch=not args.no_garch,
            strategies=strategies,
            verbose=False,
        )

        returns_df = pd.DataFrame({name: data['returns'] for name, data in backtest.items()})
        nav_df = (1 + returns_df).cumprod()

        crash_ts = returns.index[crash_start_pos]
        try:
            crash_loc = returns_df.index.get_loc(crash_ts)
        except KeyError:
            crash_loc = returns_df.index.get_indexer([crash_ts], method='nearest')[0]

        window_slice = get_window_slice(returns_df.index, crash_loc, scenario.window_before, scenario.window_after)
        nav_window = nav_df.iloc[window_slice]
        plot_crash_nav(nav_window, scenario.name, crash_start=crash_ts)

        # Extract weights at (or nearest to) crash start for each strategy
        weights_at_crash = {}
        for name, data in backtest.items():
            weights_df = data['weights']
            try:
                w_idx = weights_df.index.get_loc(crash_ts)
            except KeyError:
                w_idx = weights_df.index.get_indexer([crash_ts], method='nearest')[0]
            weights_at_crash[name] = weights_df.iloc[w_idx]

        plot_strategy_weight_bars(weights_at_crash, scenario.name, top_n=args.top_assets)

        # Compute efficient frontier from training window preceding crash
        train_lookback = args.train or T_TRAIN
        training_data = shocked_returns.loc[:crash_ts].tail(train_lookback)
        mu, Sigma = estimate_mean_cov(training_data, use_shrinkage=not args.no_shrinkage)
        frontier = compute_efficient_frontier(mu, Sigma, n_points=args.frontier_points)
        strategy_points = compute_strategy_points(weights_at_crash, mu, Sigma)
        plot_efficient_frontier(frontier, strategy_points, scenario.name)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate crash scenarios and visualize portfolio behaviour.")
    parser.add_argument("--tickers", nargs='*', default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--source", type=str, choices=['yahoo', 'csv'], default='yahoo')
    parser.add_argument("--csv_dir", type=str, default="./data")
    parser.add_argument("--train", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--delta_mult", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lambda_risk", type=float, default=None)
    parser.add_argument("--no_shrinkage", action="store_true")
    parser.add_argument("--no_garch", action="store_true")
    parser.add_argument("--strategies", nargs='*', default=None, help="Subset of strategies to analyze.")

    parser.add_argument(
        "--crash",
        nargs='*',
        help="Crash specification(s) name:date:severity[:duration]; severity as simple return (e.g. -0.25).",
    )
    parser.add_argument("--window_before", type=int, default=60, help="OOS observations to show before crash.")
    parser.add_argument("--window_after", type=int, default=40, help="OOS observations to show after crash.")
    parser.add_argument("--top_assets", type=int, default=8, help="Number of top holdings to display in bar charts.")
    parser.add_argument("--frontier_points", type=int, default=40, help="Number of points on the efficient frontier.")

    args = parser.parse_args()
    main(args)
