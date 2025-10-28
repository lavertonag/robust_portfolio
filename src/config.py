# src/config.py
DEFAULT_TICKERS = [
    # Large-cap US equities
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA',
    # Defensive / diversified equities
    'JPM', 'JNJ', 'PG', 'UNH', 'XOM',
    # Bond and fixed-income ETFs
    'TLT',  # Long-term Treasuries
    'IEF',  # Intermediate Treasuries
    'BND',  # Aggregate US bonds
    'LQD',  # Investment-grade credit
    'HYG',  # High-yield credit
    'TIP',  # TIPS (inflation-protected)
]

START_DATE = '2008-01-01'
END_DATE = None  # None => up to today

# Backtest windowing (trading days)
T_TRAIN = 252      # ~ 1 year
T_TEST = 21        # ~ 1 month rebalancing

# Robust params (default grid-ish)
DELTA_MULT = 1.0   # for Box: delta_i = DELTA_MULT * sigma_forecast_i
RHO = 1.0          # for Ellipsoid: penalty rho * ||M^T w||_2
GAMMA = 2          # for Budgeted: number of active worst deviations

# Risk aversion in Markowitz-like objectives
LAMBDA_RISK = 1.5e-1

# Transaction cost (bps): applied on turnover
TC_BPS = 5.0
