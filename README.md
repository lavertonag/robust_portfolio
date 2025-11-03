# Robust Portfolio Optimization - Project Scaffold

This repository ships a minimal, ready-to-run pipeline to compare **Markowitz** vs **Robust** (Box, Ellipsoid, Budgeted) portfolio optimization with **rolling backtests**.

## Quick start

```bash

#0) Puor moi le lancer 
Set-ExecutionPolicy -Scope Process Bypass                                  
 .\.venv\Scripts\Activate.ps1

 
# 1) Create and activate a fresh virtual environment (built-in venv)
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the demo
python demo.py
```

If you prefer Conda, you can still create an environment with `conda create -n robust-ptf python=3.10` before installing the requirements.

If Yahoo Finance is blocked, place your CSVs in `./data/` named `{TICKER}.csv` (OHLCV or at least `Date,Adj Close`). The loader will read from there when you pass `--source csv`.

## Project layout

```
robust_portfolio/
  README.md
  requirements.txt
  demo.py
  src/
    backtest.py
    config.py
    data_loader.py
    estimators.py
    metrics.py
    optimizers.py
    scenario_analysis.py
    visualization.py
```

## Notes

- The **GARCH** step is optional; if `arch` is missing the code falls back gracefully.
- The **Budgeted (Bertsimas-Sim)** model includes a simple convex formulation for long-only portfolios.
- This scaffold favors clarity and extensibility over micro-optimizations.
