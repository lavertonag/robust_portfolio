# Robust Portfolio Optimization — Project Scaffold

This repo provides a minimal, ready-to-run pipeline to compare **Markowitz** vs **Robust** (Box, Ellipsoid, Budgeted) portfolio optimization with **rolling backtests**.

## Quick start

```bash
# 1) Create and activate a fresh environment (example with conda)
conda create -n robust-ptf python=3.10 -y
conda activate robust-ptf

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the demo
python demo.py
```

If Yahoo Finance is blocked, place your CSVs in `./data/` named as `{TICKER}.csv` (OHLCV or at least `Date,Adj Close`), and the loader will read from there with `--source csv`.

## Structure

```
robust_portfolio/
  ├─ src/
  │   ├─ data_loader.py
  │   ├─ estimators.py
  │   ├─ optimizers.py
  │   ├─ backtest.py
  │   ├─ metrics.py
  │   ├─ visualization.py
  │   └─ config.py
  ├─ notebooks/
  ├─ demo.py
  ├─ requirements.txt
  └─ README.md
```

## Notes

- The **GARCH** step is optional; if `arch` isn't installed, the code falls back gracefully.
- The **Budgeted (Bertsimas–Sim)** model includes a simple convex formulation for long-only portfolios.
- This scaffold emphasizes clarity and extensibility over micro-optimizations.
