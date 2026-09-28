# Colombian Portfolio Architecture

A single Python script, [`COL_Portfolio_architecture.py`](COL_Portfolio_architecture.py), that builds a direct-investment portfolio of Colombian stocks and ETFs: it downloads prices, reads the risk-free rate from the TES zero-coupon curve, reports per-asset risk metrics, finds the maximum Sharpe portfolio, and validates it with an out-of-sample walk-forward backtest.

It prints tables to the console and writes an interactive Plotly report to `outputs/reporte_interactivo.html`.

> Code comments, console output and the HTML report are in Spanish.

---

## Investable universe

Defined in the `AssetUniverse` dataclass:

| Group | Tickers |
|---|---|
| Colombian stocks (BVC) | 28 names, e.g. Ecopetrol, ISA, GEB, Celsia, Promigas, Grupo Sura, Grupo Argos, Cementos Argos, Davivienda, Grupo Aval, Cibest (Bancolombia), Corficolombiana, PEI, nuam, among others (common and preferred shares) |
| Local ETFs (BVC) | `ICOLCAP.CL`, `HCOLSEL.CL`, `GXTESCOL.CL` |
| Global ETFs | `SPY`, `QQQ`, `TLT` |
| Benchmark (for betas) | `^GSPC` |

## How it works

The script runs as a demo from `main()`, in six steps:

### 1. Market data (`MarketDataPipeline`)
- Downloads daily adjusted close prices from **Yahoo Finance** (`yfinance`) for 2021-01-01 to 2024-12-31.
- If a ticker fails to download, it generates a **synthetic series** (geometric Brownian motion with a common market factor plus an idiosyncratic shock), so the pipeline always has data to run.
- Tickers that are synthetic or whose real history starts late are **excluded from the joint optimization and the backtest**, so they don't shrink the common sample window. They still appear in the per-asset risk report.

### 2. Risk-free rate from the TES curve (`TESYieldCurve`)
- Models the zero-coupon term structure (ETTI) of Colombian government bonds (TES), interpolated with a natural cubic spline.
- Provides the interpolated rate, discount factors and a curve chart.
- The curve can be loaded from a Banco de la República CSV (`from_banrep_csv`, with columns `plazo_anios` and `tasa`). The demo uses an illustrative sample curve (`synthetic_example`).
- The risk-free rate used in the Sharpe ratio is the **1-year TES rate** (`RF_TENOR_YEARS`).

### 3. Per-asset risk metrics (`RiskMetrics`)
For each asset: annualized volatility, annualized geometric return, beta against the S&P 500, maximum drawdown and Sharpe ratio.

### 4. Maximum Sharpe optimization (`PortfolioOptimizer`)
- Long-only, fully invested, with a **35% cap per asset**.
- Covariance estimated with **Ledoit-Wolf shrinkage** (scikit-learn), which fixes the ill-conditioning of the sample covariance and prevents extreme corner solutions. The shrinkage intensity δ is reported.
- Solved with SciPy SLSQP. If the solver doesn't converge, it falls back to equal weights.
- Also plots the **efficient frontier** from 3,000 random Monte Carlo portfolios, highlighting the tangency portfolio.

### 5. Walk-forward backtest (`WalkForwardBacktester`)
- At each month end, re-optimizes the maximum Sharpe portfolio using only the previous 252 trading days, then applies those weights to the following month's returns. This avoids look-ahead bias.
- Reports total and annualized return, volatility, Sharpe ratio, maximum drawdown and number of rebalances.
- Plots the equity curve and how the weights evolve across rebalances.

### 6. HTML report
All charts (efficient frontier, TES curve, equity curve, weight evolution) are combined into a single interactive HTML page.

---

## Requirements

Python 3.10 or later and:

```bash
pip install numpy pandas scipy scikit-learn plotly yfinance
```

`yfinance` is optional; without it every series falls back to synthetic data.

## Usage

```bash
python COL_Portfolio_architecture.py
```

The report is written to `outputs/` next to the script. Set the `OUTPUT_DIR` environment variable to change the location.

To adjust the universe, date range, weight cap, rebalance frequency or risk-free tenor, edit `AssetUniverse` and the parameters inside `main()`.

## Disclaimer

This code is for academic and research purposes only and does not constitute investment advice. The sample TES curve is illustrative; use the official Banco de la República data for real analysis.
