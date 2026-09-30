# Colombian Portfolio Architecture

A single Python script, [`COL_Portfolio_architecture.py`](COL_Portfolio_architecture.py), that runs a **multi-asset allocation engine** for the Colombian market. It combines equities (BVC stocks and ETFs) and fixed income (TES curve nodes, bond ETFs and individual bonds) in one return/covariance matrix. It finds the maximum Sharpe portfolio under strategic bands per asset class and validates it with an out-of-sample walk-forward backtest.

It prints tables to the console and writes an interactive Plotly report to `outputs/reporte_interactivo.html`.

> Code comments, console output and the HTML report are in Spanish.

---

## Investable universe

Defined in the `AssetUniverse` dataclass. Its defaults come from the editable parameter block at the top of the script.

| Class | Group | Tickers |
|---|---|---|
| Equities (RV) | Colombian stocks (BVC) | 28 names, e.g. Ecopetrol, ISA, GEB, Celsia, Grupo Sura, Grupo Argos, Cementos Argos, Davivienda, Grupo Aval, Cibest (Bancolombia), Corficolombiana, PEI, nuam (common and preferred shares) |
| Equities (RV) | Local ETFs | `ICOLCAP.CL`, `HCOLSEL.CL` |
| Equities (RV) | Global ETFs | `SPY`, `QQQ` |
| Fixed income (RF) | A: TES curve nodes | `TES_1Y`, `TES_3Y`, `TES_5Y`, `TES_10Y` (synthetic) |
| Fixed income (RF) | B: bond ETFs / funds | `TLT` (`GXTESCOL.CL` is left out: Yahoo Finance has no history for it) |
| Fixed income (RF) | C: individual bonds | Local bonds that pass the screening (see below) |
| — | Benchmark (for betas) | `^GSPC` |

Not every listed ticker ends up in the optimization. See the data rules in step 1.

## How it works

The script runs from `main()` in eight steps.

### 1. Market data and universe selection (`MarketDataPipeline`)
- Downloads daily adjusted close prices from **Yahoo Finance** (`yfinance`) for 2021-01-01 to 2024-12-31.
- If a ticker fails to download, it generates a **synthetic series** (geometric Brownian motion with a common market factor plus an idiosyncratic shock). The seed is stable per ticker (CRC32), so runs are reproducible.
- When calendars are aligned, a missing price is carried forward for **at most 5 days**. A security that stops trading shows up as missing data instead of a flat price.
- A ticker is **excluded** from the optimization and the backtest when any of these holds:
  - its real history starts more than 45 days after the start date, or ends more than 45 days before the end date;
  - it only has a synthetic series (`EXCLUIR_ACTIVOS_SIN_PRECIO_REAL`);
  - it is **illiquid**: more than 25% of its days have exactly zero return, or its price stays frozen for more than 30 days in a row. This is measured on the ticker's own calendar. Stale prices understate volatility and correlation, and a Sharpe optimizer rewards exactly that.
- Every exclusion is logged with its reason and listed in the HTML report.
- If fewer than 5 equities with real prices remain (for example, with no network access), the engine switches to a **demo mode** on the synthetic series and says so in the log.

### 2. TES yield curve (`TESYieldCurve`)
- Models the zero-coupon term structure (ETTI) of Colombian government bonds, interpolated with a natural cubic spline.
- Provides interpolated rates, discount factors and par yields.
- The curve comes from the nodes in `CURVA_TES_PLAZOS` / `CURVA_TES_TASAS`, or from a Banco de la República CSV (`RUTA_CURVA_TES_CSV`, with columns `plazo_anios` and `tasa`).
- The risk-free rate is the **1-year TES rate** (`RF_TENOR_YEARS`).

### 3. Fixed-income engine (`FixedIncomeEngine`)
The fixed-income leg combines three approaches:

- **A. TES curve nodes (`TESNodeBuilder`).** Each node is a par-coupon bond at its tenor. Its daily return is carry plus the second-order price change ΔP/P ≈ −D·Δy + ½·C·(Δy)².
- **B. Bond ETFs / funds.** Vehicles with an observable market price, taken from the price pipeline.
- **C. Individual bonds (`BondScreener`).** A policy filter over a bond universe. The default is 12 illustrative sovereign and corporate bonds, which you can replace with real ISINs. The filters are:
  - rating ≥ AA+ on the local scale;
  - residual maturity between 1 and 10 years;
  - liquidity score ≥ 0.35;
  - spread over the TES curve ≤ 450 bp;
  - fixed-rate peso bonds only (no UVR/IPC);
  - at most 2 bonds per issuer.

Daily yield-curve changes come from a **three-factor Nelson-Siegel model** (level, slope, curvature) with mean reversion and a link to equity returns (`CurveShockGenerator`). A real curve history can replace it through `from_history`.

The expected return (μ) of A and C is **analytical** (YTM + roll-down), not a sample mean. It is recomputed from the curve in force at each rebalance date, which is the tactical part of the allocation.

### 4. Unified multi-asset matrix
Equity prices, bond-ETF prices and the modeled fixed-income series are aligned by date into one price and return matrix.

### 5. Per-asset risk metrics (`RiskMetrics`)
For each asset: annualized volatility, annualized geometric return, beta against the S&P 500, maximum drawdown, Sharpe ratio and modified duration (fixed income).

### 6. Strategic optimization (`PortfolioOptimizer`, `AllocationPolicy`)
Maximum Sharpe, long-only and fully invested, with:
- **strategic bands per asset class:** equities 40%–60%, fixed income 40%–60%;
- **caps per instrument type:** single stock 15%, equity ETF 20%, bond ETF 20%, TES node 20%, individual bond 15%.

Other details:
- The covariance is estimated with **Ledoit-Wolf shrinkage** (scikit-learn), and the shrinkage intensity δ is reported.
- Because moments are estimated on log returns, fixed-income μ and the risk-free rate are both converted to log scale, ln(1 + r).
- The problem is solved with SciPy SLSQP from a feasible starting point, with random feasible restarts if it fails. As a last resort it falls back to the midpoint of the bands.
- Infeasible bands are detected before calling the solver.
- The efficient frontier plot uses 3,000 Monte Carlo portfolios drawn **inside the bands**.

### 7. Walk-forward backtest (`WalkForwardBacktester`)
- Rebalances on the **last trading day of each month** (`FRECUENCIA_REBALANCEO`), using only the previous 252 trading days. The weights are applied to the following period, which avoids look-ahead bias.
- Each rebalance uses the tactical fixed-income μ and the risk-free rate read from the curve at that date.
- Daily portfolio returns are aggregated from **simple returns**, then converted to log returns.
- Reports total and annualized return, volatility, Sharpe ratio, maximum drawdown, number of rebalances, average weight per class, and whether the bands were respected at every rebalance.

### 8. HTML report
A single interactive page with:
- **Charts:** the TES curve and its simulated path per node, the bond-screening map, the efficient frontier, equity vs fixed-income allocation over time with the policy bands, the internal composition of the fixed-income leg (A/B/C), weights per asset, and the equity curve.
- **Tables:** fixed-income analytics, screening verdicts, the optimal portfolio, band compliance, backtest performance, per-asset risk metrics, and liquidity and price coverage per ticker.

---

## Requirements

Python 3.10 or later and:

```bash
pip install numpy pandas scipy scikit-learn plotly yfinance
```

`yfinance` is optional; without it every series falls back to synthetic data (demo mode).

## Usage

```bash
python COL_Portfolio_architecture.py
```

The report is written to `outputs/` next to the script. Set the `OUTPUT_DIR` environment variable to change the location.

All settings live in the **`PARÁMETROS EDITABLES`** block at the top of the script:
- date range and universe;
- TES curve (nodes or Banrep CSV) and risk-free tenor;
- class bands and per-instrument caps;
- bond-screening criteria;
- lookback and rebalance frequency;
- Ledoit-Wolf on/off;
- liquidity and coverage thresholds;
- curve-volatility parameters and random seed.

## Known limitations

- **Fixed income is modeled, not observed.** Unless you load a real curve history, the TES nodes and bonds follow a simulated path around a fixed base curve, so their returns do not reflect actual market episodes (for example, the 2022 TES sell-off).
- **Bond maturities are static.** Residual maturity and duration are computed once at the start date and do not roll down through the backtest; a bond can stay in the portfolio past its maturity date.
- **Equity μ is a sample mean** (full sample in the strategic optimization, 252-day window in the backtest). It is noisy, and a maximum Sharpe optimizer amplifies that noise.
- **Yahoo Finance data for BVC tickers** fills Colombian holidays with the previous close. Even liquid stocks show about 10% of days with zero return, which is why the illiquidity threshold is set at 25%.

## Disclaimer

This code is for academic and research purposes only and does not constitute investment advice. The default TES curve and bond universe are illustrative; use official Banco de la República and price-vendor data for real analysis.
