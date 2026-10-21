# Colombian Portfolio Architecture

A single Python script, [`COL_Portfolio_architecture.py`](COL_Portfolio_architecture.py), that runs a **multi-asset allocation engine** for the Colombian market. It combines equities (BVC stocks and ETFs) and fixed income (TES curve nodes, bond ETFs and individual bonds) in one return/covariance matrix. It finds the optimal portfolio with the latest available data, so it can be used to decide what to buy today. The criterion is selectable: maximum Sharpe, minimum variance, maximum return for a target volatility, minimum CVaR, or maximum return for a target CVaR. By default the optimizer is free (long-only, fully invested): the RV/RF split is a result, not an input. Strategic bands and weight caps are optional. An out-of-sample walk-forward backtest is available as a separate validation mode.

It prints tables to the console and writes an interactive Plotly report to `outputs/reporte_interactivo.html`. The report is short and visual: the recommended portfolio with its measures (expected return, volatility, Sharpe, CVaR, RV/RF split) and amounts in COP, the efficient frontier and the TES curve.

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
| Fixed income (RF) | C: individual bonds | The bonds you enter in `BONOS_UNIVERSO` (empty by default) that pass the screening (see below) |
| — | Benchmark (for betas) | `^GSPC` |

Not every listed ticker ends up in the optimization. See the data rules in step 1.

## How it works

The script runs from `main()` in eight steps. Everything you can edit sits in the **CONFIGURACIÓN (set up)** block at the top of the file.

**Two execution modes** (`MODO_EJECUCION`):
- `"recomendacion"` (default): computes the optimal portfolio at the last available date, for investing today. Steps 1–6 and 8.
- `"validacion"`: the same, plus the walk-forward backtest (step 7) to check how the method would have behaved in the past.

### 1. Market data and universe selection (`MarketDataPipeline`)
- Downloads daily adjusted close prices from **Yahoo Finance** (`yfinance`). The window ends at `FECHA_FIN` (`None` = today) and starts `HISTORIA_ANIOS` (8) earlier, unless `FECHA_INICIO` is set. It must cover the CVaR window; in validation mode what remains is the backtest period (a backtest with CVaR criteria takes several minutes).
- If a ticker fails to download, it generates a **synthetic series** (geometric Brownian motion with a common market factor plus an idiosyncratic shock). The seed is stable per ticker (CRC32), so runs are reproducible.
- When calendars are aligned, a missing price is carried forward for **at most 5 days**. A security listed after the start date stays as missing data before its first price, without truncating the matrix.
- A ticker with only a synthetic series is excluded from the whole run (`EXCLUIR_ACTIVOS_SIN_PRECIO_REAL`).
- Every other filter is evaluated **at each rebalance, using only the prices in that rebalance's estimation window** (`LOOKBACK_DIAS`). The full sample is never used, because that would tell the 2022 backtest which stocks later become illiquid or stop trading (look-ahead bias). A ticker is not investable on a date when any of these holds:
  - its history does not cover the estimation window;
  - it has no quote in the last 5 business days;
  - it is **illiquid** within the window: more than 25% of its days have exactly zero return, or its price stays frozen for more than 30 days in a row. This is measured on the ticker's own calendar. Stale prices understate volatility and correlation, and a Sharpe optimizer rewards exactly that.
- The strategic portfolio applies the same filters at the closing date, over its own estimation window.
- Exclusions at the closing date are logged with their reason. The HTML report lists, per ticker, how many rebalances it was eligible for and its most frequent exclusion reason.
- If fewer than 5 equities with real prices remain (for example, with no network access) the run **stops** with a clear message. Only with `PERMITIR_DATOS_SINTETICOS = True` does it switch to a **demo mode** on synthetic series, for testing the code without internet; those results are not market inference.
- **Freshness alert:** if the last market price is more than `MAX_DIAS_DATOS_DESACTUALIZADOS` days before the closing date, the run stops (or warns, in demo mode). In recommendation mode it also warns when `FECHA_FIN` is not today.

### 2. TES yield curve (`TESYieldCurve`)
- Models the zero-coupon term structure (ETTI) of Colombian government bonds, interpolated with a natural cubic spline.
- Provides interpolated rates, discount factors and par yields.
- **Real daily history (default, `CURVA_TES_FUENTE = "banrep"`).** Downloads the zero-coupon TES rates in pesos at 1, 5 and 10 years published by Banco de la República on its SUAMECA portal (`BanrepTESHistory`). Each day's curve is completed to every tenor with a Nelson-Siegel fit (τ = 1.37 years, Diebold-Li), which passes exactly through the three published nodes.
  - The data is cached in `datos/tes_cero_cupon_banrep.csv` and reused when it covers the window.
  - The SUAMECA server does not send its intermediate TLS certificate. The loader fetches it from the address in the certificate and still requires the chain to end at a trusted root.
  - The curve must reach the closing date: if the download fails, or the cache ends more than `MAX_DIAS_DATOS_DESACTUALIZADOS` days before it, the run **stops**, because the risk-free rate and all fixed-income returns come from this curve. Only with `PERMITIR_DATOS_SINTETICOS = True` does it fall back to the simulated curve, and it says so in the log and the report.
- **Simulated (`"simulada"`).** A fixed curve from `CURVA_TES_PLAZOS` / `CURVA_TES_TASAS` (or a Banrep CSV snapshot in `RUTA_CURVA_TES_CSV`) with simulated daily shocks.
- The risk-free rate is the **1-year TES rate** (`RF_TENOR_YEARS`) read from the curve in force on each date: at the close for the strategic portfolio, at each rebalance in the backtest, and averaged over the window for per-asset Sharpe ratios.

### 3. Fixed-income engine (`FixedIncomeEngine`)
The fixed-income leg combines three approaches:

- **A. TES curve nodes (`TESNodeBuilder`).** Each node is a par-coupon bond at its tenor. Its daily return is carry plus the second-order price change ΔP/P ≈ −D·Δy + ½·C·(Δy)².
- **B. Bond ETFs / funds.** Vehicles with an observable market price, taken from the price pipeline.
- **C. Individual bonds (`BondScreener`).** The bonds **you enter** in `BONOS_UNIVERSO` (ISIN, issuer, rating, coupon, maturity and one of YTM / dirty price / spread over TES, quoted at the closing date). The list is empty by default; the configuration block explains which bonds to look for and what each field means. A policy filter then applies:
  - rating ≥ AA+ on the local scale;
  - residual maturity between 1 and 10 years;
  - liquidity score ≥ 0.35;
  - spread over the TES curve ≤ 450 bp;
  - fixed-rate peso bonds only (no UVR/IPC);
  - at most 2 bonds per issuer.

Individual bonds **age through time**:
- **Daily repricing.** Each bond is priced from its remaining cash flows. The yield is the day's curve at the bond's current residual maturity plus a credit spread that follows a random walk scaled by illiquidity. Returns include coupons, and the price pulls to par.
- **Dynamic screening.** At every rebalance, the time-dependent criteria (maturity band, spread, YTM, issuer limit) are re-applied with that date's data. A bond enters the investable universe when it falls inside the band and leaves it when its remaining maturity drops below the 1-year minimum, so it never reaches maturity while held.
- **Pro-forma covariance.** Inside each estimation window, a bond's returns are rebuilt by applying the historical curve and spread moves to its *current* maturity, duration and convexity. Without this, the window would reflect the longer duration the bond used to have and overstate its risk.
- The screening table is evaluated at the closing date, where the bonds are quoted. The credit spread path is anchored there: on the closing date it equals the spread implied by your quote, and backwards in time it is a simulated reconstruction. In validation mode the report adds a table with each bond's eligible rebalances.

TES nodes behave like par bonds of constant maturity rolled daily. Their carry, roll-down, duration and convexity come from the previous day's curve, so they track the rate level through time.

Bonds in pesos are defined by their **spread over the TES curve**, so their yields stay consistent with whichever curve is loaded.

With the simulated curve, daily changes come from a **three-factor Nelson-Siegel model** (level, slope, curvature) with mean reversion and a link to equity returns (`CurveShockGenerator`).

The expected return (μ) of A and C is **analytical** (YTM + roll-down), not a sample mean. It is recomputed from the curve in force at each rebalance date, which is the tactical part of the allocation.

### 4. Unified multi-asset matrix
Assets quoted in dollars (`ACTIVOS_EN_USD`: SPY, QQQ and TLT by default) are converted to pesos with the daily USD/COP rate (`TICKER_USDCOP`, Yahoo Finance `COP=X`), so their returns and risk include the exchange-rate effect, as they do for an investor in pesos. If the rate cannot be downloaded, those assets are left out and the report says so. The S&P 500 benchmark used for betas stays in dollars.

Equity prices, bond-ETF prices and the modeled fixed-income series are aligned by date into one price and return matrix.

### 5. Per-asset risk metrics (`RiskMetrics`)
For each asset: annualized volatility, annualized geometric return, beta against the S&P 500, maximum drawdown, Sharpe ratio and modified duration (fixed income).

### 6. Optimal portfolio at the closing date (`PortfolioOptimizer`, `ObjectiveSpec`, `AllocationPolicy`)
**Criteria** (`CRITERIO_OPTIMIZACION`):
- `max_sharpe`: best return per unit of volatility (convex-QP reformulation, below).
- `min_varianza`: the least volatile portfolio.
- `max_retorno_vol`: highest expected return with annual volatility at most `VOLATILIDAD_OBJETIVO_ANUAL` (a target variance, expressed as volatility).
- `min_cvar`: lowest CVaR. A linear program (Rockafellar-Uryasev) over historical scenarios.
- `max_retorno_cvar`: highest expected return with CVaR at most `CVAR_OBJETIVO_MENSUAL`. Same linear program.

**CVaR scenarios and reliability.** The tail needs far more data than the mean and covariance, so it has its own, longer window (`CVAR_VENTANA_ANIOS`, 5 years; mean and covariance use `LOOKBACK_DIAS`, 2 years). Monthly scenarios (`HORIZONTE_DIAS`, 21 = one month) are simulated with a **stationary block bootstrap** of the continuous daily (or weekly) log returns of that window, re-centred on the expected return the optimizer uses (`CVAR_CENTRAR_EN_MU`): blocks of real days keep correlations and volatility clusters, and the CVaR rests on about 1,260 daily observations instead of about 60 overlapping months. The level is `CVAR_NIVEL` (95%) and `CVAR_ESCENARIOS` paths are simulated. The program checks that at least `CVAR_MIN_OBS_COLA` observations (30) fall in the tail (about 2.4 years of daily data, more than 11 years of weekly data), and the report shows a 90% confidence interval of the portfolio's CVaR, obtained by resampling the history: a wide interval means the history does not pin the tail down. Assets without the full CVaR window are left out and listed in the alerts.

**Number of assets** (section 9 of the configuration). `MAX_ACTIVOS` (12) caps the number of assets and `PESO_MIN_POSICION` (3%) sets the smallest position; both are solved with a heuristic (solve the free optimum, then remove the smallest-weight assets one by one and re-optimize), which gives good but not provably optimal solutions because the problem is combinatorial. `MIN_ACTIVOS` (5) forces at least that many assets by capping each at 1/N (20%). With `CARDINALIDAD_AUTOMATICA` the program picks, between the minimum and the maximum, the portfolio with the fewest assets whose objective is within `CARDINALIDAD_TOLERANCIA` (2%) of the best, and the report plots the objective against the number of assets. In-sample, more assets never hurt the objective, so an "optimal number" only makes sense with a tolerance like this one.

**Reachable range and interactive choice.** Before optimizing, the program shows a table of reference portfolios (computed with the minimum-assets cap) (minimum variance, minimum CVaR, maximum Sharpe, maximum return) with their measures; the minimum and the maximum-return rows delimit the reachable volatility and CVaR. In an interactive terminal it then asks for the criterion and the target (Enter keeps what is configured; `PREGUNTAR_ANTES_DE_OPTIMIZAR = False` disables the questions). A target left empty takes the middle of the range, and one outside the range is replaced by the closest feasible value, with a warning.
Long-only and fully invested, for the criterion chosen in section 8 of the configuration (`CRITERIO_OPTIMIZACION`). **By default there are no other restrictions:** the optimizer picks the assets that best serve the criterion, and the RV/RF split comes out of the result (the report shows it by class and sub-class, with amounts in COP if `CAPITAL_INVERSION_COP` is set). Without caps the optimum tends to concentrate in few assets, so the report also shows the number of assets and the effective number (1/HHI), and raises an alert when it is very concentrated. Optional constraints, off by default (section 9 of the configuration):
- **strategic bands per asset class** (`BANDAS_POR_CLASE`), e.g. RV 40%–60% and RF 40%–60%;
- **a general weight cap per asset** (`MAX_PESO_POR_ACTIVO`) and **caps per instrument type** (`TOPES_POR_TIPO`: stock, equity ETF, TES node, bond ETF, individual bond).

Other details:
- The covariance is estimated with **Ledoit-Wolf shrinkage** (scikit-learn), and the shrinkage intensity δ is reported.
- Because moments are estimated on log returns, fixed-income μ and the risk-free rate are both converted to log scale, ln(1 + r).
- **Equity and ETF μ use a Bayes-Stein estimator** (Jorion, 1986; `ESTIMADOR_MU`). Within each asset class, sample means are shrunk toward the mean of that class's minimum-variance portfolio, with an intensity φ estimated from the data. It approaches 1 when differences between sample means are mostly noise. φ is logged and averaged over the backtest. Set `ESTIMADOR_MU = "muestral"` to use plain sample means.
- The Sharpe ratio is not concave in the weights, so the problem is **reformulated as a convex QP** (Cornuejols and Tütüncü, 2006): minimize u'Σu subject to (μ−rf)'u = c, with caps and bands (if any) scaled by Σu, and w = u/Σu. SciPy SLSQP then reaches the global optimum from a single starting point, the feasible portfolio with the highest excess return (found with `linprog`).
- **When no feasible portfolio beats the risk-free rate** (it happens in 16 of the 39 optimizations of the default 2021–2024 run), the Sharpe-optimal portfolio does not exist, and the engine returns the **minimum-variance portfolio** inside the caps and bands (if any). If the solver fails, the last resort is the feasible point closest to equal weights.
- Infeasible bands or caps are detected before calling the solver.
- The frontier plot traces the maximum return for each volatility level inside the active constraints, with the individual assets, the reference portfolios and the chosen one.

### 7. Walk-forward backtest (`WalkForwardBacktester`, `"validacion"` mode only)
- Rebalances on the **last trading day of each month** (`FRECUENCIA_REBALANCEO`), using only the previous `LOOKBACK_DIAS` trading days (504, about 2 years). The weights are applied to the following period, which avoids look-ahead bias.
- Each rebalance uses the tactical fixed-income μ and the risk-free rate read from the curve at that date.
- Daily portfolio returns are aggregated from **simple returns**, then converted to log returns.
- Reports total and annualized return, volatility, Sharpe ratio, maximum drawdown, number of rebalances, average weight per class, and, if bands are active, whether they were respected at every rebalance.

### 8. HTML report
A short page, with the central information first:
- **Alerts** (stale data, demo mode, concentration, a target outside the reachable range, `FECHA_FIN` in the past).
- **Measure cards:** expected return, volatility, Sharpe, CVaR and VaR at the horizon, RV/RF split, number of assets and the effective number, risk-free rate and capital.
- **The recommended portfolio:** table with weights, contribution to risk and amounts in COP; a bar chart of weight against risk contribution; and a ring chart by class and sub-class.
- **Objective against number of assets** (when it varies), the **reachable range** table and the **efficient frontier**.
- **Fixed income:** the TES curve at the closing date and, if you entered bonds, their screening table.
- **Validation mode only:** the equity curve, the RV/RF allocation over time and the out-of-sample performance.
- **Collapsed:** assets that are not investable at the close, CVaR reliability (years of history, tail observations, confidence interval), and methodology and limits.

---

## Requirements

Python 3.10 or later and:

```bash
pip install numpy pandas scipy scikit-learn plotly yfinance
```

`yfinance` is needed for real prices. Without it (or without internet) the run stops, unless `PERMITIR_DATOS_SINTETICOS = True`, which switches to demo mode on synthetic data.

## Usage

```bash
python COL_Portfolio_architecture.py
```

The report is written to `outputs/` next to the script. Set the `OUTPUT_DIR` environment variable to change the location.

All settings live in the **CONFIGURACIÓN (set up)** block at the top of the script, in 14 numbered sections with a comment on each line:
1. execution mode (`recomendacion` / `validacion`) and capital in COP;
2. date window (`FECHA_FIN = None` means today);
3. real-data policy and data-quality alerts;
4. equity universe;
5. fixed-income ETFs and TES nodes;
6. **individual bonds (`BONOS_UNIVERSO`)**, with a guide on which bonds to look for, the fields of each bond and the screening criteria;
7. TES curve source, cache path and risk-free tenor;
8. **optimization criterion and targets** (Sharpe, variance, CVaR); CVaR level, horizon, history window, scenarios and return frequency; and whether to ask before optimizing;
9. portfolio constraints: optional bands and weight caps, and **minimum / maximum number of assets** and minimum position;
10. estimation window for return and covariance (2 years), Ledoit-Wolf and the expected-return estimator;
11. backtest frequency and frontier size;
12. liquidity and coverage thresholds;
13. simulation parameters and random seed;
14. output location.

**To invest today:** leave `FECHA_FIN = None`, `MODO_EJECUCION = "recomendacion"` and `PERMITIR_DATOS_SINTETICOS = False`, set your capital, and fill `BONOS_UNIVERSO` with the quotes of that day.

## Known limitations

- **Only three curve nodes are observed.** Banrep publishes the 1, 5 and 10-year rates; tenors outside that range (below 1 year, above 10) are Nelson-Siegel extrapolations.
- **A sovereign bond's rate is the curve's.** Enter TES with `"spread_bp": 0` (they follow the closing-date curve) or with a `ytm` of that same date; a stale rate inflates the bond's expected return and biases the optimizer toward it.
- **Credit-spread history is simulated.** Individual bonds follow the real TES curve and their spread is anchored to your quote at the closing date, but before that date it is a random-walk reconstruction (the daily spread volatility is `BONOS_VOL_SPREAD_BP_DIARIA`); there is no public source of historical prices per bond. A bond is also assumed to have existed throughout the window.
- **Bayes-Stein reduces, but does not remove, estimation error in μ.** It still relies on a 2-year window (`LOOKBACK_DIAS`), and it assumes a common mean within each class is a sensible target.
- **One historical path.** The backtest covers a single three-year period, so a better estimator is not guaranteed to produce a better result in it.
- **Yahoo Finance data for BVC tickers** fills Colombian holidays with the previous close. Even liquid stocks show about 10% of days with zero return, which is why the illiquidity threshold is set at 25%.

## Disclaimer

This code is for academic and research purposes only and does not constitute investment advice. The bonds are whatever you enter, and the simulated-curve fallback is illustrative; use official Banco de la República and price-vendor data for real analysis.
