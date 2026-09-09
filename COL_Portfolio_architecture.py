from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import interpolate, optimize
from sklearn.covariance import LedoitWolf

import plotly.graph_objects as go
import plotly.io as pio

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
pio.templates.default = "plotly_white"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("QuantPM")


# ==============================================================================
# 0. UTILIDADES Y CONFIGURACIÓN
# ==============================================================================

@dataclass
class AssetUniverse:
    """Define el universo invertible del portafolio de inversión directa."""

    acciones_bvc: List[str] = field(
        default_factory=lambda: [
            "ECOPETROL.CL",     # Ecopetrol
            "TERPEL.CL",        # Organización Terpel
            "PROMIGAS.CL",      # Promigas
            "GRUPOSURA.CL",     # Grupo de Inversiones Suramericana (común)
            "PFGRUPSURA.CL",    # Grupo Sura (preferencial)
            "GRUPOARGOS.CL",    # Grupo Argos (común)
            "PFGRUPOARG.CL",    # Grupo Argos (preferencial)
            "PFDAVVNDA.CL",     # Davivienda (preferencial)
            "CEMARGOS.CL",      # Cementos Argos (común)
            "PFCEMARGOS.CL",    # Cementos Argos (preferencial)
            "PFAVAL.CL",        # Grupo Aval (preferencial)
            "NUTRESA.CL",       # Grupo Nutresa
            "MINEROS.CL",       # Mineros S.A.
            "ISA.CL",           # Interconexión Eléctrica S.A.
            "GEB.CL",           # Grupo Energía Bogotá
            "EXITO.CL",         # Almacenes Éxito
            "ETB.CL",           # Empresa de Telecomunicaciones de Bogotá
            "ENKA.CL",          # Enka de Colombia
            "CELSIA.CL",        # Celsia
            "BVC.CL",           # Bolsa de Valores de Colombia
            "BOGOTA.CL",        # Banco de Bogotá
            "PEI.CL",           # Fideicomiso PEI (títulos inmobiliarios)
            "CIBEST.CL",        # Cibest (holding de Bancolombia, antes Grupo Bolívar/BCOLOMBIA)
            "PFCORFICOL.CL",    # Corficolombiana (preferencial)
            "CONCONCRET.CL",    # Conconcreto
            "CNEC.CL",          # Canacol Energy
            "BHI.CL",           # BHI
            "NUAMCO.CL",        # nuam (holding fusionado BVC + Bolsa de Santiago + Bolsa de Lima)
        ]
    )
    etfs_locales_bvc: List[str] = field(
        default_factory=lambda: ["ICOLCAP.CL", "HCOLSEL.CL", "GXTESCOL.CL"]
    )
    etfs_globales: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "TLT"])
    benchmark: str = "^GSPC"  # proxy de mercado para cálculo de betas

    @property
    def all_tickers(self) -> List[str]:
        return self.acciones_bvc + self.etfs_locales_bvc + self.etfs_globales


# ==============================================================================
# 1. PIPELINE DE DATOS DE MERCADO (RENTA VARIABLE / ETFs)
# ==============================================================================

class MarketDataPipeline:
    """
    Encapsula la descarga y limpieza de precios ajustados diarios.

    Intenta usar yfinance; si falla (sin red, ticker inexistente, rate-limit,
    etc.) recurre a un generador sintético de precios (GBM) calibrado con
    parámetros de mercado plausibles, de modo que el resto del pipeline
    (optimización, backtest) siempre tenga datos con los que trabajar.
    """

    def __init__(
        self,
        tickers: Sequence[str],
        start: Union[str, datetime],
        end: Union[str, datetime],
        seed: int = 42,
    ) -> None:
        self.tickers = list(tickers)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self._rng = np.random.default_rng(seed)
        self.prices_: Optional[pd.DataFrame] = None
        self._common_factor: Optional[np.ndarray] = None  # factor de mercado compartido (lazy)
        self.synthetic_tickers_: List[str] = []  # tickers sin datos reales en el rango solicitado

    # ------------------------------------------------------------------ #
    def fetch_prices(self) -> pd.DataFrame:
        """Descarga precios ajustados de cierre para todos los tickers."""
        frames: Dict[str, pd.Series] = {}
        self.synthetic_tickers_ = []
        for tk in self.tickers:
            series = None
            if _YFINANCE_AVAILABLE:
                try:
                    raw = yf.download(
                        tk,
                        start=self.start,
                        end=self.end,
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                    )
                    if raw is not None and not raw.empty:
                        col = "Close" if "Close" in raw.columns else raw.columns[0]
                        series = raw[col].dropna()
                        if isinstance(series, pd.DataFrame):
                            series = series.iloc[:, 0]
                except Exception as exc:  # pragma: no cover - red externa
                    logger.warning("yfinance falló para %s (%s). Usando fallback sintético.", tk, exc)

            if series is None or series.empty:
                logger.info("Generando serie sintética (GBM) para %s.", tk)
                series = self._synthetic_price_series(tk)
                self.synthetic_tickers_.append(tk)

            frames[tk] = series

        prices = pd.DataFrame(frames)
        prices = prices.sort_index().ffill().dropna(how="all")
        prices = prices.loc[self.start : self.end]
        self.prices_ = prices
        return prices

    # ------------------------------------------------------------------ #
    def _synthetic_price_series(self, ticker: str) -> pd.Series:
        """
        Genera una serie de precios vía Movimiento Browniano Geométrico con un
        factor de mercado común compartido entre activos (más un componente
        idiosincrático propio de cada ticker), de modo que la matriz de
        correlaciones resultante sea realista y útil para la optimización.
        """
        n_days = max((self.end - self.start).days, 252)
        dates = pd.bdate_range(self.start, periods=n_days)

        local_seed = abs(hash(ticker)) % (2**32)
        rng = np.random.default_rng(local_seed)

        is_bond_proxy = ticker.upper() in {"TLT"}
        is_local_equity = ticker.upper().endswith(".CL")
        is_benchmark = ticker.upper() in {"^GSPC", "SPY", "QQQ"}

        if is_bond_proxy:
            mu, sigma, s0, beta_mkt = 0.03, 0.12, 100.0, -0.15  # TLT anticorrelacionado con equities
        elif is_local_equity:
            mu, sigma, s0, beta_mkt = 0.09, 0.28, 25000.0, 0.55
        elif is_benchmark:
            mu, sigma, s0, beta_mkt = 0.11, 0.20, 400.0, 0.95
        else:
            mu, sigma, s0, beta_mkt = 0.10, 0.20, 400.0, 0.70

        dt = 1 / 252
        # Factor de mercado común: se genera una única vez por pipeline y se
        # reutiliza para todos los tickers, garantizando co-movimiento realista.
        if self._common_factor is None or len(self._common_factor) != len(dates):
            self._common_factor = self._rng.normal(0, np.sqrt(dt), size=len(dates))
        common_factor = self._common_factor

        idio_vol = sigma * np.sqrt(max(1 - beta_mkt**2, 0.05))
        idio_shocks = rng.normal(0, idio_vol * np.sqrt(dt), size=len(dates))
        total_shocks = beta_mkt * sigma * common_factor + idio_shocks
        drift = (mu - 0.5 * sigma**2) * dt

        log_path = np.cumsum(drift + total_shocks)
        prices = s0 * np.exp(log_path)
        return pd.Series(prices, index=dates, name=ticker)

    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        """Calcula retornos diarios (log o simples) a partir de precios."""
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        elif method == "simple":
            returns = prices.pct_change()
        else:
            raise ValueError("method debe ser 'log' o 'simple'.")
        return returns.dropna(how="all")


# ==============================================================================
# 2. CURVA CERO CUPÓN DE TES (ETTI) — INTERPOLACIÓN CON CUBIC SPLINE
# ==============================================================================

class TESYieldCurve:
    """
    Representa la Estructura Temporal de Tasas de Interés (ETTI) de TES,
    tal como la publica el Banco de la República, interpolada con splines
    cúbicos para obtener la tasa cero cupón a cualquier plazo residual t
    (en años).
    """

    def __init__(self, tenors_years: Sequence[float], zero_rates: Sequence[float]) -> None:
        """
        Parameters
        ----------
        tenors_years : plazos en años (ej. [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20])
        zero_rates   : tasas cero cupón efectivas anuales (en decimal, ej. 0.095)
        """
        tenors = np.asarray(tenors_years, dtype=float)
        rates = np.asarray(zero_rates, dtype=float)
        order = np.argsort(tenors)
        self.tenors = tenors[order]
        self.rates = rates[order]
        self._spline = interpolate.CubicSpline(
            self.tenors, self.rates, bc_type="natural", extrapolate=True
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_banrep_csv(cls, path: str) -> "TESYieldCurve":
        """
        Construye la curva a partir de un CSV descargado manualmente del
        portal de series estadísticas del Banco de la República
        (columnas esperadas: 'plazo_anios', 'tasa').
        """
        df = pd.read_csv(path)
        required = {"plazo_anios", "tasa"}
        if not required.issubset(df.columns):
            raise ValueError(f"El CSV debe contener las columnas {required}.")
        return cls(df["plazo_anios"].values, df["tasa"].values)

    # ------------------------------------------------------------------ #
    @classmethod
    def synthetic_example(cls) -> "TESYieldCurve":
        """
        Curva ETTI de ejemplo, con forma razonable (empinada en el corto
        plazo, aplanándose en el largo plazo) representativa de un entorno
        de tasas en Colombia. Uso exclusivo para demostración/backtesting
        cuando no se dispone del archivo real de Banrep.
        """
        tenors = np.array([0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20])
        rates = np.array(
            [0.0980, 0.0975, 0.0965, 0.0950, 0.0940, 0.0935, 0.0955, 0.0975, 0.0995, 0.1010, 0.1015]
        )
        return cls(tenors, rates)

    # ------------------------------------------------------------------ #
    def get_rate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Tasa cero cupón interpolada (efectiva anual) para el plazo t (años)."""
        t_arr = np.asarray(t, dtype=float)
        t_clipped = np.clip(t_arr, self.tenors.min(), self.tenors.max())
        rate = self._spline(t_clipped)
        return float(rate) if np.isscalar(t) or t_arr.ndim == 0 else rate

    # ------------------------------------------------------------------ #
    def discount_factor(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Factor de descuento continuo-compuesto-anual: DF = (1+y_t)^(-t)."""
        y_t = self.get_rate(t)
        return (1.0 + y_t) ** (-np.asarray(t, dtype=float))

    # ------------------------------------------------------------------ #
    def plot_curve(self, label: str = "ETTI TES") -> go.Figure:
        """Grafica interactiva (Plotly) de la curva observada (puntos) y la interpolación (línea)."""
        t_fine = np.linspace(self.tenors.min(), self.tenors.max(), 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_fine, y=self.get_rate(t_fine) * 100, mode="lines",
            name=f"{label} (spline)", line=dict(width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=self.tenors, y=self.rates * 100, mode="markers",
            name="Nodos observados", marker=dict(color="crimson", size=9),
        ))
        fig.update_layout(
            title="Estructura Temporal de Tasas de Interés — TES",
            xaxis_title="Plazo (años)", yaxis_title="Tasa cero cupón (% E.A.)",
        )
        return fig


# ==============================================================================
# 4. MÉTRICAS DE RIESGO MICRO (RENTA VARIABLE)
# ==============================================================================

class RiskMetrics:
    """Colección de métricas de riesgo estándar a nivel de activo individual."""

    TRADING_DAYS: int = 252

    @staticmethod
    def annualized_volatility(returns: pd.Series) -> float:
        """Volatilidad anualizada a partir de retornos diarios."""
        return float(returns.std(ddof=1) * np.sqrt(RiskMetrics.TRADING_DAYS))

    @staticmethod
    def annualized_return(returns: pd.Series) -> float:
        """Retorno geométrico anualizado a partir de retornos log diarios."""
        cumulative = np.exp(returns.sum())
        n_years = len(returns) / RiskMetrics.TRADING_DAYS
        return float(cumulative ** (1 / n_years) - 1) if n_years > 0 else 0.0

    @staticmethod
    def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Beta sectorial/de mercado: Cov(asset, market) / Var(market)."""
        aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if aligned.shape[0] < 2:
            return float("nan")
        cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(cov_matrix[0, 1] / cov_matrix[1, 1])

    @staticmethod
    def max_drawdown(price_series: pd.Series) -> float:
        """Máxima caída porcentual desde un máximo histórico (peak-to-trough)."""
        cumulative_max = price_series.cummax()
        drawdown = price_series / cumulative_max - 1.0
        return float(drawdown.min())

    @staticmethod
    def sharpe_ratio(returns: pd.Series, rf: float) -> float:
        """Sharpe Ratio anualizado usando rf anual (decimal) como referencia."""
        rf_daily = (1 + rf) ** (1 / RiskMetrics.TRADING_DAYS) - 1
        excess = returns - rf_daily
        if excess.std(ddof=1) == 0:
            return 0.0
        return float(excess.mean() / excess.std(ddof=1) * np.sqrt(RiskMetrics.TRADING_DAYS))

    @classmethod
    def asset_risk_report(
        cls, prices: pd.DataFrame, returns: pd.DataFrame, market_col: str, rf: float
    ) -> pd.DataFrame:
        """Genera un reporte tabular de métricas de riesgo por activo."""
        rows = []
        for col in returns.columns:
            rows.append(
                {
                    "ticker": col,
                    "vol_anualizada": cls.annualized_volatility(returns[col]),
                    "retorno_anualizado": cls.annualized_return(returns[col]),
                    "beta": cls.beta(returns[col], returns[market_col]) if market_col in returns else np.nan,
                    "max_drawdown": cls.max_drawdown(prices[col]),
                    "sharpe": cls.sharpe_ratio(returns[col], rf),
                }
            )
        return pd.DataFrame(rows).set_index("ticker")


# ==============================================================================
# 5. MOTOR DE OPTIMIZACIÓN MULTI-ALGORITMO
# ==============================================================================

class PortfolioOptimizer:
    """
    Optimización de portafolios por Máximo Sharpe (Markowitz clásico) bajo
    restricción de no-cortaje (long-only, w_i >= 0), suma de pesos = 1, y
    límite máximo por activo, vía scipy.optimize.minimize (SLSQP).

    La matriz de covarianza se estima por defecto con shrinkage de
    Ledoit-Wolf (2004), que combina la covarianza muestral con un target
    estructurado (identidad escalada) usando la intensidad δ óptima en
    sentido de error cuadrático medio:

        Σ_shrunk = (1 - δ)·S + δ·(tr(S)/p)·I

    Esto corrige el mal condicionamiento de S cuando el número de activos
    es grande frente al número de observaciones, que es la causa de que
    Markowitz produzca soluciones de esquina extremas.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        rf: float = 0.0,
        max_weight: float = 0.30,
        min_weight: float = 0.0,
        trading_days: int = 252,
        shrinkage: bool = True,
    ) -> None:
        if returns.empty:
            raise ValueError("La matriz de retornos no puede estar vacía.")
        self.returns = returns.dropna(how="any")
        self.assets = list(self.returns.columns)
        self.n = len(self.assets)
        self.rf = rf
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.trading_days = trading_days
        self.shrinkage = shrinkage

        self.mu_ = self.returns.mean().values * trading_days           # retornos esperados anualizados

        if shrinkage:
            # Ledoit-Wolf se ajusta sobre los retornos diarios y luego se
            # anualiza, no al revés: la intensidad óptima δ se deriva de la
            # dispersión de las observaciones en su escala original.
            lw = LedoitWolf().fit(self.returns.values)
            self.cov_ = lw.covariance_ * trading_days
            self.shrinkage_intensity_ = float(lw.shrinkage_)
        else:
            self.cov_ = self.returns.cov().values * trading_days        # covarianza muestral anualizada
            self.shrinkage_intensity_ = 0.0

    # ------------------------------------------------------------------ #
    # a) MARKOWITZ / MÁXIMO SHARPE
    # ------------------------------------------------------------------ #
    def _portfolio_perf(self, w: np.ndarray) -> Tuple[float, float]:
        ret = float(w @ self.mu_)
        vol = float(np.sqrt(w @ self.cov_ @ w))
        return ret, vol

    def _neg_sharpe(self, w: np.ndarray) -> float:
        ret, vol = self._portfolio_perf(w)
        if vol == 0:
            return 0.0
        return -(ret - self.rf) / vol

    def max_sharpe(self) -> pd.Series:
        """
        Resuelve el portafolio de máximo Sharpe Ratio (tangencia) bajo
        restricciones long-only y límite máximo por activo, usando SLSQP.
        La tasa libre de riesgo (rf) debe derivarse externamente (ej. de la
        IBR overnight o la tasa de referencia de mercado monetario).
        """
        w0 = np.repeat(1 / self.n, self.n)
        bounds = [(self.min_weight, self.max_weight)] * self.n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = optimize.minimize(
            self._neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not result.success:
            logger.warning("Max Sharpe: optimización no convergió (%s). Usando pesos iguales.", result.message)
            w = w0
        else:
            w = result.x
        return pd.Series(w / w.sum(), index=self.assets, name="max_sharpe")

    # ------------------------------------------------------------------ #
    # VISUALIZACIONES
    # ------------------------------------------------------------------ #
    def plot_efficient_frontier(self, n_portfolios: int = 5000, seed: int = 7) -> go.Figure:
        """
        Simula portafolios aleatorios long-only (Monte Carlo) para trazar la
        nube de riesgo-retorno y la frontera eficiente, superponiendo el
        portafolio de Máximo Sharpe (tangencia).
        """
        rng = np.random.default_rng(seed)
        results = np.zeros((3, n_portfolios))
        for i in range(n_portfolios):
            w = rng.dirichlet(np.ones(self.n))
            w = np.minimum(w, self.max_weight)
            w = w / w.sum()
            ret, vol = self._portfolio_perf(w)
            sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
            results[:, i] = [ret, vol, sharpe]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results[1], y=results[0], mode="markers",
            marker=dict(
                size=5, color=results[2], colorscale="Viridis", opacity=0.5,
                colorbar=dict(title="Sharpe"),
            ),
            name="Portafolios simulados", hoverinfo="skip",
        ))

        w_opt = self.max_sharpe()
        ret, vol = self._portfolio_perf(w_opt.values)
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers", name="Máximo Sharpe",
            marker=dict(symbol="star", size=18, color="red", line=dict(width=1, color="white")),
        ))

        fig.update_layout(
            title="Frontera Eficiente — Simulación Monte Carlo",
            xaxis_title="Volatilidad Anualizada", yaxis_title="Retorno Esperado Anualizado",
        )
        return fig


# ==============================================================================
# 6. MOTOR DE BACKTESTING WALK-FORWARD
# ==============================================================================

class WalkForwardBacktester:
    """
    Simula el rebalanceo histórico fuera de muestra de un portafolio:
    en cada fecha de rebalanceo, estima pesos óptimos usando únicamente
    datos hasta ese momento (ventana de lookback) y aplica esos pesos a los
    retornos *futuros* hasta el siguiente rebalanceo (evita look-ahead bias).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        lookback_days: int = 252,
        rebalance_freq: str = "ME",   # 'ME' = fin de mes, 'W' = semanal, 'Q' = trimestral
        rf: float = 0.0,
        max_weight: float = 0.30,
        shrinkage: bool = True,
    ) -> None:
        self.prices = prices.dropna(how="any")
        self.returns = MarketDataPipeline.compute_returns(self.prices, method="log")
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq
        self.rf = rf
        self.max_weight = max_weight
        self.shrinkage = shrinkage

        self.weights_history_: Optional[pd.DataFrame] = None
        self.portfolio_returns_: Optional[pd.Series] = None
        self.equity_curve_: Optional[pd.Series] = None

    # ------------------------------------------------------------------ #
    def _rebalance_dates(self) -> List[pd.Timestamp]:
        dates = pd.date_range(self.returns.index[0], self.returns.index[-1], freq=self.rebalance_freq)
        # Solo fechas con al menos `lookback_days` de historia disponible
        valid = [d for d in dates if (self.returns.index <= d).sum() >= self.lookback_days]
        return valid

    # ------------------------------------------------------------------ #
    def _optimize_weights(self, window_returns: pd.DataFrame) -> pd.Series:
        opt = PortfolioOptimizer(
            window_returns, rf=self.rf, max_weight=self.max_weight, shrinkage=self.shrinkage
        )
        return opt.max_sharpe()

    # ------------------------------------------------------------------ #
    def run(self) -> pd.DataFrame:
        """Ejecuta la simulación walk-forward completa y retorna el historial de pesos."""
        rebal_dates = self._rebalance_dates()
        if len(rebal_dates) < 2:
            raise RuntimeError(
                "Historial insuficiente para el lookback_days especificado; "
                "reduzca lookback_days o amplíe el rango de fechas."
            )

        weight_records: Dict[pd.Timestamp, pd.Series] = {}
        daily_portfolio_returns: List[pd.Series] = []

        for i, reb_date in enumerate(rebal_dates):
            in_sample = self.returns.loc[:reb_date].tail(self.lookback_days)
            try:
                w = self._optimize_weights(in_sample)
            except Exception as exc:  # robustez ante ventanas degeneradas
                logger.warning("Optimización falló en %s (%s). Usando pesos iguales.", reb_date, exc)
                w = pd.Series(1 / self.returns.shape[1], index=self.returns.columns)

            weight_records[reb_date] = w

            next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else self.returns.index[-1]
            out_of_sample = self.returns.loc[reb_date:next_date].iloc[1:]  # excluye el día de rebalanceo
            if not out_of_sample.empty:
                port_ret = out_of_sample[w.index] @ w.values
                daily_portfolio_returns.append(port_ret)

        self.weights_history_ = pd.DataFrame(weight_records).T
        self.portfolio_returns_ = pd.concat(daily_portfolio_returns).sort_index()
        self.equity_curve_ = np.exp(self.portfolio_returns_.cumsum())
        return self.weights_history_

    # ------------------------------------------------------------------ #
    def performance_summary(self) -> Dict[str, float]:
        """Resumen de desempeño del backtest fuera de muestra."""
        if self.portfolio_returns_ is None:
            raise RuntimeError("Ejecute run() antes de solicitar el resumen de desempeño.")
        r = self.portfolio_returns_
        return {
            "retorno_total": float(self.equity_curve_.iloc[-1] - 1),
            "retorno_anualizado": RiskMetrics.annualized_return(r),
            "volatilidad_anualizada": RiskMetrics.annualized_volatility(r),
            "sharpe_ratio": RiskMetrics.sharpe_ratio(r, self.rf),
            "max_drawdown": RiskMetrics.max_drawdown(self.equity_curve_),
            "num_rebalanceos": self.weights_history_.shape[0],
        }

    # ------------------------------------------------------------------ #
    def plot_equity_curve(self) -> go.Figure:
        """Grafica interactiva (Plotly) de la curva de equity acumulada del backtest."""
        if self.equity_curve_ is None:
            raise RuntimeError("Ejecute run() antes de graficar.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.equity_curve_.index, y=self.equity_curve_.values,
            mode="lines", line=dict(width=2, color="darkgreen"), name="Equity",
        ))
        fig.add_hline(y=1.0, line=dict(color="gray", dash="dash", width=1))
        fig.update_layout(
            title="Curva de Equity — Backtest Walk-Forward (Máximo Sharpe)",
            xaxis_title="Fecha", yaxis_title="Valor del Portafolio (base = 1.0)",
        )
        return fig

    # ------------------------------------------------------------------ #
    def plot_weights_evolution(self) -> go.Figure:
        """Grafica interactiva (Plotly) de la evolución de pesos a través de los rebalanceos."""
        if self.weights_history_ is None:
            raise RuntimeError("Ejecute run() antes de graficar.")
        fig = go.Figure()
        for col in self.weights_history_.columns:
            fig.add_trace(go.Scatter(
                x=self.weights_history_.index, y=self.weights_history_[col] * 100,
                mode="lines", stackgroup="pesos", name=col,
            ))
        fig.update_layout(
            title="Evolución de Pesos por Rebalanceo (Máximo Sharpe)",
            xaxis_title="Fecha de rebalanceo", yaxis_title="Peso (%)",
        )
        return fig


# ==============================================================================
# 7. EJECUCIÓN PRINCIPAL — DEMOSTRACIÓN INTEGRAL DEL MÓDULO
# ==============================================================================

def main() -> None:
    OUTPUT_DIR = os.environ.get(
        "OUTPUT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 78)
    logger.info("QUANT PM — INVERSIÓN DIRECTA EN ACTIVOS: DEMO DE EJECUCIÓN")
    logger.info("=" * 78)

    # ---------------------------------------------------------------- #
    # 1) UNIVERSO Y PIPELINE DE DATOS
    # ---------------------------------------------------------------- #
    universe = AssetUniverse()
    start_date, end_date = "2021-01-01", "2024-12-31"

    logger.info("Descargando/generando precios para: %s", universe.all_tickers)
    pipeline = MarketDataPipeline(universe.all_tickers + [universe.benchmark], start_date, end_date)
    prices = pipeline.fetch_prices()
    returns = pipeline.compute_returns(prices, method="log")
    logger.info("Precios obtenidos: %d filas x %d activos.", *prices.shape)

    # Universo "conjunto" para optimización multi-activo y backtest: excluye
    # tickers sin datos reales en el rango (fallback sintético) y tickers con
    # historia real más corta que start_date (p.ej. activos listados
    # recientemente), ya que un dropna(how="any") conjunto colapsaría la
    # ventana común de todo el portafolio a la fecha del activo más joven.
    min_start_buffer = pd.Timestamp(start_date) + pd.Timedelta(days=45)
    short_history = [
        tk for tk in universe.all_tickers
        if prices[tk].first_valid_index() is None or prices[tk].first_valid_index() > min_start_buffer
    ]
    excluded_joint = sorted(set(pipeline.synthetic_tickers_) | set(short_history))
    if excluded_joint:
        logger.warning(
            "Excluidos de la optimización conjunta y el backtest (%s–%s) por no tener "
            "historia real completa: %s. Se mantienen en el reporte de riesgo individual.",
            start_date, end_date, excluded_joint,
        )
    joint_tickers = [tk for tk in universe.all_tickers if tk not in excluded_joint]

    # ---------------------------------------------------------------- #
    # 2) CURVA TES — ORIGEN DE LA TASA LIBRE DE RIESGO
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("CURVA TES — TASA LIBRE DE RIESGO")
    curve = TESYieldCurve.synthetic_example()

    # La tasa libre de riesgo se lee de la curva TES al plazo que corresponde
    # al horizonte de inversión, en vez de fijarse a dedo. El plazo importa:
    # la curva colombiana no es plana, y `rf` entra directamente en el Sharpe
    # que optimizamos, así que la elección de tenor mueve el resultado.
    # 1 año es coherente con métricas anualizadas; súbelo si el horizonte
    # real de la estrategia es más largo.
    RF_TENOR_YEARS = 1.0
    rf_tes = float(curve.get_rate(RF_TENOR_YEARS))
    logger.info(
        "Tasa libre de riesgo: TES a %.2g año(s) = %.4f%% E.A. (curva cero cupón interpolada)",
        RF_TENOR_YEARS, rf_tes * 100,
    )

    # ---------------------------------------------------------------- #
    # 3) MÉTRICAS DE RIESGO MICRO
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("MÉTRICAS DE RIESGO MICRO POR ACTIVO")
    equity_returns = returns[joint_tickers]
    risk_report = RiskMetrics.asset_risk_report(prices, returns, universe.benchmark, rf_tes)
    print("\n" + risk_report.round(4).to_string())

    # ---------------------------------------------------------------- #
    # 4) OPTIMIZACIÓN — MÁXIMO SHARPE
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("OPTIMIZACIÓN DE PORTAFOLIO — MÁXIMO SHARPE (covarianza Ledoit-Wolf)")
    optimizer = PortfolioOptimizer(equity_returns, rf=rf_tes, max_weight=0.35, shrinkage=True)
    weights_opt = optimizer.max_sharpe()
    ret_opt, vol_opt = optimizer._portfolio_perf(weights_opt.values)
    print("\n" + (weights_opt * 100).round(2).to_string(), "  (% del portafolio)")
    logger.info(
        "Intensidad de shrinkage Ledoit-Wolf: δ = %.4f (0 = covarianza muestral pura, 1 = target)",
        optimizer.shrinkage_intensity_,
    )
    logger.info(
        "Portafolio óptimo — retorno esp.: %.2f%% | volatilidad: %.2f%% | Sharpe: %.4f | activos con peso > 0: %d",
        ret_opt * 100, vol_opt * 100, (ret_opt - rf_tes) / vol_opt, int((weights_opt > 1e-4).sum()),
    )

    # Gráficas interactivas (Plotly) — se combinan al final en un solo HTML.
    charts: List[Tuple[str, go.Figure]] = []
    charts.append(("Frontera Eficiente", optimizer.plot_efficient_frontier(n_portfolios=3000)))
    charts.append(("Curva TES", curve.plot_curve()))

    # ---------------------------------------------------------------- #
    # 5) BACKTEST WALK-FORWARD
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("BACKTEST WALK-FORWARD FUERA DE MUESTRA")
    backtester = WalkForwardBacktester(
        prices[joint_tickers],
        lookback_days=252,
        rebalance_freq="ME",
        rf=rf_tes,
        max_weight=0.35,
    )
    weights_history = backtester.run()
    perf_summary = backtester.performance_summary()
    print("\n" + (weights_history * 100).round(2).to_string(), "  (% del portafolio por rebalanceo)")
    print("\n" + pd.Series(perf_summary, name="valor").to_string())

    charts.append(("Curva de Equity — Backtest", backtester.plot_equity_curve()))
    charts.append(("Evolución de Pesos — Backtest", backtester.plot_weights_evolution()))

    # ---------------------------------------------------------------- #
    # 6) REPORTE HTML INTERACTIVO (todas las gráficas en una sola página)
    # ---------------------------------------------------------------- #
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Quant PM — Reporte Interactivo</title>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;",
        "padding:24px 32px 64px;background:#fafafa;color:#1a1a1a;}",
        "h1{margin-bottom:4px;} .subtitle{color:#666;margin-top:0;}",
        "section{margin-top:48px;} h2{border-bottom:2px solid #ddd;padding-bottom:8px;}",
        "</style></head><body>",
        "<h1>Quant PM — Reporte Interactivo</h1>",
        f"<p class='subtitle'>Universo: {len(universe.all_tickers)} activos | "
        f"Rango: {start_date} a {end_date}</p>",
    ]
    for i, (title, fig) in enumerate(charts):
        include_js = "cdn" if i == 0 else False
        html_parts.append(f"<section><h2>{title}</h2>")
        html_parts.append(pio.to_html(fig, include_plotlyjs=include_js, full_html=False))
        html_parts.append("</section>")
    html_parts.append("</body></html>")

    report_path = os.path.join(OUTPUT_DIR, "reporte_interactivo.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info("=" * 78)
    logger.info("Ejecución completa. Reporte interactivo: %s", report_path)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
