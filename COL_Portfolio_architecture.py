from __future__ import annotations

# ==============================================================================
# IMPORTACIONES Y CONFIGURACIÓN GLOBAL
# ==============================================================================

# --- Librería estándar --------------------------------------------------------
import logging
import os
import ssl
import warnings
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

# --- Cómputo numérico y estadístico ------------------------------------------
import numpy as np
import pandas as pd
from scipy import interpolate, optimize
from sklearn.covariance import LedoitWolf

# --- Visualización ------------------------------------------------------------
import plotly.graph_objects as go
import plotly.io as pio

# --- Fuente de datos opcional (yfinance) -------------------------------------
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
except ImportError:
    _YFINANCE_AVAILABLE = False

# --- Avisos y tema gráfico ----------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
pio.templates.default = "plotly_white"

# --- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("QuantPM")

# --- Constantes globales ------------------------------------------------------
TRADING_DAYS: int = 252


# ==============================================================================
#                        ⚙  PARÁMETROS EDITABLES DEL MOTOR
# ------------------------------------------------------------------------------

# --- Ventana de análisis ------------------------------------------------------
FECHA_INICIO: str = "2021-01-01"
FECHA_FIN: str = "2024-12-31"

# --- Universo de Renta Variable ----------------------------------------------
ACCIONES_BVC: List[str] = [
    "ECOPETROL.CL",
    "TERPEL.CL",
    "PROMIGAS.CL",
    "GRUPOSURA.CL",
    "PFGRUPSURA.CL",
    "GRUPOARGOS.CL",
    "PFGRUPOARG.CL",
    "PFDAVVNDA.CL",
    "CEMARGOS.CL",
    "PFCEMARGOS.CL",
    "PFAVAL.CL",
    "NUTRESA.CL",
    "MINEROS.CL",
    "ISA.CL",
    "GEB.CL",
    "EXITO.CL",
    "ETB.CL",
    "ENKA.CL",
    "CELSIA.CL",
    "BVC.CL",
    "BOGOTA.CL",
    "PEI.CL",
    "CIBEST.CL",
    "PFCORFICOL.CL",
    "CONCONCRET.CL",
    "CNEC.CL",
    "BHI.CL",
    "NUAMCO.CL",
]
ETFS_RV_LOCALES: List[str] = ["ICOLCAP.CL", "HCOLSEL.CL"]
ETFS_RV_GLOBALES: List[str] = ["SPY", "QQQ"]

# --- Universo de Renta Fija ---------------------------------------------------
ETFS_RF_LOCALES: List[str] = []
ETFS_RF_GLOBALES: List[str] = ["TLT"]

NODOS_TES: List[float] = [1.0, 3.0, 5.0, 10.0]

BENCHMARK: str = "^GSPC"

# --- Curva cero cupón de TES (ETTI) ------------------------------------------
CURVA_TES_PLAZOS: List[float] = [0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]
CURVA_TES_TASAS: List[float] = [
    0.0980, 0.0975, 0.0965, 0.0950, 0.0940, 0.0935, 0.0955, 0.0975, 0.0995, 0.1010, 0.1015
]
RUTA_CURVA_TES_CSV: Optional[str] = None

CURVA_TES_FUENTE: str = "banrep"
CURVA_TES_TAU_NS: float = 1.37
RUTA_CACHE_CURVA_TES: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datos", "tes_cero_cupon_banrep.csv"
)

RF_TENOR_YEARS: float = 1.0

# --- Bandas estratégicas por clase de activo ---------------------------------
BANDA_RV: Tuple[float, float] = (0.40, 0.60)
BANDA_RF: Tuple[float, float] = (0.40, 0.60)

# --- Topes de concentración por tipo de instrumento --------------------------
MAX_PESO_ACCION_INDIVIDUAL: float = 0.15
MAX_PESO_ETF_RV: float = 0.20
MAX_PESO_ETF_RF: float = 0.20
MAX_PESO_NODO_TES: float = 0.20
MAX_PESO_BONO_INDIVIDUAL: float = 0.15

# --- Screening de bonos individuales (Enfoque C) ------------------------------
BONOS_MIN_RATING: str = "AA+"
BONOS_PLAZO_MIN_ANIOS: float = 1.0
BONOS_PLAZO_MAX_ANIOS: float = 10.0
BONOS_MIN_LIQUIDEZ: float = 0.35
BONOS_MAX_SPREAD_BP: Optional[float] = 450.0
BONOS_MAX_POR_EMISOR: int = 2
BONOS_INDEXACIONES_PERMITIDAS: Optional[Set[str]] = {"TF"}
BONOS_EMISORES_EXCLUIDOS: Set[str] = set()

# --- Backtest walk-forward ----------------------------------------------------
LOOKBACK_DIAS: int = 252
FRECUENCIA_REBALANCEO: str = "ME"
USAR_SHRINKAGE_LEDOIT_WOLF: bool = True
ESTIMADOR_MU: str = "bayes_stein"

# --- Política de datos --------------------------------------------------------
EXCLUIR_ACTIVOS_SIN_PRECIO_REAL: bool = True
MIN_ACTIVOS_RV_REALES: int = 5
SEMILLA_ALEATORIA: int = 42

# --- Liquidez y cobertura de precios -----------------------------------------
MAX_PCT_DIAS_SIN_MOVIMIENTO: float = 0.25
MAX_DIAS_PRECIO_CONGELADO: int = 30
MAX_DIAS_RELLENO_PRECIO: int = 5
HOLGURA_COBERTURA_DIAS: int = 45

# --- Volatilidad de la curva (modelo de 3 factores, puntos básicos diarios) ---
CURVA_VOL_NIVEL_BP: float = 5.5
CURVA_VOL_PENDIENTE_BP: float = 3.5
CURVA_VOL_CURVATURA_BP: float = 2.5
CURVA_BETA_RV_BP: float = -1.2

# --- Salida -------------------------------------------------------------------
DIRECTORIO_SALIDA: str = os.environ.get(
    "OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
)

# ==============================================================================
#                     FIN DE LOS PARÁMETROS EDITABLES
# ==============================================================================


# ==============================================================================
# 0. TAXONOMÍA DE ACTIVOS, UNIVERSO Y CONFIGURACIÓN
# ==============================================================================

class AssetClass(str, Enum):
    """Clase de activo de primer nivel (define las bandas estratégicas)."""

    RENTA_VARIABLE = "RV"
    RENTA_FIJA = "RF"


class AssetSubClass(str, Enum):
    """Sub-clase (define el desglose interno del reporte)."""

    RV_ACCION_LOCAL = "RV — Acciones BVC"
    RV_ETF_LOCAL = "RV — ETFs locales"
    RV_ETF_GLOBAL = "RV — ETFs globales"
    RF_NODO_TES = "RF — Nodos TES (Enfoque A)"
    RF_ETF = "RF — ETFs / FICs (Enfoque B)"
    RF_BONO = "RF — Bonos individuales (Enfoque C)"


@dataclass
class AssetSpec:
    """
    Ficha de un activo invertible dentro de la matriz unificada.

    `expected_return` sólo se completa para activos cuyo μ se deriva
    analíticamente (RF: YTM + roll-down). Para el resto se estima con la media
    muestral de los retornos.
    """

    ticker: str
    name: str
    asset_class: AssetClass
    sub_class: AssetSubClass
    expected_return: Optional[float] = None
    modified_duration: Optional[float] = None
    convexity: Optional[float] = None
    ytm: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "nombre": self.name,
            "clase": self.asset_class.value,
            "sub_clase": self.sub_class.value,
            "mu_analitico": self.expected_return,
            "ytm": self.ytm,
            "duracion_mod": self.modified_duration,
            "convexidad": self.convexity,
            **{k: v for k, v in self.metadata.items() if not isinstance(v, (list, dict))},
        }


@dataclass
class AssetUniverse:
    """
    Define el universo invertible. Los valores por defecto se leen del bloque
    de PARÁMETROS EDITABLES en la cabecera del archivo; se pueden sobrescribir
    al instanciar (p. ej. `AssetUniverse(acciones_bvc=[...])`).
    """

    acciones_bvc: List[str] = field(default_factory=lambda: list(ACCIONES_BVC))
    etfs_rv_locales: List[str] = field(default_factory=lambda: list(ETFS_RV_LOCALES))
    etfs_rv_globales: List[str] = field(default_factory=lambda: list(ETFS_RV_GLOBALES))

    # --- Enfoque B: vehículos colectivos de renta fija con precio de mercado ---
    etfs_rf_locales: List[str] = field(default_factory=lambda: list(ETFS_RF_LOCALES))
    etfs_rf_globales: List[str] = field(default_factory=lambda: list(ETFS_RF_GLOBALES))

    # --- Enfoque A: nodos de duración de la curva soberana (años) ---
    nodos_tes: List[float] = field(default_factory=lambda: list(NODOS_TES))

    benchmark: str = field(default_factory=lambda: BENCHMARK)

    # ------------------------------------------------------------------ #
    @property
    def rv_tickers(self) -> List[str]:
        return self.acciones_bvc + self.etfs_rv_locales + self.etfs_rv_globales

    @property
    def rf_etf_tickers(self) -> List[str]:
        return self.etfs_rf_locales + self.etfs_rf_globales

    @property
    def all_tickers(self) -> List[str]:
        """Todos los tickers con precio de mercado descargable (RV + ETFs de RF)."""
        return self.rv_tickers + self.rf_etf_tickers

    # ------------------------------------------------------------------ #
    def market_specs(self) -> Dict[str, AssetSpec]:
        """Fichas de los activos que provienen de precios de mercado."""
        specs: Dict[str, AssetSpec] = {}
        buckets = [
            (self.acciones_bvc, AssetClass.RENTA_VARIABLE, AssetSubClass.RV_ACCION_LOCAL),
            (self.etfs_rv_locales, AssetClass.RENTA_VARIABLE, AssetSubClass.RV_ETF_LOCAL),
            (self.etfs_rv_globales, AssetClass.RENTA_VARIABLE, AssetSubClass.RV_ETF_GLOBAL),
            (self.etfs_rf_locales, AssetClass.RENTA_FIJA, AssetSubClass.RF_ETF),
            (self.etfs_rf_globales, AssetClass.RENTA_FIJA, AssetSubClass.RF_ETF),
        ]
        for tickers, cls, sub in buckets:
            for tk in tickers:
                specs[tk] = AssetSpec(ticker=tk, name=tk, asset_class=cls, sub_class=sub)
        return specs


# ==============================================================================
# 1. PIPELINE DE DATOS DE MERCADO Y ENSAMBLE DE LA MATRIZ MULTI-ACTIVO
# ==============================================================================

class MarketDataPipeline:
    """
    Encapsula la descarga y limpieza de precios ajustados diarios y el ensamble
    de la matriz unificada de retornos/covarianzas Multi-Activo (RV + RF).

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
        max_fill_days: Optional[int] = MAX_DIAS_RELLENO_PRECIO,
    ) -> None:
        self.tickers = list(tickers)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.seed = int(seed)
        self.max_fill_days = max_fill_days
        self._rng = np.random.default_rng(seed)
        self.prices_: Optional[pd.DataFrame] = None
        self._common_factor: Optional[np.ndarray] = None
        self.synthetic_tickers_: List[str] = []
        self.raw_prices_: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------ #
    # DESCARGA DE PRECIOS
    # ------------------------------------------------------------------ #
    def fetch_prices(self) -> pd.DataFrame:
        """Descarga precios ajustados de cierre para todos los tickers."""
        # --- Inicialización del estado ---
        frames: Dict[str, pd.Series] = {}
        self.synthetic_tickers_ = []
        self.raw_prices_ = {}

        # --- Descarga ticker a ticker (yfinance) con fallback sintético ---
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
                except Exception as exc:
                    logger.warning("yfinance falló para %s (%s). Usando fallback sintético.", tk, exc)

            if series is None or series.empty:
                logger.warning("Sin precio de mercado para %s: se genera serie sintética (GBM).", tk)
                series = self._synthetic_price_series(tk)
                self.synthetic_tickers_.append(tk)
            else:
                self.raw_prices_[tk] = series

            frames[tk] = series

        # --- Alineación de calendarios y recorte a la ventana ---
        prices = pd.DataFrame(frames)
        prices = prices.sort_index().ffill(limit=self.max_fill_days).dropna(how="all")
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
        # --- Calendario y semilla estable por ticker ---
        n_days = max((self.end - self.start).days, 252)
        dates = pd.bdate_range(self.start, periods=n_days)

        local_seed = (zlib.crc32(ticker.encode("utf-8")) + self.seed) % (2**32)
        rng = np.random.default_rng(local_seed)

        # --- Parámetros de mercado según el tipo de activo ---
        tk = ticker.upper()
        is_global_bond_proxy = tk in {"TLT", "IEF", "AGG", "BND"}
        is_local_bond_proxy = tk in {"GXTESCOL.CL", "TESCOL.CL"}
        is_local_equity = tk.endswith(".CL")
        is_benchmark = tk in {"^GSPC", "SPY", "QQQ"}

        if is_global_bond_proxy:
            mu, sigma, s0, beta_mkt = 0.03, 0.12, 100.0, -0.15
        elif is_local_bond_proxy:
            mu, sigma, s0, beta_mkt = 0.085, 0.07, 12000.0, 0.10
        elif is_local_equity:
            mu, sigma, s0, beta_mkt = 0.09, 0.28, 25000.0, 0.55
        elif is_benchmark:
            mu, sigma, s0, beta_mkt = 0.11, 0.20, 400.0, 0.95
        else:
            mu, sigma, s0, beta_mkt = 0.10, 0.20, 400.0, 0.70

        # --- Factor de mercado común ---
        dt = 1 / TRADING_DAYS
        if self._common_factor is None or len(self._common_factor) != len(dates):
            self._common_factor = self._rng.normal(0, np.sqrt(dt), size=len(dates))
        common_factor = self._common_factor

        # --- Choques y trayectoria de precios (GBM) ---
        idio_vol = sigma * np.sqrt(max(1 - beta_mkt**2, 0.05))
        idio_shocks = rng.normal(0, idio_vol * np.sqrt(dt), size=len(dates))
        total_shocks = beta_mkt * sigma * common_factor + idio_shocks
        drift = (mu - 0.5 * sigma**2) * dt

        log_path = np.cumsum(drift + total_shocks)
        prices = s0 * np.exp(log_path)
        return pd.Series(prices, index=dates, name=ticker)

    # ------------------------------------------------------------------ #
    # CALIDAD DE DATOS
    # ------------------------------------------------------------------ #
    def liquidity_report(self) -> pd.DataFrame:
        """
        Indicadores de negociación de cada ticker con precio real, medidos en
        su propio calendario (antes de alinear con el resto del universo, para
        no contar como inactividad los festivos de otros mercados).
        """
        rows: Dict[str, Dict[str, object]] = {}
        for tk, s in self.raw_prices_.items():
            rows[tk] = {
                **self._liquidity_stats(s),
                "primer_dato": s.index[0],
                "ultimo_dato": s.index[-1],
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _liquidity_stats(s: pd.Series) -> Dict[str, float]:
        """% de días con retorno exactamente cero y racha máxima de precio congelado."""
        r = np.log(s / s.shift(1)).dropna()
        return {
            "pct_sin_movimiento": float((r.abs() < 1e-12).mean()) if len(r) else float("nan"),
            "racha_precio_congelado": int(s.ne(s.shift(1)).cumsum().value_counts().max()) if len(s) else 0,
        }

    # ------------------------------------------------------------------ #
    def tradability_issues_at(
        self,
        date: Union[str, pd.Timestamp],
        window_start: Union[str, pd.Timestamp],
        tickers: Sequence[str],
    ) -> Dict[str, str]:
        """
        Motivo por el que cada ticker NO es invertible en `date`, usando sólo
        los precios observados en [window_start, date] — la ventana de
        estimación del optimizador. Evaluar los filtros con toda la muestra
        haría que el backtest supiera en 2022 qué títulos se volverían
        ilíquidos o dejarían de cotizar después (sesgo de look-ahead).

        Los tickers sin precio real (serie sintética) no se evalúan aquí: su
        exclusión no depende de la fecha y se decide al armar el universo.
        """
        date, window_start = pd.Timestamp(date), pd.Timestamp(window_start)
        max_gap = self.max_fill_days if self.max_fill_days is not None else MAX_DIAS_RELLENO_PRECIO
        issues: Dict[str, str] = {}
        for tk in tickers:
            s = self.raw_prices_.get(tk)
            if s is None:
                continue
            s = s.loc[:date]

            # --- Cobertura: historia completa y cotización reciente ---
            if s.empty or s.index[0] > window_start:
                issues[tk] = "historia más corta que la ventana de estimación"
                continue
            if len(pd.bdate_range(s.index[-1], date)) - 1 > max_gap:
                issues[tk] = f"sin cotización desde {s.index[-1].date()}"
                continue

            # --- Liquidez dentro de la ventana ---
            liq = self._liquidity_stats(s.loc[window_start:])
            if liq["pct_sin_movimiento"] > MAX_PCT_DIAS_SIN_MOVIMIENTO:
                issues[tk] = (f"ilíquido: {liq['pct_sin_movimiento']:.0%} de días sin movimiento "
                              f"(máx. {MAX_PCT_DIAS_SIN_MOVIMIENTO:.0%})")
            elif liq["racha_precio_congelado"] > MAX_DIAS_PRECIO_CONGELADO:
                issues[tk] = (f"ilíquido: precio congelado {liq['racha_precio_congelado']} días "
                              f"seguidos (máx. {MAX_DIAS_PRECIO_CONGELADO})")
        return issues

    # ------------------------------------------------------------------ #
    # TRANSFORMACIONES Y ENSAMBLE DE LA MATRIZ
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

    # ------------------------------------------------------------------ #
    @staticmethod
    def prices_from_returns(
        returns: pd.DataFrame, base: float = 100.0
    ) -> pd.DataFrame:
        """
        Convierte retornos simples diarios en una serie de 'precios' indexada en
        `base`. Permite que los activos sintéticos de RF (nodos TES y bonos
        individuales) circulen por el mismo pipeline de precios que la RV.
        """
        levels = (1.0 + returns.fillna(0.0)).cumprod() * base
        first_row = pd.DataFrame(
            [[base] * returns.shape[1]],
            columns=returns.columns,
            index=[returns.index[0] - pd.Timedelta(days=1)],
        )
        return pd.concat([first_row, levels])

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_multi_asset_prices(
        *blocks: Optional[pd.DataFrame], late_start: Iterable[str] = ()
    ) -> pd.DataFrame:
        """
        Ensambla la matriz unificada de precios RV + RF alineando por fecha.

        Se conserva únicamente la intersección de fechas con dato en todos los
        bloques (tras `ffill`), que es la ventana sobre la que la covarianza
        conjunta es estimable sin imputaciones agresivas. Los tickers de
        `late_start` (listados después del inicio) quedan en NaN antes de su
        primer precio sin recortar la matriz; cada rebalanceo decide si su
        historia cubre ya la ventana de estimación.
        """
        valid = [b for b in blocks if b is not None and not b.empty]
        if not valid:
            raise ValueError("No hay bloques de precios para ensamblar.")
        merged = pd.concat(valid, axis=1, sort=True)
        merged = merged.loc[:, ~merged.columns.duplicated()].ffill()
        late = set(late_start)
        return merged.dropna(subset=[c for c in merged.columns if c not in late], how="any")

    # ------------------------------------------------------------------ #
    @staticmethod
    def covariance_matrix(
        returns: pd.DataFrame, shrinkage: bool = True, trading_days: int = TRADING_DAYS
    ) -> Tuple[np.ndarray, float]:
        """
        Matriz de covarianza anualizada. Con `shrinkage=True` aplica
        Ledoit-Wolf (2004) sobre los retornos diarios y anualiza después.
        """
        clean = returns.dropna(how="any")
        if shrinkage:
            lw = LedoitWolf().fit(clean.values)
            return lw.covariance_ * trading_days, float(lw.shrinkage_)
        return clean.cov().values * trading_days, 0.0


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
    # CONSTRUCTORES
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
    def from_levels(cls, tenors: Sequence[float], levels: Sequence[float]) -> "TESYieldCurve":
        """Reconstruye una curva a partir de un corte transversal de niveles."""
        return cls(tenors, levels)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls) -> "TESYieldCurve":
        """
        Curva construida desde el bloque de PARÁMETROS EDITABLES: usa el CSV de
        Banrep si `RUTA_CURVA_TES_CSV` está definido, y en su defecto los nodos
        de `CURVA_TES_PLAZOS` / `CURVA_TES_TASAS`.
        """
        if RUTA_CURVA_TES_CSV:
            logger.info("Curva TES cargada desde %s.", RUTA_CURVA_TES_CSV)
            return cls.from_banrep_csv(RUTA_CURVA_TES_CSV)
        return cls(CURVA_TES_PLAZOS, CURVA_TES_TASAS)

    # ------------------------------------------------------------------ #
    @classmethod
    def synthetic_example(cls) -> "TESYieldCurve":
        """Alias histórico de `from_config()`; se mantiene por compatibilidad."""
        return cls.from_config()

    # ------------------------------------------------------------------ #
    # CONSULTA DE TASAS Y FACTORES DE DESCUENTO
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
    def par_yield(self, maturity_years: float, freq: int = 1) -> float:
        """
        Tasa par (cupón que hace que el bono cotice a la par) para el plazo
        dado, derivada de los factores de descuento de la curva cero cupón:

            c_par = (1 - DF(T)) / Σ_i DF(t_i) / freq

        Los TES tasa fija en pesos pagan cupón anual (freq = 1), por lo que
        esta es la convención por defecto.
        """
        n = max(int(round(maturity_years * freq)), 1)
        times = np.arange(1, n + 1, dtype=float) / freq
        dfs = np.asarray(self.discount_factor(times), dtype=float)
        annuity = dfs.sum() / freq
        if annuity <= 0:
            return float(self.get_rate(maturity_years))
        return float((1.0 - dfs[-1]) / annuity)

    # ------------------------------------------------------------------ #
    # VISUALIZACIONES
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
# 2b. HISTÓRICO REAL DE LA CURVA TES (BANCO DE LA REPÚBLICA)
# ==============================================================================

class BanrepTESHistory:
    """
    Histórico diario de las tasas cero cupón de los TES en pesos a 1, 5 y 10
    años que publica el Banco de la República en su portal SUAMECA (extraídas
    de su curva Nelson-Siegel estimada con operaciones del SEN y el MEC).

    Guarda una copia local en `cache_path`: se reutiliza si cubre la ventana
    pedida, y sirve de respaldo si la descarga falla.
    """

    URL = ("https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/"
           "estadisticaEconomicaRestService/consultaInformacionSerie")
    SERIES: Dict[float, int] = {1.0: 15272, 5.0: 15273, 10.0: 15274}
    _HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://suameca.banrep.gov.co/estadisticas-economicas/",
        "Origin": "https://suameca.banrep.gov.co",
    }

    def __init__(self, cache_path: Optional[str] = None, timeout: float = 60.0) -> None:
        self.cache_path = cache_path
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # OBTENCIÓN DEL HISTÓRICO
    # ------------------------------------------------------------------ #
    def fetch(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]) -> pd.DataFrame:
        """
        Tasas E.A. en decimal (índice = fechas hábiles, columnas = plazos en
        años) para la ventana [start, end].
        """
        # --- Lectura de caché local ---
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        cached = self._read_cache()
        slack = pd.Timedelta(days=7)
        if cached is not None and cached.index[0] <= start + slack and cached.index[-1] >= end - slack:
            logger.info("Curva TES histórica leída de la caché %s.", self.cache_path)
            return cached.loc[start:end]

        # --- Descarga desde SUAMECA (con respaldo en caché) ---
        try:
            data = self._download()
        except Exception as exc:
            if cached is None:
                raise
            logger.warning("Descarga de la curva TES falló (%s); se usa la caché %s aunque no "
                           "cubra toda la ventana.", exc, self.cache_path)
            return cached.loc[start:end]

        # --- Actualización de caché y recorte a la ventana ---
        self._write_cache(data)
        logger.info("Curva TES histórica descargada de Banrep: %d días (%s a %s).",
                    len(data), data.index[0].date(), data.index[-1].date())
        return data.loc[start:end]

    # ------------------------------------------------------------------ #
    def _download(self) -> pd.DataFrame:
        import json

        # --- Descarga de una serie por plazo ---
        cols: Dict[float, pd.Series] = {}
        for tenor, series_id in self.SERIES.items():
            payload = json.loads(self._get(f"{self.URL}?idSerie={series_id}").decode("utf-8"))
            if not payload or not payload[0].get("data"):
                raise ValueError(f"Serie {series_id} vacía en SUAMECA.")
            ms, values = zip(*payload[0]["data"])
            idx = (pd.to_datetime(list(ms), unit="ms", utc=True)
                   .tz_convert("America/Bogota").normalize().tz_localize(None))
            cols[tenor] = pd.Series(np.asarray(values, dtype=float) / 100.0, index=idx)

        # --- Ensamble en un solo DataFrame (fechas x plazos) ---
        return pd.DataFrame(cols).sort_index().dropna(how="any")

    # ------------------------------------------------------------------ #
    # CONEXIÓN SEGURA (TLS)
    # ------------------------------------------------------------------ #
    def _get(self, url: str) -> bytes:
        """
        GET con verificación TLS completa. El servidor de SUAMECA no envía el
        certificado intermedio de su cadena (los navegadores lo completan solos
        vía AIA; Python no). Si la verificación falla por eso, se descarga el
        intermedio desde la URL que declara el propio certificado y se reintenta
        exigiendo que la cadena termine en una raíz de confianza.
        """
        import ssl
        import urllib.error
        import urllib.request

        # --- Primer intento: verificación TLS estándar ---
        req = urllib.request.Request(url, headers=self._HEADERS)
        context = self._base_ssl_context()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=context) as resp:
                return resp.read()
        except urllib.error.URLError as exc:
            if not isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise

        # --- Reintento: completando la cadena con el intermedio (AIA) ---
        host = urllib.request.urlparse(url).hostname
        context = self._context_with_intermediate(host)
        with urllib.request.urlopen(req, timeout=self.timeout, context=context) as resp:
            return resp.read()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _base_ssl_context() -> "ssl.SSLContext":
        import ssl

        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    # ------------------------------------------------------------------ #
    def _context_with_intermediate(self, host: str) -> "ssl.SSLContext":
        import re
        import ssl
        import urllib.request

        # --- Localizar la URL del certificado intermedio ---
        leaf_der = ssl.PEM_cert_to_DER_cert(ssl.get_server_certificate((host, 443), timeout=self.timeout))
        urls = re.findall(rb"http://[\x21-\x7e]+?\.(?:crt|cer|der)", leaf_der)
        if not urls:
            raise ssl.SSLError(f"{host}: cadena TLS incompleta y sin URL del emisor (AIA).")

        # --- Contexto TLS estricto con el intermedio descargado ---
        context = self._base_ssl_context()
        context.verify_flags &= ~getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0)
        for url in urls:
            with urllib.request.urlopen(url.decode("ascii"), timeout=self.timeout) as resp:
                cert = resp.read()
            pem = cert.decode("ascii") if cert.startswith(b"-----BEGIN") else ssl.DER_cert_to_PEM_cert(cert)
            context.load_verify_locations(cadata=pem)
        return context

    # ------------------------------------------------------------------ #
    # CACHÉ LOCAL
    # ------------------------------------------------------------------ #
    def _read_cache(self) -> Optional[pd.DataFrame]:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return None
        df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
        df.columns = [float(c) for c in df.columns]
        return df.sort_index() if not df.empty else None

    def _write_cache(self, data: pd.DataFrame) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        data.to_csv(self.cache_path, index_label="fecha")

    # ------------------------------------------------------------------ #
    # COMPLETAR LA CURVA (NELSON-SIEGEL)
    # ------------------------------------------------------------------ #
    @staticmethod
    def complete_curve(nodes: pd.DataFrame, tenors: Sequence[float], tau: float) -> pd.DataFrame:
        """
        Completa la curva a todos los `tenors` ajustando cada día un
        Nelson-Siegel con `tau` fijo: con tres nodos publicados (1, 5, 10 años)
        los tres β quedan determinados exactamente, y la curva resultante pasa
        por los nodos y extrapola de forma acotada fuera de ellos.
        """
        # --- Ajuste diario de los β de Nelson-Siegel sobre los nodos ---
        node_t = np.asarray(nodes.columns, dtype=float)
        betas = np.linalg.solve(nelson_siegel_loadings(node_t, tau), nodes.to_numpy().T).T

        # --- Evaluación de la curva en todos los plazos ---
        levels = betas @ nelson_siegel_loadings(tenors, tau).T
        return pd.DataFrame(levels, index=nodes.index, columns=[float(t) for t in tenors])


# ==============================================================================
# 3. ANALÍTICA DE RENTA FIJA (precio, YTM, duración, convexidad)
# ==============================================================================

class BondAnalytics:
    """
    Analítica estándar de bonos bullet bajo convención de tasa efectiva anual
    (la convención de mercado en Colombia para TES y deuda privada en pesos).
    """

    @staticmethod
    def cashflow_schedule(
        maturity_years: float, coupon_rate: float, freq: int = 1, face: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tiempos (años) y flujos de caja de un bono bullet con cupón periódico."""
        n = max(int(np.ceil(maturity_years * freq)), 1)
        times = maturity_years - np.arange(n - 1, -1, -1, dtype=float) / freq
        times = np.clip(times, 1e-6, None)
        cfs = np.full(n, face * coupon_rate / freq, dtype=float)
        cfs[-1] += face
        return times, cfs

    # ------------------------------------------------------------------ #
    @staticmethod
    def price_from_ytm(times: np.ndarray, cashflows: np.ndarray, ytm: float) -> float:
        """Precio sucio (limpio si no hay cupón corrido) descontando a YTM E.A."""
        return float(np.sum(cashflows * (1.0 + ytm) ** (-times)))

    # ------------------------------------------------------------------ #
    @staticmethod
    def ytm_from_price(times: np.ndarray, cashflows: np.ndarray, price: float) -> float:
        """YTM E.A. implícita en un precio de mercado (Brent sobre [-50%, 200%])."""
        def _f(y: float) -> float:
            return BondAnalytics.price_from_ytm(times, cashflows, y) - price

        try:
            return float(optimize.brentq(_f, -0.49, 2.0, maxiter=200, xtol=1e-10))
        except ValueError:
            logger.warning("YTM no acotada para precio %.4f; se retorna NaN.", price)
            return float("nan")

    # ------------------------------------------------------------------ #
    @staticmethod
    def duration_convexity(
        times: np.ndarray, cashflows: np.ndarray, ytm: float
    ) -> Tuple[float, float, float]:
        """
        Duración de Macaulay, Duración Modificada y Convexidad bajo tasa E.A.

            D_mod = D_mac / (1 + y)
            C     = Σ t·(t+1)·PV_t / (P · (1+y)²)
        """
        pv = cashflows * (1.0 + ytm) ** (-times)
        price = pv.sum()
        if price <= 0:
            return float("nan"), float("nan"), float("nan")
        macaulay = float((times * pv).sum() / price)
        modified = macaulay / (1.0 + ytm)
        convexity = float((times * (times + 1.0) * pv).sum() / (price * (1.0 + ytm) ** 2))
        return macaulay, modified, convexity

    # ------------------------------------------------------------------ #
    @staticmethod
    def price_return(
        delta_y: Union[float, np.ndarray],
        modified_duration: float,
        convexity: float,
    ) -> Union[float, np.ndarray]:
        """
        Aproximación de segundo orden del retorno de precio ante un choque de
        tasa:  ΔP/P ≈ −D_mod·Δy + ½·C·(Δy)²
        """
        dy = np.asarray(delta_y, dtype=float)
        return -modified_duration * dy + 0.5 * convexity * dy**2


# ==============================================================================
# 4. SIMULADOR DE CHOQUES DE CURVA (motor común de los Enfoques A y C)
# ==============================================================================

def nelson_siegel_loadings(tenors: Sequence[float], tau: float) -> np.ndarray:
    """Matriz (n_plazos x 3) de cargas Nelson-Siegel: nivel, pendiente, curvatura."""
    t = np.maximum(np.asarray(tenors, dtype=float), 1e-6) / tau
    l1 = np.ones_like(t)
    l2 = (1.0 - np.exp(-t)) / t
    l3 = l2 - np.exp(-t)
    return np.column_stack([l1, l2, l3])


@dataclass
class CurveShockConfig:
    """
    Parámetros del modelo de tres factores (nivel, pendiente, curvatura) tipo
    Nelson-Siegel que genera los choques diarios de la curva TES cuando no se
    dispone del histórico real de Banrep.

    Las volatilidades están expresadas en puntos básicos diarios.
    """

    level_vol_bp: float = field(default_factory=lambda: CURVA_VOL_NIVEL_BP)
    slope_vol_bp: float = field(default_factory=lambda: CURVA_VOL_PENDIENTE_BP)
    curvature_vol_bp: float = field(default_factory=lambda: CURVA_VOL_CURVATURA_BP)
    tau: float = 2.5
    mean_reversion: float = 0.015
    equity_beta_bp: float = field(default_factory=lambda: CURVA_BETA_RV_BP)
    floor_rate: float = 0.005
    seed: int = field(default_factory=lambda: SEMILLA_ALEATORIA)


class CurveShockGenerator:
    """
    Genera el histórico diario de niveles y variaciones de la curva TES por
    nodo de plazo. Admite dos fuentes:

      * `from_history(...)` — histórico real (p. ej. serie de Banrep).
      * `simulate(...)`     — modelo de 3 factores, para operación autónoma.

    El parámetro `equity_beta_bp` inyecta correlación entre los choques de tasa
    y el mercado accionario, evitando que la matriz de covarianza conjunta
    RV/RF sea artificialmente diagonal por bloques.
    """

    def __init__(
        self,
        curve: TESYieldCurve,
        tenors: Sequence[float],
        config: Optional[CurveShockConfig] = None,
    ) -> None:
        self.curve = curve
        self.tenors = np.asarray(sorted(tenors), dtype=float)
        self.config = config or CurveShockConfig()
        self.levels_: Optional[pd.DataFrame] = None
        self.changes_: Optional[pd.DataFrame] = None
        self.source_: Optional[str] = None

    # ------------------------------------------------------------------ #
    # GENERACIÓN DE LA CURVA DIARIA
    # ------------------------------------------------------------------ #
    def _nelson_siegel_loadings(self) -> np.ndarray:
        """Matriz (n_tenors x 3) de cargas factoriales nivel/pendiente/curvatura."""
        return nelson_siegel_loadings(self.tenors, self.config.tau)

    # ------------------------------------------------------------------ #
    def simulate(
        self, dates: pd.DatetimeIndex, market_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Simula niveles diarios de la curva. Retorna un DataFrame
        (index = fechas, columns = plazos en años, valores = tasa E.A.).
        """
        # --- Configuración y cargas factoriales ---
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n = len(dates)
        loadings = self._nelson_siegel_loadings()
        factor_vols = np.array([cfg.level_vol_bp, cfg.slope_vol_bp, cfg.curvature_vol_bp]) / 1e4

        # --- Factores con reversión a la media ---
        factors = np.zeros((n, 3))
        f = np.zeros(3)
        for i in range(n):
            f = f * (1.0 - cfg.mean_reversion) + rng.normal(0.0, factor_vols)
            factors[i] = f

        delta_factors = np.diff(np.vstack([np.zeros((1, 3)), factors]), axis=0)
        changes = delta_factors @ loadings.T

        # --- Componente correlacionado con la renta variable ---
        if market_returns is not None and cfg.equity_beta_bp != 0.0:
            mkt = market_returns.reindex(dates).fillna(0.0).values
            sigma_mkt = float(np.std(mkt)) or 1.0
            equity_shock = (cfg.equity_beta_bp / 1e4) * (mkt / sigma_mkt)
            changes = changes + equity_shock[:, None] * loadings[:, 0][None, :]

        # --- Niveles de la curva con piso y registro del resultado ---
        base = np.asarray(self.curve.get_rate(self.tenors), dtype=float)
        levels = np.maximum(base[None, :] + np.cumsum(changes, axis=0), cfg.floor_rate)
        levels_df = pd.DataFrame(levels, index=dates, columns=self.tenors)
        self.levels_ = levels_df
        self.changes_ = levels_df.diff().fillna(0.0)
        self.source_ = "simulada"
        return levels_df

    # ------------------------------------------------------------------ #
    def from_history(self, history: pd.DataFrame) -> pd.DataFrame:
        """
        Carga un histórico real de la curva. `history` debe tener fechas en el
        índice y plazos (años) en las columnas, con tasas en decimal. Los
        plazos solicitados se interpolan linealmente sobre los disponibles.
        """
        # --- Interpolación a los plazos solicitados ---
        hist = history.sort_index().astype(float)
        available = np.asarray([float(c) for c in hist.columns], dtype=float)
        interp = np.vstack([
            np.interp(self.tenors, available, row) for row in hist.values
        ])

        # --- Registro de niveles y variaciones ---
        levels_df = pd.DataFrame(interp, index=hist.index, columns=self.tenors)
        self.levels_ = levels_df
        self.changes_ = levels_df.diff().fillna(0.0)
        self.source_ = "histórica"
        return levels_df

    # ------------------------------------------------------------------ #
    # CONSULTA E INTERPOLACIÓN
    # ------------------------------------------------------------------ #
    def changes_at(self, maturities: Sequence[float]) -> pd.DataFrame:
        """Interpola las variaciones diarias Δy a plazos arbitrarios."""
        if self.changes_ is None:
            raise RuntimeError("Ejecute simulate() o from_history() primero.")
        return self._interp_columns(self.changes_, maturities)

    # ------------------------------------------------------------------ #
    def levels_at(self, maturities: Sequence[float]) -> pd.DataFrame:
        """Interpola los niveles diarios de la curva a plazos arbitrarios."""
        if self.levels_ is None:
            raise RuntimeError("Ejecute simulate() o from_history() primero.")
        return self._interp_columns(self.levels_, maturities)

    # ------------------------------------------------------------------ #
    def _interp_columns(self, frame: pd.DataFrame, maturities: Sequence[float]) -> pd.DataFrame:
        """
        Interpolación lineal por columnas: equivale a np.interp fila a fila,
        con extrapolación plana en los extremos, pero vectorizada.
        """
        mats = np.asarray(maturities, dtype=float)
        t = self.tenors
        m = np.clip(mats, t[0], t[-1])
        j = np.clip(np.searchsorted(t, m, side="right") - 1, 0, len(t) - 2)
        w = (m - t[j]) / (t[j + 1] - t[j])
        c = frame.to_numpy()
        vals = c[:, j] * (1.0 - w) + c[:, j + 1] * w
        return pd.DataFrame(vals, index=frame.index, columns=mats)

    # ------------------------------------------------------------------ #
    def curve_at(self, date: pd.Timestamp) -> TESYieldCurve:
        """Reconstruye la curva vigente en una fecha (para μ táctico)."""
        if self.levels_ is None:
            raise RuntimeError("Ejecute simulate() o from_history() primero.")
        idx = self.levels_.index
        pos = idx.searchsorted(pd.Timestamp(date), side="right") - 1
        pos = int(np.clip(pos, 0, len(idx) - 1))
        return TESYieldCurve.from_levels(self.tenors, self.levels_.iloc[pos].values)

    # ------------------------------------------------------------------ #
    # VISUALIZACIONES
    # ------------------------------------------------------------------ #
    def plot_curve_history(self) -> go.Figure:
        """Evolución del nivel de la curva por nodo de plazo."""
        if self.levels_ is None:
            raise RuntimeError("Ejecute simulate() o from_history() primero.")
        fig = go.Figure()
        for tenor in self.levels_.columns:
            fig.add_trace(go.Scatter(
                x=self.levels_.index, y=self.levels_[tenor] * 100,
                mode="lines", name=f"{tenor:g}A", line=dict(width=1.8),
            ))
        fig.update_layout(
            title=(f"Evolución {'histórica (Banrep)' if self.source_ == 'histórica' else 'simulada'} "
                   "de la curva TES por nodo de plazo"),
            xaxis_title="Fecha", yaxis_title="Tasa cero cupón (% E.A.)",
        )
        return fig


# ==============================================================================
# 5. ENFOQUE A — NODOS SINTÉTICOS DE DURACIÓN SOBRE LA CURVA TES
# ==============================================================================

class TESNodeBuilder:
    """
    Convierte la curva TES en 'activos sintéticos de renta fija' a plazos clave
    (TES_1Y, TES_3Y, TES_5Y, TES_10Y).

    Cada nodo se modela como un bono par-cupón vigente al plazo del nodo:
      * μ (retorno esperado)  = carry (YTM) + roll-down sobre la curva.
      * Retorno diario        = carry_diario + (−D_mod·Δy + ½·C·Δy²).
    """

    def __init__(
        self,
        curve: TESYieldCurve,
        tenors: Sequence[float] = (1.0, 3.0, 5.0, 10.0),
        coupon_freq: int = 1,
        include_rolldown: bool = True,
    ) -> None:
        self.curve = curve
        self.tenors = [float(t) for t in tenors]
        self.coupon_freq = coupon_freq
        self.include_rolldown = include_rolldown

    # ------------------------------------------------------------------ #
    # ANALÍTICA Y FICHAS DE LOS NODOS
    # ------------------------------------------------------------------ #
    @staticmethod
    def ticker_for(tenor: float) -> str:
        return f"TES_{tenor:g}Y"

    # ------------------------------------------------------------------ #
    def _node_analytics(self, curve: TESYieldCurve, tenor: float) -> Dict[str, float]:
        """Analítica de un nodo bajo una curva dada (par-cupón a la par)."""
        # --- Cupón par, YTM, duración y convexidad ---
        coupon = curve.par_yield(tenor, freq=self.coupon_freq)
        times, cfs = BondAnalytics.cashflow_schedule(tenor, coupon, self.coupon_freq)
        ytm = BondAnalytics.ytm_from_price(times, cfs, 100.0)
        if not np.isfinite(ytm):
            ytm = coupon
        _, dmod, conv = BondAnalytics.duration_convexity(times, cfs, ytm)

        # --- Roll-down ---
        rolldown = 0.0
        if self.include_rolldown and tenor > 1.0:
            rolldown = dmod * (float(curve.get_rate(tenor)) - float(curve.get_rate(tenor - 1.0)))
        return {"coupon": coupon, "ytm": ytm, "dmod": dmod, "convexity": conv, "rolldown": rolldown}

    # ------------------------------------------------------------------ #
    def build_specs(self, curve: Optional[TESYieldCurve] = None) -> List[AssetSpec]:
        """Fichas de los nodos sintéticos (μ, duración modificada, convexidad)."""
        crv = curve or self.curve
        specs: List[AssetSpec] = []
        for tenor in self.tenors:
            a = self._node_analytics(crv, tenor)
            specs.append(AssetSpec(
                ticker=self.ticker_for(tenor),
                name=f"Nodo sintético TES {tenor:g} años",
                asset_class=AssetClass.RENTA_FIJA,
                sub_class=AssetSubClass.RF_NODO_TES,
                expected_return=a["ytm"] + a["rolldown"],
                modified_duration=a["dmod"],
                convexity=a["convexity"],
                ytm=a["ytm"],
                metadata={"plazo_anios": tenor, "cupon_par": a["coupon"], "rolldown": a["rolldown"]},
            ))
        return specs

    # ------------------------------------------------------------------ #
    # RETORNOS DIARIOS DE LOS NODOS
    # ------------------------------------------------------------------ #
    def build_returns(
        self, shocks: CurveShockGenerator, specs: Optional[Sequence[AssetSpec]] = None
    ) -> pd.DataFrame:
        """
        Retornos simples diarios de cada nodo, como un bono par de plazo
        constante que se renueva a diario:

            r_t = carry_t + roll-down_t − D_{t−1}·Δc_t + ½·C_{t−1}·(Δc_t)²

        donde c_t es el cupón par de la curva del día al plazo del nodo. El
        carry, la duración y la convexidad se toman de la curva del día
        anterior: con tasas que se mueven de 2% a 13% (2021-2022), fijarlos en
        la curva inicial distorsionaría el retorno.
        """
        # --- Calendario y fichas de los nodos ---
        node_specs = list(specs or self.build_specs())
        idx = shocks.levels_.index
        d_tau = np.diff(np.asarray((idx - idx[0]).days, dtype=float) / 365.25, prepend=0.0)
        out: Dict[str, pd.Series] = {}

        for spec in node_specs:
            # --- Cupón par, duración y convexidad rezagados un día ---
            tenor = float(spec.metadata["plazo_anios"])
            par = self._par_yield_history(shocks, tenor)
            dmod, conv = self._par_bond_risk(par, tenor)
            prev = lambda x: np.concatenate([[x[0]], x[:-1]])
            par_prev, dmod_prev, conv_prev = prev(par), prev(dmod), prev(conv)

            # --- Componentes del retorno: carry, roll-down y precio ---
            dc = np.diff(par, prepend=par[0])
            carry = (1.0 + par_prev) ** d_tau - 1.0
            rolldown = np.zeros_like(par)
            if self.include_rolldown and tenor > 1.0:
                slope = par - self._par_yield_history(shocks, tenor - 1.0)
                rolldown = dmod_prev * prev(slope) * d_tau
            price_ret = -dmod_prev * dc + 0.5 * conv_prev * dc ** 2
            out[spec.ticker] = pd.Series(carry + rolldown + price_ret, index=idx)
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ #
    def _par_yield_history(self, shocks: CurveShockGenerator, tenor: float) -> np.ndarray:
        """Cupón par diario al plazo `tenor`: (1 − DF_T) / Σ DF_i / freq."""
        f = self.coupon_freq
        n = max(int(round(tenor * f)), 1)
        times = np.arange(1, n + 1, dtype=float) / f
        zeros = shocks.levels_at(times).to_numpy()
        dfs = (1.0 + zeros) ** (-times[None, :])
        return (1.0 - dfs[:, -1]) / (dfs.sum(axis=1) / f)

    # ------------------------------------------------------------------ #
    def _par_bond_risk(self, par: np.ndarray, tenor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Duración modificada y convexidad de un bono par (cupón = tasa) por fecha."""
        f = self.coupon_freq
        n = max(int(round(tenor * f)), 1)
        times = np.arange(1, n + 1, dtype=float) / f
        y = par[:, None]
        cfs = np.repeat(y / f, n, axis=1)
        cfs[:, -1] += 1.0
        pv = cfs * (1.0 + y) ** (-times[None, :])
        price = pv.sum(axis=1)
        dmod = (pv * times).sum(axis=1) / price / (1.0 + par)
        conv = (pv * times * (times + 1.0)).sum(axis=1) / (price * (1.0 + par) ** 2)
        return dmod, conv


# ==============================================================================
# 6. ENFOQUE C — SCREENING DE BONOS INDIVIDUALES (ISIN / emisor / rating)
# ==============================================================================

RATING_SCALE: Dict[str, int] = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
    "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16,
    "CCC": 17, "CC": 18, "C": 19, "D": 20,
}


@dataclass
class BondSpec:
    """
    Características de un bono individual del mercado local.

    `ytm`, `price` y `spread_bp` son alternativos (en ese orden de prioridad):
    si sólo se conoce el precio, la YTM se deriva por Brent; si sólo se conoce
    la tasa, el precio se calcula descontando los flujos; si sólo se conoce el
    spread, la YTM es la curva TES al plazo residual más el spread.
    """

    isin: str
    emisor: str
    rating: str
    coupon_rate: float
    maturity_date: Union[str, datetime]
    ytm: Optional[float] = None
    price: Optional[float] = None
    spread_bp: Optional[float] = None
    freq: int = 1
    face: float = 100.0
    indexacion: str = "TF"
    liquidez: float = 0.5
    sector: str = "Corporativo"


@dataclass
class BondScreeningCriteria:
    """Restricciones previas de elegibilidad (filtro de política de inversión)."""

    min_rating: str = field(default_factory=lambda: BONOS_MIN_RATING)
    min_maturity_years: float = field(default_factory=lambda: BONOS_PLAZO_MIN_ANIOS)
    max_maturity_years: float = field(default_factory=lambda: BONOS_PLAZO_MAX_ANIOS)
    min_liquidez: float = field(default_factory=lambda: BONOS_MIN_LIQUIDEZ)
    min_ytm: Optional[float] = None
    max_spread_bp: Optional[float] = field(default_factory=lambda: BONOS_MAX_SPREAD_BP)
    indexaciones_permitidas: Optional[Set[str]] = field(
        default_factory=lambda: set(BONOS_INDEXACIONES_PERMITIDAS)
        if BONOS_INDEXACIONES_PERMITIDAS else None
    )
    emisores_excluidos: Set[str] = field(default_factory=lambda: set(BONOS_EMISORES_EXCLUIDOS))
    max_por_emisor: int = field(default_factory=lambda: BONOS_MAX_POR_EMISOR)


class BondScreener:
    """
    Filtra un universo de bonos individuales según criterios de política y
    convierte los aprobados en activos candidatos con μ, duración modificada
    y convexidad calculados sobre la curva vigente.

    A diferencia de los nodos TES (plazo constante), un bono envejece: su
    plazo residual se acorta con el tiempo y vence en una fecha contractual.
    Por eso su retorno diario se obtiene revaluando los flujos que le quedan
    con la curva vigente de cada día más su spread, y tras el vencimiento el
    capital se reinvierte a la tasa de la curva al plazo `reinvest_tenor`.
    """

    def __init__(
        self,
        bonds: Sequence[BondSpec],
        criteria: Optional[BondScreeningCriteria] = None,
        curve: Optional[TESYieldCurve] = None,
        as_of: Optional[Union[str, datetime]] = None,
        reinvest_tenor: float = RF_TENOR_YEARS,
    ) -> None:
        self.bonds = list(bonds)
        self._bonds_by_isin: Dict[str, BondSpec] = {b.isin: b for b in self.bonds}
        self.criteria = criteria or BondScreeningCriteria()
        self.curve = curve or TESYieldCurve.synthetic_example()
        self.as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
        self.reinvest_tenor = float(reinvest_tenor)
        self.report_: Optional[pd.DataFrame] = None
        self.approved_: List[AssetSpec] = []
        self.spread_paths_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # UNIVERSO DE BONOS
    # ------------------------------------------------------------------ #
    @staticmethod
    def default_universe(as_of: Union[str, datetime] = "2021-01-01") -> List[BondSpec]:
        """
        Universo simulado de deuda pública y privada local, representativo de
        lo que entregaría un proveedor de precios (PiP / Precia) o una mesa de
        distribución. Sustituible por la lista real de ISINs sin cambiar el
        resto del pipeline.

        Los títulos en pesos se definen por su spread sobre la curva TES, de
        modo que su YTM sea coherente con la curva cargada (histórica o
        simulada). El TES UVR conserva su tasa real, que no es comparable con
        la curva en pesos.
        """
        base = pd.Timestamp(as_of)

        def mat(years: float) -> str:
            return (base + pd.Timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")

        return [
            BondSpec("COL17CT02622", "Ministerio de Hacienda (TES)", "AAA", 0.0700, mat(3.5),
                     spread_bp=3, freq=1, indexacion="TF", liquidez=0.95, sector="Soberano"),
            BondSpec("COL17CT03000", "Ministerio de Hacienda (TES)", "AAA", 0.0725, mat(7.2),
                     spread_bp=3, freq=1, indexacion="TF", liquidez=0.92, sector="Soberano"),
            BondSpec("COL17CT03109", "Ministerio de Hacienda (TES UVR)", "AAA", 0.0325, mat(9.0),
                     ytm=0.0365, freq=1, indexacion="UVR", liquidez=0.70, sector="Soberano"),
            BondSpec("COB07CB00123", "Bancolombia", "AAA", 0.0810, mat(4.0),
                     spread_bp=73, freq=2, indexacion="TF", liquidez=0.62, sector="Financiero"),
            BondSpec("COB07CB00456", "Banco de Bogotá", "AAA", 0.0790, mat(2.5),
                     spread_bp=68, freq=2, indexacion="TF", liquidez=0.58, sector="Financiero"),
            BondSpec("COE12CB00777", "Empresas Públicas de Medellín", "AA+", 0.0865, mat(6.0),
                     spread_bp=104, freq=1, indexacion="TF", liquidez=0.48, sector="Utilities"),
            BondSpec("COI15CB00321", "Interconexión Eléctrica (ISA)", "AAA", 0.0840, mat(8.5),
                     spread_bp=69, freq=1, indexacion="TF", liquidez=0.52, sector="Utilities"),
            BondSpec("COG21CB00654", "Grupo Argos", "AA", 0.0925, mat(5.0),
                     spread_bp=185, freq=1, indexacion="TF", liquidez=0.30, sector="Holding"),
            BondSpec("COD09CB00888", "Davivienda", "AA+", 0.0880, mat(12.0),
                     spread_bp=117, freq=2, indexacion="TF", liquidez=0.44, sector="Financiero"),
            BondSpec("COT31CB00999", "Titularizadora Colombiana", "AA-", 0.0950, mat(6.5),
                     spread_bp=244, freq=1, indexacion="TF", liquidez=0.22, sector="Titularizado"),
            BondSpec("COA44CB00111", "Avianca", "BBB", 0.1150, mat(4.5),
                     spread_bp=572, freq=2, indexacion="TF", liquidez=0.18, sector="Transporte"),
            BondSpec("COC55CB00222", "Celsia", "AA+", 0.0895, mat(0.6),
                     spread_bp=24, freq=1, indexacion="TF", liquidez=0.40, sector="Utilities"),
        ]

    # ------------------------------------------------------------------ #
    # ANALÍTICA Y CRITERIOS DE ELEGIBILIDAD
    # ------------------------------------------------------------------ #
    def _analytics(self, bond: BondSpec) -> Dict[str, float]:
        """Plazo residual, YTM, precio, duración modificada, convexidad y spread."""
        # --- Plazo residual y flujos de caja ---
        ttm = float((pd.Timestamp(bond.maturity_date) - self.as_of).days) / 365.25
        ttm = max(ttm, 1e-3)
        times, cfs = BondAnalytics.cashflow_schedule(ttm, bond.coupon_rate, bond.freq, bond.face)

        # --- YTM y precio (según el dato disponible) ---
        if bond.ytm is not None:
            ytm = float(bond.ytm)
            price = BondAnalytics.price_from_ytm(times, cfs, ytm)
        elif bond.price is not None:
            price = float(bond.price)
            ytm = BondAnalytics.ytm_from_price(times, cfs, price)
        elif bond.spread_bp is not None:
            ytm = float(self.curve.get_rate(ttm)) + float(bond.spread_bp) / 1e4
            price = BondAnalytics.price_from_ytm(times, cfs, ytm)
        else:
            raise ValueError(f"El bono {bond.isin} debe traer 'ytm', 'price' o 'spread_bp'.")

        # --- Duración, convexidad y spread sobre la curva ---
        _, dmod, conv = BondAnalytics.duration_convexity(times, cfs, ytm)
        spread_bp = (ytm - float(self.curve.get_rate(ttm))) * 1e4
        return {
            "plazo_anios": ttm, "ytm": ytm, "precio": price,
            "duracion_mod": dmod, "convexidad": conv, "spread_bp": spread_bp,
        }

    # ------------------------------------------------------------------ #
    def _rejection_reason(self, bond: BondSpec, a: Dict[str, float]) -> Optional[str]:
        return self._static_rejection(bond) or self._dynamic_rejection(a)

    # ------------------------------------------------------------------ #
    def _static_rejection(self, bond: BondSpec) -> Optional[str]:
        """Criterios propios del título, que no cambian con el paso del tiempo."""
        c = self.criteria
        rating_rank = RATING_SCALE.get(bond.rating.upper())
        if rating_rank is None:
            return f"rating desconocido ({bond.rating})"
        if rating_rank > RATING_SCALE[c.min_rating.upper()]:
            return f"rating {bond.rating} < mínimo {c.min_rating}"
        if bond.liquidez < c.min_liquidez:
            return f"liquidez {bond.liquidez:.2f} < {c.min_liquidez:.2f}"
        if c.indexaciones_permitidas and bond.indexacion not in c.indexaciones_permitidas:
            return f"indexación {bond.indexacion} no permitida"
        if bond.emisor in c.emisores_excluidos:
            return "emisor excluido por política"
        return None

    # ------------------------------------------------------------------ #
    def _dynamic_rejection(self, a: Mapping[str, float]) -> Optional[str]:
        """Criterios que dependen de la fecha: plazo residual, YTM y spread."""
        c = self.criteria
        if not (c.min_maturity_years <= a["plazo_anios"] <= c.max_maturity_years):
            return (f"plazo {a['plazo_anios']:.2f}A fuera de "
                    f"[{c.min_maturity_years:g}, {c.max_maturity_years:g}]")
        if c.min_ytm is not None and a["ytm"] < c.min_ytm:
            return f"YTM {a['ytm']*100:.2f}% < mínimo {c.min_ytm*100:.2f}%"
        if c.max_spread_bp is not None and a["spread_bp"] > c.max_spread_bp:
            return f"spread {a['spread_bp']:.0f}pb > máximo {c.max_spread_bp:.0f}pb"
        return None

    # ------------------------------------------------------------------ #
    def _cap_per_issuer(self, candidates: pd.DataFrame) -> pd.Index:
        """
        Índices que exceden el máximo de títulos por emisor. Se conservan los de
        mayor YTM ajustada por liquidez (columnas 'emisor', 'ytm', 'liquidez').
        """
        drop: List[object] = []
        if self.criteria.max_por_emisor > 0 and not candidates.empty:
            score = candidates["ytm"] * candidates["liquidez"]
            for _, grp in candidates.groupby("emisor"):
                if len(grp) > self.criteria.max_por_emisor:
                    ranked = score.loc[grp.index].sort_values(ascending=False)
                    drop.extend(ranked.index[self.criteria.max_por_emisor:])
        return pd.Index(drop)

    # ------------------------------------------------------------------ #
    # SCREENING
    # ------------------------------------------------------------------ #
    def screen(self) -> pd.DataFrame:
        """
        Aplica los filtros y construye el reporte de screening. Retorna un
        DataFrame con la analítica de cada título y su veredicto.
        """
        # --- Analítica y veredicto por título ---
        rows: List[Dict[str, object]] = []
        for bond in self.bonds:
            a = self._analytics(bond)
            reason = self._rejection_reason(bond, a)
            rows.append({
                "isin": bond.isin, "emisor": bond.emisor, "sector": bond.sector,
                "rating": bond.rating, "indexacion": bond.indexacion,
                "cupon": bond.coupon_rate, "liquidez": bond.liquidez, **a,
                "aprobado": reason is None, "motivo_rechazo": reason or "",
            })

        report = pd.DataFrame(rows).set_index("isin")

        # --- Límite de concentración por emisor ---
        drop = self._cap_per_issuer(report[report["aprobado"]])
        report.loc[drop, "aprobado"] = False
        report.loc[drop, "motivo_rechazo"] = (
            f"excede máximo de {self.criteria.max_por_emisor} títulos por emisor"
        )

        # --- Registro del reporte ---
        self.report_ = report
        logger.info(
            "Screening de bonos: %d/%d títulos aprobados.",
            int(report["aprobado"].sum()), len(report),
        )
        return report

    # ------------------------------------------------------------------ #
    # FICHAS DE LOS BONOS CANDIDATOS
    # ------------------------------------------------------------------ #
    def approved_specs(self, include_rolldown: bool = True) -> List[AssetSpec]:
        """Convierte los bonos aprobados en `as_of` en activos candidatos del optimizador."""
        report = self.report_ if self.report_ is not None else self.screen()
        specs = self._specs_from_rows(report[report["aprobado"]], include_rolldown)
        self.approved_ = specs
        return specs

    # ------------------------------------------------------------------ #
    def candidate_specs(
        self, until: Union[str, pd.Timestamp], include_rolldown: bool = True
    ) -> List[AssetSpec]:
        """
        Bonos que podrían ser elegibles en algún momento entre `as_of` y
        `until`: cumplen los criterios estáticos (rating, liquidez, indexación,
        emisor) y su plazo residual cruza la banda de plazo permitida en esa
        ventana. La elegibilidad efectiva se decide fecha a fecha con
        `eligible_at`.
        """
        report = self.report_ if self.report_ is not None else self.screen()
        tau_end = float(self._years_since_as_of(pd.Timestamp(until))[0])
        c = self.criteria
        keep = [
            isin for isin, row in report.iterrows()
            if self._static_rejection(self._bonds_by_isin[isin]) is None
            and row["plazo_anios"] >= c.min_maturity_years
            and row["plazo_anios"] - tau_end <= c.max_maturity_years
        ]
        return self._specs_from_rows(report.loc[keep], include_rolldown)

    # ------------------------------------------------------------------ #
    def eligible_at(
        self, specs: Sequence[AssetSpec], curve: TESYieldCurve, date: Union[str, pd.Timestamp]
    ) -> List[str]:
        """
        Screening en la fecha `date`: aplica los criterios dependientes del
        tiempo (plazo residual, YTM, spread vigente) y el máximo por emisor.
        Sólo usa información disponible en esa fecha.
        """
        # --- Criterios dependientes de la fecha ---
        rows: Dict[str, Dict[str, object]] = {}
        for spec in specs:
            a = self.analytics_at(spec, curve, date, include_rolldown=False)
            if a["vencido"] or self._dynamic_rejection(a) is not None:
                continue
            rows[spec.ticker] = {"emisor": spec.metadata["emisor"], "ytm": a["ytm"],
                                 "liquidez": float(spec.metadata["liquidez"])}

        # --- Máximo de títulos por emisor ---
        passed = pd.DataFrame.from_dict(rows, orient="index")
        return [tk for tk in passed.index if tk not in self._cap_per_issuer(passed)]

    # ------------------------------------------------------------------ #
    def _specs_from_rows(self, rows: pd.DataFrame, include_rolldown: bool) -> List[AssetSpec]:
        specs: List[AssetSpec] = []
        for isin, row in rows.iterrows():
            ttm = float(row["plazo_anios"])
            dmod = float(row["duracion_mod"])
            rolldown = 0.0
            if include_rolldown and ttm > 1.0:
                rolldown = dmod * (
                    float(self.curve.get_rate(ttm)) - float(self.curve.get_rate(ttm - 1.0))
                )
            specs.append(AssetSpec(
                ticker=f"BOND_{isin}",
                name=f"{row['emisor']} {row['rating']} {ttm:.1f}A",
                asset_class=AssetClass.RENTA_FIJA,
                sub_class=AssetSubClass.RF_BONO,
                expected_return=float(row["ytm"]) + rolldown,
                modified_duration=dmod,
                convexity=float(row["convexidad"]),
                ytm=float(row["ytm"]),
                metadata={
                    "isin": isin, "emisor": row["emisor"], "rating": row["rating"],
                    "sector": row["sector"], "plazo_anios": ttm,
                    "vencimiento": pd.Timestamp(self._bonds_by_isin[isin].maturity_date).strftime("%Y-%m-%d"),
                    "spread_bp": float(row["spread_bp"]), "liquidez": float(row["liquidez"]),
                    "indexacion": row["indexacion"], "rolldown": rolldown,
                },
            ))
        return specs

    # ------------------------------------------------------------------ #
    # RETORNOS DIARIOS (REVALUACIÓN COMPLETA)
    # ------------------------------------------------------------------ #
    def build_returns(
        self,
        shocks: CurveShockGenerator,
        specs: Optional[Sequence[AssetSpec]] = None,
        spread_vol_bp: float = 2.0,
        seed: int = 77,
    ) -> pd.DataFrame:
        """
        Retornos simples diarios de los bonos aprobados por revaluación
        completa: cada día se descuentan los flujos que le quedan al título a
        su YTM vigente,

            YTM_t = z_t(plazo_t) + spread_t,     plazo_t = plazo_0 − τ_t

        donde z_t es la curva simulada de ese día y el spread sigue una
        caminata aleatoria escalada por la (i)liquidez del título. El retorno
        total incluye el cupón cobrado en el día:

            r_t = (P_t + cupón_t) / P_{t−1} − 1

        Así el plazo, la duración y la convexidad se acortan con el tiempo y el
        precio converge a la par. Tras el vencimiento el capital se reinvierte
        a la tasa de la curva al plazo `reinvest_tenor`.
        """
        # --- Validaciones y universo de bonos ---
        bond_specs = list(specs or self.approved_specs())
        if not bond_specs:
            return pd.DataFrame(index=shocks.changes_.index if shocks.changes_ is not None else None)
        if shocks.levels_ is None:
            raise RuntimeError("Ejecute simulate() o from_history() en el generador de choques primero.")

        # --- Calendario, curva y retorno de reinversión ---
        levels = shocks.levels_
        tenors = np.asarray(levels.columns, dtype=float)
        level_rows = levels.to_numpy()
        tau = self._years_since_as_of(levels.index)
        d_tau = np.diff(tau, prepend=tau[0])
        reinvest = np.array([np.interp(self.reinvest_tenor, tenors, row) for row in level_rows])
        cash_ret = (1.0 + reinvest) ** d_tau - 1.0

        rng = np.random.default_rng(seed)
        out: Dict[str, pd.Series] = {}
        spreads: Dict[str, pd.Series] = {}

        for spec in bond_specs:
            # --- Flujos y plazo inicial del bono ---
            pay_t, cfs = self._cashflow_calendar(spec)
            ttm0 = float(spec.metadata["plazo_anios"])

            # --- Spread simulado (escalado por iliquidez) ---
            liq = float(spec.metadata.get("liquidez", 0.5))
            spread_shock = rng.normal(0.0, (spread_vol_bp / 1e4) * (1.5 - liq), size=len(levels))
            spread = float(spec.metadata["spread_bp"]) / 1e4 + np.cumsum(spread_shock)

            # --- YTM diaria: curva al plazo residual + spread ---
            ttm = np.maximum(ttm0 - tau, 0.0)
            zero = np.array([np.interp(m, tenors, row) for m, row in zip(ttm, level_rows)])
            ytm = zero + spread

            # --- Revaluación completa: precio y cupones cobrados ---
            t_to_pay = pay_t[None, :] - tau[:, None]
            pending = t_to_pay > 0.0
            disc = (1.0 + ytm[:, None]) ** (-np.where(pending, t_to_pay, 0.0))
            price = np.where(pending, cfs[None, :] * disc, 0.0).sum(axis=1)
            paid = np.zeros(len(tau))
            paid[1:] = (cfs[None, :] * (pending[:-1] & ~pending[1:])).sum(axis=1)

            # --- Retorno total (con reinversión tras el vencimiento) ---
            ret = np.zeros(len(tau))
            alive_prev = pending[:-1].any(axis=1)
            prev_price = price[:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                bond_ret = (price[1:] + paid[1:]) / prev_price - 1.0
            ret[1:] = np.where(alive_prev, bond_ret, cash_ret[1:])

            # --- Registro de la serie y aviso de vencimiento ---
            out[spec.ticker] = pd.Series(ret, index=levels.index)
            spreads[spec.ticker] = pd.Series(spread, index=levels.index)

            if tau[-1] >= ttm0:
                logger.info(
                    "%s vence el %s dentro de la ventana: desde entonces su serie rinde la tasa "
                    "de reinversión (curva a %.2g años); deja de ser invertible antes, al "
                    "bajar del plazo mínimo del screening.",
                    spec.ticker, spec.metadata.get("vencimiento", "?"), self.reinvest_tenor,
                )

        # --- Salida ---
        self.spread_paths_ = pd.DataFrame(spreads)
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ #
    def _years_since_as_of(self, dates: Union[pd.DatetimeIndex, pd.Timestamp]) -> np.ndarray:
        """Años (ACT/365.25) transcurridos desde `as_of`, misma base que el plazo residual."""
        idx = pd.DatetimeIndex([dates]) if isinstance(dates, pd.Timestamp) else pd.DatetimeIndex(dates)
        return np.asarray((idx - self.as_of).days, dtype=float) / 365.25

    # ------------------------------------------------------------------ #
    def _cashflow_calendar(self, spec: AssetSpec) -> Tuple[np.ndarray, np.ndarray]:
        """Fechas de pago (años desde `as_of`) y montos de los flujos del bono."""
        bond = self._bonds_by_isin[str(spec.metadata["isin"])]
        return BondAnalytics.cashflow_schedule(
            float(spec.metadata["plazo_anios"]), bond.coupon_rate, bond.freq, bond.face
        )

    # ------------------------------------------------------------------ #
    # ANALÍTICA A UNA FECHA
    # ------------------------------------------------------------------ #
    def analytics_at(
        self,
        spec: AssetSpec,
        curve: TESYieldCurve,
        date: Union[str, pd.Timestamp],
        include_rolldown: bool = True,
    ) -> Dict[str, float]:
        """
        Plazo residual, YTM, duración modificada, convexidad, spread, roll-down y μ del bono en
        `date` con la curva dada. Usa el spread simulado vigente en esa fecha
        (conocido en t, sin mirar el futuro). Si el bono ya venció, μ es la
        tasa de reinversión.
        """
        # --- Flujos pendientes a la fecha ---
        date = pd.Timestamp(date)
        tau = float(self._years_since_as_of(date)[0])
        pay_t, cfs = self._cashflow_calendar(spec)
        pending = pay_t > tau

        # --- Bono vencido: rinde la tasa de reinversión ---
        if not pending.any():
            rate = float(curve.get_rate(self.reinvest_tenor))
            return {"plazo_anios": 0.0, "ytm": rate, "duracion_mod": 0.0, "convexidad": 0.0,
                    "spread_bp": 0.0, "rolldown": 0.0, "mu": rate, "vencido": 1.0}

        # --- Spread vigente en la fecha ---
        spread = float(spec.metadata["spread_bp"]) / 1e4
        if self.spread_paths_ is not None and spec.ticker in self.spread_paths_:
            path = self.spread_paths_[spec.ticker]
            pos = int(np.clip(path.index.searchsorted(date, side="right") - 1, 0, len(path) - 1))
            spread = float(path.iloc[pos])

        # --- YTM, duración y convexidad ---
        ttm = float(pay_t[-1] - tau)
        ytm = float(curve.get_rate(ttm)) + spread
        _, dmod, conv = BondAnalytics.duration_convexity(pay_t[pending] - tau, cfs[pending], ytm)

        # --- Roll-down y μ ---
        rolldown = 0.0
        if include_rolldown and ttm > 1.0:
            rolldown = dmod * (float(curve.get_rate(ttm)) - float(curve.get_rate(ttm - 1.0)))
        return {"plazo_anios": ttm, "ytm": ytm, "duracion_mod": dmod, "convexidad": conv,
                "spread_bp": spread * 1e4, "rolldown": rolldown, "mu": ytm + rolldown, "vencido": 0.0}

    # ------------------------------------------------------------------ #
    # VISUALIZACIONES
    # ------------------------------------------------------------------ #
    def plot_screening(self) -> go.Figure:
        """Mapa plazo–YTM del universo de bonos, coloreado por veredicto."""
        # --- Puntos del universo por veredicto ---
        report = self.report_ if self.report_ is not None else self.screen()
        fig = go.Figure()
        for aprobado, color, label in [(True, "seagreen", "Aprobado"), (False, "indianred", "Rechazado")]:
            sub = report[report["aprobado"] == aprobado]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["plazo_anios"], y=sub["ytm"] * 100, mode="markers", name=label,
                marker=dict(size=12, color=color, line=dict(width=1, color="white")),
                text=[f"{i}<br>{e} ({r})<br>{m}" for i, e, r, m in
                      zip(sub.index, sub["emisor"], sub["rating"], sub["motivo_rechazo"])],
                hovertemplate="%{text}<br>Plazo: %{x:.2f}A<br>YTM: %{y:.2f}%<extra></extra>",
            ))

        # --- Curva TES de referencia ---
        t_fine = np.linspace(0.25, max(float(report["plazo_anios"].max()), 10.0), 200)
        fig.add_trace(go.Scatter(
            x=t_fine, y=np.asarray(self.curve.get_rate(t_fine)) * 100, mode="lines",
            name="Curva TES (referencia)", line=dict(width=2, dash="dot", color="steelblue"),
        ))

        # --- Formato ---
        fig.update_layout(
            title=(f"Screening de bonos individuales — criterio: rating ≥ {self.criteria.min_rating}, "
                   f"plazo ∈ [{self.criteria.min_maturity_years:g}, {self.criteria.max_maturity_years:g}] años"),
            xaxis_title="Plazo residual (años)", yaxis_title="YTM (% E.A.)",
        )
        return fig


# ==============================================================================
# 7. MOTOR DE RENTA FIJA — ORQUESTACIÓN DE LOS ENFOQUES A + B + C
# ==============================================================================

@dataclass
class FixedIncomeBundle:
    """Resultado del ensamble de la pata de renta fija."""

    specs: Dict[str, AssetSpec]
    prices: pd.DataFrame
    returns: pd.DataFrame
    analytics: pd.DataFrame
    screening_report: pd.DataFrame
    curve_levels: pd.DataFrame

    @property
    def tickers(self) -> List[str]:
        return list(self.prices.columns)


class FixedIncomeEngine:
    """
    Ensambla la pata de Renta Fija combinando las tres vías:

      A. Nodos sintéticos de la curva TES  (`TESNodeBuilder`)
      B. ETFs / FICs con precio de mercado (se etiquetan; su serie viene del
         `MarketDataPipeline`)
      C. Bonos individuales que aprueban el screening (`BondScreener`)

    Expone además `expected_returns_at(date)`, que recalcula el μ analítico de
    la RF con la curva vigente en cada fecha de rebalanceo — el componente
    táctico de la asignación.
    """

    def __init__(
        self,
        curve: TESYieldCurve,
        node_tenors: Sequence[float] = (1.0, 3.0, 5.0, 10.0),
        bonds: Optional[Sequence[BondSpec]] = None,
        criteria: Optional[BondScreeningCriteria] = None,
        shock_config: Optional[CurveShockConfig] = None,
        as_of: Optional[Union[str, datetime]] = None,
        include_rolldown: bool = True,
    ) -> None:
        # --- Enfoque A: constructor de nodos TES ---
        self.curve = curve
        self.node_builder = TESNodeBuilder(curve, node_tenors, include_rolldown=include_rolldown)
        self.include_rolldown = include_rolldown

        # --- Generador de choques de curva ---
        tenors = sorted(set([float(t) for t in node_tenors]) | set(curve.tenors.tolist()))
        self.shocks = CurveShockGenerator(curve, tenors, shock_config)

        # --- Enfoque C: screener de bonos individuales ---
        self.screener = BondScreener(
            bonds if bonds is not None else BondScreener.default_universe(as_of or "2021-01-01"),
            criteria, curve, as_of,
        )
        self.bundle_: Optional[FixedIncomeBundle] = None

    # ------------------------------------------------------------------ #
    # ENSAMBLE DE LA RENTA FIJA
    # ------------------------------------------------------------------ #
    def build(
        self,
        dates: pd.DatetimeIndex,
        market_returns: Optional[pd.Series] = None,
        curve_history: Optional[pd.DataFrame] = None,
    ) -> FixedIncomeBundle:
        """
        Construye el bloque de RF modelada (Enfoques A y C) sobre el calendario
        `dates`. Si se entrega `curve_history` (histórico real de Banrep) se usa
        en lugar del simulador de choques.
        """
        # --- Curva diaria: histórica o simulada ---
        if curve_history is not None:
            hist = curve_history.sort_index()
            hist = hist.reindex(hist.index.union(pd.DatetimeIndex(dates))).ffill().bfill()
            self.shocks.from_history(hist.reindex(pd.DatetimeIndex(dates)))
        else:
            self.shocks.simulate(pd.DatetimeIndex(dates), market_returns)

        # --- Enfoque A: nodos TES ---
        node_specs = self.node_builder.build_specs()
        node_returns = self.node_builder.build_returns(self.shocks, node_specs)

        # --- Enfoque C: bonos individuales ---
        self.screener.screen()
        approved_now = self.screener.approved_specs(include_rolldown=self.include_rolldown)
        bond_specs = self.screener.candidate_specs(
            until=pd.DatetimeIndex(dates)[-1], include_rolldown=self.include_rolldown
        )
        bond_returns = self.screener.build_returns(self.shocks, bond_specs)

        # --- Ensamble de retornos, precios y analítica ---
        returns = pd.concat([node_returns, bond_returns], axis=1) if not bond_returns.empty else node_returns
        prices = MarketDataPipeline.prices_from_returns(returns, base=100.0)

        specs = {s.ticker: s for s in list(node_specs) + list(bond_specs)}
        analytics = pd.DataFrame([s.as_row() for s in specs.values()]).set_index("ticker")

        # --- Paquete de salida ---
        self.bundle_ =FixedIncomeBundle(
            specs=specs,
            prices=prices,
            returns=returns,
            analytics=analytics,
            screening_report=self.screener.report_,
            curve_levels=self.shocks.levels_,
        )
        logger.info(
            "Renta Fija ensamblada: %d nodos TES (A) + %d bonos candidatos (C; %d aprobados al "
            "inicio, el resto entra o sale según su plazo) sobre %d días.",
            len(node_specs), len(bond_specs), len(approved_now), len(returns),
        )
        return self.bundle_

    # ------------------------------------------------------------------ #
    # SCREENING DINÁMICO Y RETORNOS PRO-FORMA
    # ------------------------------------------------------------------ #
    def _bond_specs(self) -> List[AssetSpec]:
        if self.bundle_ is None:
            raise RuntimeError("Ejecute build() primero.")
        return [s for s in self.bundle_.specs.values() if s.sub_class is AssetSubClass.RF_BONO]

    # ------------------------------------------------------------------ #
    def excluded_bonds_at(self, date: Union[str, pd.Timestamp]) -> Set[str]:
        """
        Bonos candidatos que NO son invertibles en `date`: vencidos, fuera de
        la banda de plazo, con spread o YTM fuera de política, o que exceden el
        máximo por emisor. El screening se repite con la curva y el spread
        vigentes en esa fecha.
        """
        specs = self._bond_specs()
        crv = self.shocks.curve_at(pd.Timestamp(date))
        eligible = set(self.screener.eligible_at(specs, crv, date))
        return {s.ticker for s in specs} - eligible

    # ------------------------------------------------------------------ #
    def proforma_log_returns(
        self, date: Union[str, pd.Timestamp], window: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Reemplaza, dentro de una ventana de retornos log, la serie de cada bono
        por su versión *pro-forma*: los choques históricos de curva y spread
        aplicados a las características que el bono tiene HOY (plazo, YTM,
        duración y convexidad en `date`),

            r_s = carry(YTM_hoy) − D_hoy·Δy_s + ½·C_hoy·Δy_s²,
            Δy_s = Δz_s(plazo_hoy) + Δspread_s

        Sin este ajuste la covarianza reflejaría la duración que el bono tenía
        durante la ventana (mayor que la actual) y sobreestimaría su riesgo.
        """
        # --- Curva vigente y calendario ---
        date = pd.Timestamp(date)
        out = window.copy()
        crv = self.shocks.curve_at(date)
        levels_idx = self.shocks.levels_.index
        d_tau = pd.Series(
            np.diff(self.screener._years_since_as_of(levels_idx), prepend=np.nan), index=levels_idx
        )
        spreads = self.screener.spread_paths_

        # --- Reemplazo de la serie de cada bono por su versión pro-forma ---
        for tk in window.columns:
            spec = self.bundle_.specs.get(tk) if self.bundle_ is not None else None
            if spec is None or spec.sub_class is not AssetSubClass.RF_BONO:
                continue
            a = self.screener.analytics_at(spec, crv, date, include_rolldown=False)
            if a["vencido"]:
                continue
            dz = self.shocks.changes_at([a["plazo_anios"]]).iloc[:, 0]
            ds = spreads[tk].diff() if spreads is not None and tk in spreads else 0.0 * dz
            dy = (dz + ds).reindex(window.index).fillna(0.0).to_numpy()
            dt = d_tau.reindex(window.index).fillna(1.0 / 365.25).to_numpy()
            carry = (1.0 + a["ytm"]) ** dt - 1.0
            simple = carry - a["duracion_mod"] * dy + 0.5 * a["convexidad"] * dy ** 2
            out[tk] = np.log1p(simple)
        return out

    # ------------------------------------------------------------------ #
    # SEÑALES TÁCTICAS (μ Y TASA LIBRE DE RIESGO)
    # ------------------------------------------------------------------ #
    def expected_returns_at(self, date: Union[str, pd.Timestamp]) -> pd.Series:
        """
        μ analítico (YTM + roll-down) de los activos de RF modelada usando la
        curva vigente en `date`. Es la señal táctica que alimenta al
        optimizador en cada rebalanceo del walk-forward.
        """
        if self.bundle_ is None:
            raise RuntimeError("Ejecute build() antes de solicitar μ táctico.")
        crv = self.shocks.curve_at(pd.Timestamp(date))
        mu: Dict[str, float] = {}

        # --- Nodos TES ---
        for spec in self.node_builder.build_specs(crv):
            mu[spec.ticker] = float(spec.expected_return)

        # --- Bonos individuales ---
        for ticker, spec in self.bundle_.specs.items():
            if spec.sub_class is not AssetSubClass.RF_BONO:
                continue
            mu[ticker] = self.screener.analytics_at(spec, crv, date, self.include_rolldown)["mu"]

        return pd.Series(mu, name="mu_rf")

    # ------------------------------------------------------------------ #
    def risk_free_at(self, date: Union[str, pd.Timestamp], tenor: float = 1.0) -> float:
        """Tasa libre de riesgo vigente en `date`, leída de la curva simulada."""
        return float(self.shocks.curve_at(pd.Timestamp(date)).get_rate(tenor))


# ==============================================================================
# 8. MÉTRICAS DE RIESGO MICRO
# ==============================================================================

class RiskMetrics:
    """Colección de métricas de riesgo estándar a nivel de activo individual."""

    TRADING_DAYS: int = TRADING_DAYS

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
        cls,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        market_col: str,
        rf: float,
        specs: Optional[Mapping[str, AssetSpec]] = None,
    ) -> pd.DataFrame:
        """Genera un reporte tabular de métricas de riesgo por activo."""
        # --- Métricas por activo ---
        rows = []
        for col in returns.columns:
            spec = specs.get(col) if specs else None
            r = returns[col].dropna()
            rows.append(
                {
                    "ticker": col,
                    "clase": spec.asset_class.value if spec else "",
                    "sub_clase": spec.sub_class.value if spec else "",
                    "vol_anualizada": cls.annualized_volatility(r),
                    "retorno_anualizado": cls.annualized_return(r),
                    "beta": cls.beta(r, returns[market_col]) if market_col in returns else np.nan,
                    "max_drawdown": cls.max_drawdown(prices[col]) if col in prices else np.nan,
                    "sharpe": cls.sharpe_ratio(r, rf),
                    "duracion_mod": spec.modified_duration if spec else np.nan,
                }
            )

        # --- Tabla de salida ---
        return pd.DataFrame(rows).set_index("ticker")


# ==============================================================================
# 9. MOTOR DE OPTIMIZACIÓN CON RESTRICCIONES POR CLASE DE ACTIVO
# ==============================================================================

@dataclass
class GroupConstraint:
    """Banda de asignación [min, max] para un grupo de activos."""

    label: str
    min_weight: float = 0.0
    max_weight: float = 1.0

    def validate(self) -> None:
        if not (0.0 <= self.min_weight <= self.max_weight <= 1.0):
            raise ValueError(
                f"Banda inválida para '{self.label}': "
                f"[{self.min_weight}, {self.max_weight}] debe cumplir 0 ≤ min ≤ max ≤ 1."
            )


class PortfolioOptimizer:
    """
    Optimización de portafolios por Máximo Sharpe (Markowitz clásico) bajo
    restricción de no-cortaje (long-only, w_i >= 0), suma de pesos = 1,
    límite máximo por activo y **bandas por clase de activo**:

        Σ w = 1
        0 ≤ w_i ≤ w_max,i
        min_g ≤ Σ_{i ∈ g} w_i ≤ max_g       (p. ej. 40% ≤ Σ w_RV ≤ 60%)

    Todo se resuelve con `scipy.optimize.minimize` (SLSQP): la igualdad de
    presupuesto entra como restricción 'eq' y cada banda de grupo como dos
    restricciones 'ineq'.

    La matriz de covarianza se estima por defecto con shrinkage de
    Ledoit-Wolf (2004), que combina la covarianza muestral con un target
    estructurado (identidad escalada) usando la intensidad δ óptima en
    sentido de error cuadrático medio:

        Σ_shrunk = (1 - δ)·S + δ·(tr(S)/p)·I

    Esto corrige el mal condicionamiento de S cuando el número de activos
    es grande frente al número de observaciones, que es la causa de que
    Markowitz produzca soluciones de esquina extremas.

    `expected_returns` permite sobrescribir μ por activo: los activos de RF
    usan su μ analítico (YTM + roll-down) en lugar de la media muestral, que
    para un bono es una estimación ruidosa y sin fundamento económico.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        rf: float = 0.0,
        max_weight: Union[float, Mapping[str, float]] = 0.30,
        min_weight: float = 0.0,
        trading_days: int = TRADING_DAYS,
        shrinkage: bool = True,
        asset_groups: Optional[Mapping[str, str]] = None,
        group_constraints: Optional[Sequence[GroupConstraint]] = None,
        expected_returns: Optional[Union[pd.Series, Mapping[str, float]]] = None,
        mu_estimator: Optional[str] = None,
    ) -> None:
        # --- Validación de entradas y estimador de μ ---
        if returns.empty:
            raise ValueError("La matriz de retornos no puede estar vacía.")
        self.mu_estimator = mu_estimator or ESTIMADOR_MU
        if self.mu_estimator not in ("bayes_stein", "muestral"):
            raise ValueError(f"mu_estimator debe ser 'bayes_stein' o 'muestral', no '{self.mu_estimator}'.")

        # --- Datos base y parámetros ---
        self.returns = returns.dropna(how="any")
        self.assets = list(self.returns.columns)
        self.n = len(self.assets)
        self.rf = rf
        self.rf_log_ = float(np.log1p(rf))
        self.min_weight = min_weight
        self.trading_days = trading_days
        self.shrinkage = shrinkage

        # --- Límites individuales (escalar o por activo) ---
        if isinstance(max_weight, Mapping):
            self.max_weights = np.array([float(max_weight.get(a, 1.0)) for a in self.assets])
        else:
            self.max_weights = np.repeat(float(max_weight), self.n)
        self.max_weight = float(np.max(self.max_weights))

        # --- Agrupación por clase de activo ---
        self.asset_groups = dict(asset_groups) if asset_groups else {}
        self.group_constraints = [gc for gc in (group_constraints or [])]
        for gc in self.group_constraints:
            gc.validate()
        self._group_index = self._build_group_index()
        self._validate_feasibility()

        # --- Momentos: covarianza ---
        self.cov_, self.shrinkage_intensity_ = MarketDataPipeline.covariance_matrix(
            self.returns, shrinkage=shrinkage, trading_days=trading_days
        )
        self.cov_df_ = pd.DataFrame(self.cov_, index=self.assets, columns=self.assets)

        # --- Momentos: retorno esperado (μ) ---
        override = (pd.Series(expected_returns, dtype=float).reindex(self.assets).dropna()
                    if expected_returns is not None else pd.Series(dtype=float))
        self.mu_sample_series_ = pd.Series(self.returns.mean().values * trading_days,
                                           index=self.assets, name="mu_muestral")
        mu_series = self.mu_sample_series_.copy()
        self.mu_shrinkage_: Dict[str, float] = {}
        if self.mu_estimator == "bayes_stein":
            mu_series = self._bayes_stein(mu_series, exclude=set(override.index))
        mu_series.update(override)
        self.mu_ = mu_series.values
        self.mu_series_ = pd.Series(self.mu_, index=self.assets, name="mu")
        self.result_: Optional[optimize.OptimizeResult] = None

    # ------------------------------------------------------------------ #
    # ESTIMADOR DE μ (BAYES-STEIN)
    # ------------------------------------------------------------------ #
    def _bayes_stein(self, mu: pd.Series, exclude: Set[str]) -> pd.Series:
        """
        Estimador de Bayes-Stein (Jorion, 1986) para los activos cuyo μ sale de
        la media muestral. Dentro de cada clase de activo contrae las medias
        hacia la media del portafolio de mínima varianza de esa clase:

            μ_BS = (1 − φ)·μ̂ + φ·μ₀·1
            μ₀   = 1'Σ⁻¹μ̂ / 1'Σ⁻¹1
            φ    = (N + 2) / [(N + 2) + T·(μ̂ − μ₀1)'Σ⁻¹(μ̂ − μ₀1)]

        con Σ = covarianza diaria·(T − 1)/(T − N − 2). Con una ventana corta
        (T pequeño) las diferencias entre medias muestrales son casi todo ruido
        y φ se acerca a 1; con más historia pesan más los datos. Se usa la
        covarianza del optimizador (Ledoit-Wolf si está activo) por estabilidad
        numérica.
        """
        # --- Agrupación de activos por clase ---
        out = mu.copy()
        T = len(self.returns)
        cov_daily = self.cov_ / self.trading_days
        groups: Dict[str, List[int]] = {}
        for i, a in enumerate(self.assets):
            if a not in exclude:
                groups.setdefault(self.asset_groups.get(a, "SIN_CLASE"), []).append(i)

        # --- Contracción de las medias dentro de cada clase ---
        for label, idx in groups.items():
            n = len(idx)
            if n < 3 or T <= n + 2:
                continue
            sigma = cov_daily[np.ix_(idx, idx)] * (T - 1) / (T - n - 2)
            inv = np.linalg.pinv(sigma)
            ones = np.ones(n)
            m = mu.iloc[idx].to_numpy() / self.trading_days
            mu0 = float(ones @ inv @ m / (ones @ inv @ ones))
            d = m - mu0
            phi = float(np.clip((n + 2) / ((n + 2) + T * float(d @ inv @ d)), 0.0, 1.0))
            out.iloc[idx] = ((1.0 - phi) * m + phi * mu0) * self.trading_days
            self.mu_shrinkage_[label] = phi
        return out

    # ------------------------------------------------------------------ #
    # RESTRICCIONES DE GRUPO
    # ------------------------------------------------------------------ #
    def _build_group_index(self) -> Dict[str, np.ndarray]:
        """Posiciones (índices de columna) de los activos de cada grupo restringido."""
        index: Dict[str, np.ndarray] = {}
        for gc in self.group_constraints:
            members = [i for i, a in enumerate(self.assets) if self.asset_groups.get(a) == gc.label]
            index[gc.label] = np.asarray(members, dtype=int)
        return index

    # ------------------------------------------------------------------ #
    def _validate_feasibility(self) -> None:
        """Detecta bandas imposibles antes de invocar al solver."""
        # --- Capacidad de cada banda ---
        total_min, total_max = 0.0, 0.0
        for gc in self.group_constraints:
            idx = self._group_index[gc.label]
            capacity = float(self.max_weights[idx].sum()) if len(idx) else 0.0
            if gc.min_weight > capacity + 1e-9:
                raise ValueError(
                    f"Banda infactible: '{gc.label}' exige mínimo {gc.min_weight:.0%} pero sus "
                    f"{len(idx)} activos sólo admiten {capacity:.0%} con el límite por activo vigente."
                )
            total_min += gc.min_weight
            total_max += min(gc.max_weight, capacity)

        # --- Consistencia global de las bandas ---
        if self.group_constraints:
            if total_min > 1.0 + 1e-9:
                raise ValueError(f"Bandas infactibles: los mínimos suman {total_min:.0%} > 100%.")
            if total_max < 1.0 - 1e-9:
                raise ValueError(f"Bandas infactibles: los máximos suman {total_max:.0%} < 100%.")

    # ------------------------------------------------------------------ #
    def _constraints(self) -> List[Dict[str, object]]:
        # --- Presupuesto: Σ w = 1 ---
        cons: List[Dict[str, object]] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # --- Bandas por clase: min ≤ Σ w_g ≤ max ---
        for gc in self.group_constraints:
            idx = self._group_index[gc.label]
            if len(idx) == 0:
                continue
            cons.append({"type": "ineq", "fun": lambda w, i=idx, lo=gc.min_weight: float(w[i].sum() - lo)})
            cons.append({"type": "ineq", "fun": lambda w, i=idx, hi=gc.max_weight: float(hi - w[i].sum())})
        return cons

    # ------------------------------------------------------------------ #
    # PUNTOS INICIALES FACTIBLES
    # ------------------------------------------------------------------ #
    def _initial_weights(self) -> np.ndarray:
        """
        Punto inicial factible: presupuesto por grupo en el punto medio de su
        banda (reescalado a 1) y reparto equiponderado dentro del grupo con
        tope por activo. SLSQP converge mucho mejor desde un punto factible.
        """
        # --- Sin bandas: reparto equiponderado con tope ---
        if not self.group_constraints:
            w = np.minimum(np.repeat(1.0 / self.n, self.n), self.max_weights)
            return w / w.sum()

        # --- Presupuesto por grupo en el punto medio de su banda ---
        labels = [gc.label for gc in self.group_constraints]
        lo = np.array([gc.min_weight for gc in self.group_constraints])
        hi = np.array([min(gc.max_weight, float(self.max_weights[self._group_index[l]].sum()))
                       for gc, l in zip(self.group_constraints, labels)])
        budget = np.clip((lo + hi) / 2.0, lo, hi)

        # --- Ajuste del residual respetando las bandas ---
        for _ in range(50):
            gap = 1.0 - budget.sum()
            if abs(gap) < 1e-10:
                break
            slack = (hi - budget) if gap > 0 else (budget - lo)
            total_slack = slack.sum()
            if total_slack <= 1e-12:
                break
            budget = budget + gap * slack / total_slack

        # --- Reparto dentro de cada grupo con tope por activo ---
        w = np.zeros(self.n)
        assigned = np.zeros(self.n, dtype=bool)
        for label, b in zip(labels, budget):
            idx = self._group_index[label]
            if len(idx) == 0:
                continue
            share = np.minimum(b / len(idx), self.max_weights[idx])
            deficit = b - share.sum()
            if deficit > 1e-12:
                room = self.max_weights[idx] - share
                if room.sum() > 1e-12:
                    share = share + deficit * room / room.sum()
            w[idx] = share
            assigned[idx] = True

        # --- Activos sin grupo restringido: reciben el remanente ---
        rest = np.where(~assigned)[0]
        remainder = max(1.0 - w.sum(), 0.0)
        if len(rest) and remainder > 1e-12:
            w[rest] = np.minimum(remainder / len(rest), self.max_weights[rest])
        total = w.sum()
        return w / total if total > 0 else np.repeat(1.0 / self.n, self.n)

    def _random_feasible_weights(self, rng: np.random.Generator) -> np.ndarray:
        """Peso aleatorio que respeta las bandas por clase y el tope por activo."""
        # --- Sin bandas: Dirichlet con tope ---
        if not self.group_constraints:
            w = rng.dirichlet(np.ones(self.n))
            w = np.minimum(w, self.max_weights)
            return w / w.sum()

        # --- Presupuesto aleatorio por grupo dentro de su banda ---
        labels = [gc.label for gc in self.group_constraints]
        lo = np.array([gc.min_weight for gc in self.group_constraints])
        hi = np.array([gc.max_weight for gc in self.group_constraints])
        budget = lo + rng.random(len(lo)) * (hi - lo)
        budget = np.clip(budget, lo, hi)
        budget = budget / budget.sum() if budget.sum() > 0 else budget
        budget = np.clip(budget, lo, hi)
        budget = budget / budget.sum()

        # --- Reparto Dirichlet dentro de cada grupo ---
        w = np.zeros(self.n)
        for label, b in zip(labels, budget):
            idx = self._group_index[label]
            if len(idx) == 0:
                continue
            inner = rng.dirichlet(np.ones(len(idx)))
            inner = np.minimum(inner * b, self.max_weights[idx])
            if inner.sum() > 0:
                inner = inner * (b / inner.sum())
            w[idx] = np.minimum(inner, self.max_weights[idx])
        total = w.sum()
        return w / total if total > 0 else np.repeat(1.0 / self.n, self.n)

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
        return -(ret - self.rf_log_) / vol

    def is_feasible(self, w: np.ndarray, tol: float = 1e-6) -> bool:
        """Verifica presupuesto, topes individuales y bandas por clase."""
        if abs(float(np.sum(w)) - 1.0) > 1e-6:
            return False
        if np.any(w < -tol) or np.any(w > self.max_weights + tol):
            return False
        for gc in self.group_constraints:
            idx = self._group_index[gc.label]
            if len(idx) == 0:
                continue
            g = float(w[idx].sum())
            if g < gc.min_weight - tol or g > gc.max_weight + tol:
                return False
        return True

    def max_sharpe(self, n_restarts: int = 3, seed: int = 11) -> pd.Series:
        """
        Resuelve el portafolio de máximo Sharpe Ratio (tangencia) bajo
        restricciones long-only, límite máximo por activo y bandas por clase de
        activo, usando SLSQP. La tasa libre de riesgo (rf) debe derivarse
        externamente (ej. de la curva TES o la IBR overnight).

        El ratio de Sharpe no es convexo en w, y con muchos activos SLSQP puede
        detenerse con 'positive directional derivative'. Ante un fallo se
        reintenta desde puntos iniciales factibles aleatorios (`n_restarts`)
        antes de caer al punto medio de las bandas.
        """
        # --- Límites, restricciones y puntos iniciales ---
        bounds = [(self.min_weight, float(mx)) for mx in self.max_weights]
        constraints = self._constraints()
        w0 = self._initial_weights()

        rng = np.random.default_rng(seed)
        starts = [w0] + [self._random_feasible_weights(rng) for _ in range(max(n_restarts, 0))]

        # --- Resolución con SLSQP (reinicios si no converge) ---
        best_w: Optional[np.ndarray] = None
        best_obj = np.inf
        for start in starts:
            result = optimize.minimize(
                self._neg_sharpe,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            self.result_ = result
            if not result.success:
                continue
            w = np.clip(result.x, 0.0, None)
            total = w.sum()
            if total <= 0:
                continue
            w = w / total
            if self.is_feasible(w) and result.fun < best_obj:
                best_w, best_obj = w, float(result.fun)
                break

        # --- Respaldo: punto medio factible de las bandas ---
        if best_w is None:
            logger.warning(
                "Max Sharpe: SLSQP no convergió en %d intentos. Usando el punto medio factible "
                "de las bandas.", len(starts),
            )
            best_w = w0 / w0.sum()

        # --- Salida ---
        return pd.Series(best_w, index=self.assets, name="max_sharpe")

    # ------------------------------------------------------------------ #
    # DIAGNÓSTICO DE LA SOLUCIÓN
    # ------------------------------------------------------------------ #
    def group_exposure(self, weights: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Exposición agregada por grupo (clase de activo) de un vector de pesos."""
        w = weights if isinstance(weights, pd.Series) else pd.Series(weights, index=self.assets)
        groups = pd.Series({a: self.asset_groups.get(a, "SIN_CLASE") for a in w.index})
        return w.groupby(groups).sum().sort_index()

    # ------------------------------------------------------------------ #
    def constraints_check(self, weights: pd.Series, tol: float = 1e-6) -> pd.DataFrame:
        """Verificación explícita del cumplimiento de cada banda."""
        exposure = self.group_exposure(weights)
        rows = []
        for gc in self.group_constraints:
            val = float(exposure.get(gc.label, 0.0))
            rows.append({
                "clase": gc.label, "min": gc.min_weight, "peso": val, "max": gc.max_weight,
                "cumple": bool(gc.min_weight - tol <= val <= gc.max_weight + tol),
            })
        return pd.DataFrame(rows).set_index("clase")

    # ------------------------------------------------------------------ #
    # VISUALIZACIONES
    # ------------------------------------------------------------------ #
    def plot_efficient_frontier(self, n_portfolios: int = 5000, seed: int = 7) -> go.Figure:
        """
        Simula portafolios aleatorios long-only (Monte Carlo) **dentro de las
        bandas por clase de activo** para trazar la nube de riesgo-retorno y la
        frontera eficiente, superponiendo el portafolio de Máximo Sharpe.
        """
        # --- Simulación Monte Carlo de portafolios factibles ---
        rng = np.random.default_rng(seed)
        results = np.zeros((3, n_portfolios))
        for i in range(n_portfolios):
            w = self._random_feasible_weights(rng)
            ret, vol = self._portfolio_perf(w)
            sharpe = (ret - self.rf_log_) / vol if vol > 0 else 0.0
            results[:, i] = [ret, vol, sharpe]

        # --- Nube de portafolios simulados ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results[1], y=results[0], mode="markers",
            marker=dict(
                size=5, color=results[2], colorscale="Viridis", opacity=0.5,
                colorbar=dict(title="Sharpe"),
            ),
            name="Portafolios simulados (dentro de bandas)", hoverinfo="skip",
        ))

        # --- Portafolio de Máximo Sharpe ---
        w_opt = self.max_sharpe()
        ret, vol = self._portfolio_perf(w_opt.values)
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers", name="Máximo Sharpe",
            marker=dict(symbol="star", size=18, color="red", line=dict(width=1, color="white")),
        ))

        # --- Formato ---
        bands = " | ".join(
            f"{gc.label}: {gc.min_weight:.0%}–{gc.max_weight:.0%}" for gc in self.group_constraints
        )
        fig.update_layout(
            title=("Frontera Eficiente Multi-Activo — Simulación Monte Carlo"
                   + (f"<br><sup>Bandas estratégicas: {bands}</sup>" if bands else "")),
            xaxis_title="Volatilidad Anualizada", yaxis_title="Retorno Esperado Anualizado",
        )
        return fig


# ==============================================================================
# 10. BACKTESTING WALK-FORWARD MULTI-ACTIVO
# ==============================================================================

CLASS_COLORS: Dict[str, str] = {
    AssetClass.RENTA_VARIABLE.value: "#2E6F9E",
    AssetClass.RENTA_FIJA.value: "#C77B30",
}
SUBCLASS_COLORS: Dict[str, str] = {
    AssetSubClass.RV_ACCION_LOCAL.value: "#1F4E79",
    AssetSubClass.RV_ETF_LOCAL.value: "#2E6F9E",
    AssetSubClass.RV_ETF_GLOBAL.value: "#71A6CE",
    AssetSubClass.RF_NODO_TES.value: "#8C4B10",
    AssetSubClass.RF_ETF.value: "#C77B30",
    AssetSubClass.RF_BONO.value: "#E8B577",
}


class WalkForwardBacktester:
    """
    Simula el rebalanceo histórico fuera de muestra de un portafolio
    Multi-Activo: en cada fecha de rebalanceo estima pesos óptimos usando
    únicamente datos hasta ese momento (ventana de lookback) y aplica esos
    pesos a los retornos *futuros* hasta el siguiente rebalanceo (evita
    look-ahead bias).

    Cada rebalanceo respeta las bandas estratégicas por clase de activo
    (RV vs RF), y puede recibir un μ táctico dependiente de la fecha
    (`expected_returns_fn`) — típicamente la YTM + roll-down leída de la curva
    TES vigente en ese momento — así como una tasa libre de riesgo variable
    (`rf_fn`).

    El universo invertible también puede variar por fecha:
    `exclude_fn(fecha, inicio_ventana)` devuelve los activos que no son
    elegibles en ese rebalanceo (p. ej. bonos fuera de la banda de plazo o
    acciones ilíquidas en la ventana), un activo sin historia completa en la
    ventana se descarta siempre, y `window_fn(fecha, ventana)`
    permite ajustar la ventana de estimación (p. ej. retornos pro-forma de los
    bonos con sus características vigentes).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        lookback_days: int = 252,
        rebalance_freq: str = "ME",
        rf: float = 0.0,
        max_weight: Union[float, Mapping[str, float]] = 0.30,
        shrinkage: bool = True,
        asset_specs: Optional[Mapping[str, AssetSpec]] = None,
        group_constraints: Optional[Sequence[GroupConstraint]] = None,
        expected_returns_fn: Optional[Callable[[pd.Timestamp], pd.Series]] = None,
        rf_fn: Optional[Callable[[pd.Timestamp], float]] = None,
        exclude_fn: Optional[Callable[[pd.Timestamp, pd.Timestamp], Set[str]]] = None,
        window_fn: Optional[Callable[[pd.Timestamp, pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        # --- Datos y parámetros del backtest ---
        self.prices = prices.dropna(how="all")
        self.returns = MarketDataPipeline.compute_returns(self.prices, method="log")
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq
        self.rf = rf
        self.max_weight = max_weight
        self.shrinkage = shrinkage
        self.asset_specs = dict(asset_specs) if asset_specs else {}
        self.group_constraints = list(group_constraints or [])
        self.expected_returns_fn = expected_returns_fn
        self.rf_fn = rf_fn
        self.exclude_fn = exclude_fn
        self.window_fn = window_fn

        # --- Agrupación por clase y sub-clase ---
        self.asset_groups: Dict[str, str] = {
            tk: spec.asset_class.value for tk, spec in self.asset_specs.items()
        }
        self.asset_subgroups: Dict[str, str] = {
            tk: spec.sub_class.value for tk, spec in self.asset_specs.items()
        }

        # --- Resultados (se llenan en run()) ---
        self.weights_history_: Optional[pd.DataFrame] = None
        self.class_weights_history_: Optional[pd.DataFrame] = None
        self.subclass_weights_history_: Optional[pd.DataFrame] = None
        self.rf_weights_history_: Optional[pd.DataFrame] = None
        self.portfolio_returns_: Optional[pd.Series] = None
        self.equity_curve_: Optional[pd.Series] = None
        self.rf_used_: Optional[pd.Series] = None
        self.excluded_history_: Dict[pd.Timestamp, Set[str]] = {}
        self.mu_shrinkage_history_: Dict[pd.Timestamp, Dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    # PREPARACIÓN DE CADA REBALANCEO
    # ------------------------------------------------------------------ #
    def _rebalance_dates(self) -> List[pd.Timestamp]:
        idx = self.returns.index
        dates = pd.Series(idx, index=idx).resample(self.rebalance_freq).last().dropna()
        valid = [pd.Timestamp(d) for d in dates if (idx <= d).sum() >= self.lookback_days]
        return valid

    # ------------------------------------------------------------------ #
    def _in_sample(self, reb_date: pd.Timestamp) -> pd.DataFrame:
        """Ventana de estimación con sólo los activos invertibles en `reb_date`."""
        window = self.returns.loc[:reb_date].tail(self.lookback_days)
        excluded = set(self.exclude_fn(reb_date, window.index[0])) if self.exclude_fn else set()
        excluded |= set(window.columns[window.isna().any()])
        self.excluded_history_[reb_date] = excluded & set(window.columns)
        window = window.drop(columns=sorted(self.excluded_history_[reb_date]))
        return self.window_fn(reb_date, window) if self.window_fn else window

    # ------------------------------------------------------------------ #
    def _build_optimizer(
        self, window_returns: pd.DataFrame, reb_date: pd.Timestamp
    ) -> PortfolioOptimizer:
        # --- Bandas aplicables a las clases presentes ---
        cols = set(window_returns.columns)
        active_labels = {self.asset_groups.get(c) for c in cols}
        constraints = [gc for gc in self.group_constraints if gc.label in active_labels]

        # --- μ táctico y tasa libre de riesgo a la fecha ---
        mu_override = self.expected_returns_fn(reb_date) if self.expected_returns_fn else None
        rf_t = self.rf_fn(reb_date) if self.rf_fn else self.rf

        # --- Optimizador ---
        return PortfolioOptimizer(
            window_returns,
            rf=rf_t,
            max_weight=self.max_weight,
            shrinkage=self.shrinkage,
            asset_groups=self.asset_groups,
            group_constraints=constraints,
            expected_returns=mu_override,
        )

    # ------------------------------------------------------------------ #
    def _optimize_weights(
        self, window_returns: pd.DataFrame, reb_date: pd.Timestamp
    ) -> Tuple[pd.Series, float]:
        opt = self._build_optimizer(window_returns, reb_date)
        self.mu_shrinkage_history_[reb_date] = dict(opt.mu_shrinkage_)
        return opt.max_sharpe(), opt.rf

    # ------------------------------------------------------------------ #
    def _fallback_weights(self, window_returns: pd.DataFrame, reb_date: pd.Timestamp) -> pd.Series:
        """Pesos factibles (punto inicial de las bandas) cuando la optimización falla."""
        try:
            opt = self._build_optimizer(window_returns, reb_date)
            w = opt._initial_weights()
            return pd.Series(w, index=window_returns.columns)
        except Exception:
            n = window_returns.shape[1]
            return pd.Series(np.repeat(1.0 / n, n), index=window_returns.columns)

    # ------------------------------------------------------------------ #
    # EJECUCIÓN WALK-FORWARD
    # ------------------------------------------------------------------ #
    def run(self) -> pd.DataFrame:
        """Ejecuta la simulación walk-forward completa y retorna el historial de pesos."""
        # --- Fechas de rebalanceo ---
        rebal_dates = self._rebalance_dates()
        if len(rebal_dates) < 2:
            raise RuntimeError(
                "Historial insuficiente para el lookback_days especificado; "
                "reduzca lookback_days o amplíe el rango de fechas."
            )

        # --- Acumuladores ---
        weight_records: Dict[pd.Timestamp, pd.Series] = {}
        rf_records: Dict[pd.Timestamp, float] = {}
        daily_portfolio_returns: List[pd.Series] = []

        self.excluded_history_ = {}
        self.mu_shrinkage_history_ = {}

        # --- Bucle walk-forward ---
        for i, reb_date in enumerate(rebal_dates):
            # --- Estimación dentro de muestra y optimización ---
            in_sample = self._in_sample(reb_date)
            try:
                w, rf_t = self._optimize_weights(in_sample, reb_date)
            except Exception as exc:
                logger.warning("Optimización falló en %s (%s). Usando pesos factibles de banda.",
                               reb_date.date(), exc)
                w = self._fallback_weights(in_sample, reb_date)
                rf_t = self.rf_fn(reb_date) if self.rf_fn else self.rf

            weight_records[reb_date] = w
            rf_records[reb_date] = rf_t

            # --- Aplicación de los pesos fuera de muestra ---
            next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else self.returns.index[-1]
            out_of_sample = self.returns.loc[reb_date:next_date].iloc[1:]
            if not out_of_sample.empty:
                simple_ret = np.expm1(out_of_sample[w.index]) @ w.values
                daily_portfolio_returns.append(np.log1p(simple_ret))

        # --- Consolidación de resultados ---
        self.weights_history_ =pd.DataFrame(weight_records).T.fillna(0.0)
        self.rf_used_ = pd.Series(rf_records, name="rf")
        self.portfolio_returns_ = pd.concat(daily_portfolio_returns).sort_index()
        self.equity_curve_ = np.exp(self.portfolio_returns_.cumsum())
        self._aggregate_exposures()
        return self.weights_history_

    # ------------------------------------------------------------------ #
    def _aggregate_exposures(self) -> None:
        """Agrega los pesos por clase y sub-clase de activo a lo largo del tiempo."""
        wh = self.weights_history_
        if wh is None:
            return
        cls = pd.Series({c: self.asset_groups.get(c, "SIN_CLASE") for c in wh.columns})
        sub = pd.Series({c: self.asset_subgroups.get(c, "SIN_CLASE") for c in wh.columns})
        self.class_weights_history_ = wh.T.groupby(cls).sum().T
        self.subclass_weights_history_ = wh.T.groupby(sub).sum().T
        rf_cols = [c for c in wh.columns if self.asset_groups.get(c) == AssetClass.RENTA_FIJA.value]
        self.rf_weights_history_ = wh[rf_cols]

    # ------------------------------------------------------------------ #
    # MÉTRICAS DE DESEMPEÑO
    # ------------------------------------------------------------------ #
    def performance_summary(self) -> Dict[str, float]:
        """Resumen de desempeño del backtest fuera de muestra."""
        if self.portfolio_returns_ is None:
            raise RuntimeError("Ejecute run() antes de solicitar el resumen de desempeño.")
        # --- Métricas de desempeño ---
        r = self.portfolio_returns_
        summary = {
            "retorno_total": float(self.equity_curve_.iloc[-1] - 1),
            "retorno_anualizado": RiskMetrics.annualized_return(r),
            "volatilidad_anualizada": RiskMetrics.annualized_volatility(r),
            "sharpe_ratio": RiskMetrics.sharpe_ratio(r, float(self.rf_used_.mean())),
            "max_drawdown": RiskMetrics.max_drawdown(self.equity_curve_),
            "num_rebalanceos": float(self.weights_history_.shape[0]),
        }

        # --- Peso medio por clase ---
        if self.class_weights_history_ is not None:
            for cls_label, serie in self.class_weights_history_.items():
                summary[f"peso_medio_{cls_label}"] = float(serie.mean())

        # --- Intensidad media del shrinkage de μ ---
        if self.mu_shrinkage_history_:
            phi = pd.DataFrame(self.mu_shrinkage_history_).T
            for cls_label, serie in phi.items():
                summary[f"shrinkage_mu_medio_{cls_label}"] = float(serie.mean())
        return summary

    # ------------------------------------------------------------------ #
    def band_compliance(self, tol: float = 1e-6) -> pd.DataFrame:
        """Verifica el cumplimiento de las bandas estratégicas en cada rebalanceo."""
        if self.class_weights_history_ is None:
            raise RuntimeError("Ejecute run() antes de verificar bandas.")
        rows = []
        for gc in self.group_constraints:
            serie = self.class_weights_history_.get(gc.label)
            if serie is None:
                continue
            ok = ((serie >= gc.min_weight - tol) & (serie <= gc.max_weight + tol))
            rows.append({
                "clase": gc.label, "banda_min": gc.min_weight, "banda_max": gc.max_weight,
                "peso_min_obs": float(serie.min()), "peso_medio": float(serie.mean()),
                "peso_max_obs": float(serie.max()),
                "rebalanceos_en_banda": f"{int(ok.sum())}/{len(ok)}",
            })
        return pd.DataFrame(rows).set_index("clase")

    # ------------------------------------------------------------------ #
    # VISUALIZACIONES
    # ------------------------------------------------------------------ #
    def plot_equity_curve(self) -> go.Figure:
        """Grafica interactiva (Plotly) de la curva de equity acumulada del backtest."""
        if self.equity_curve_ is None:
            raise RuntimeError("Ejecute run() antes de graficar.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.equity_curve_.index, y=self.equity_curve_.values,
            mode="lines", line=dict(width=2, color="darkgreen"), name="Equity Multi-Activo",
        ))
        fig.add_hline(y=1.0, line=dict(color="gray", dash="dash", width=1))
        fig.update_layout(
            title="Curva de Equity — Backtest Walk-Forward Multi-Activo (Máximo Sharpe con bandas)",
            xaxis_title="Fecha", yaxis_title="Valor del Portafolio (base = 1.0)",
        )
        return fig

    # ------------------------------------------------------------------ #
    def plot_weights_evolution(self) -> go.Figure:
        """Evolución de pesos por activo, apilada y agrupada por clase."""
        if self.weights_history_ is None:
            raise RuntimeError("Ejecute run() antes de graficar.")
        wh = self.weights_history_
        ordered = sorted(wh.columns, key=lambda c: (self.asset_groups.get(c, "Z"),
                                                    self.asset_subgroups.get(c, "Z"), c))

        # --- Áreas apiladas por activo ---
        fig = go.Figure()
        for col in ordered:
            sub = self.asset_subgroups.get(col, "SIN_CLASE")
            fig.add_trace(go.Scatter(
                x=wh.index, y=wh[col] * 100, mode="lines", stackgroup="pesos",
                name=col, legendgroup=sub, line=dict(width=0.5),
                fillcolor=SUBCLASS_COLORS.get(sub),
                hovertemplate=f"<b>{col}</b> ({sub})<br>%{{x|%Y-%m}}: %{{y:.2f}}%<extra></extra>",
            ))
        fig.update_layout(
            title="Evolución de Pesos por Activo (color = sub-clase)",
            xaxis_title="Fecha de rebalanceo", yaxis_title="Peso (%)",
        )
        return fig

    # ------------------------------------------------------------------ #
    def plot_asset_class_evolution(self) -> go.Figure:
        """
        Asignación estratégica en el tiempo: área apilada Total RV vs Total RF,
        con las bandas de política superpuestas.
        """
        if self.class_weights_history_ is None:
            raise RuntimeError("Ejecute run() antes de graficar.")
        # --- Áreas apiladas por clase ---
        cw = self.class_weights_history_
        fig = go.Figure()
        for cls_label in [c for c in [AssetClass.RENTA_FIJA.value, AssetClass.RENTA_VARIABLE.value]
                          if c in cw.columns] + [c for c in cw.columns if c not in CLASS_COLORS]:
            fig.add_trace(go.Scatter(
                x=cw.index, y=cw[cls_label] * 100, mode="lines", stackgroup="clase",
                name=f"Total {cls_label}", line=dict(width=0.5),
                fillcolor=CLASS_COLORS.get(cls_label),
                hovertemplate=f"<b>Total {cls_label}</b><br>%{{x|%Y-%m}}: %{{y:.2f}}%<extra></extra>",
            ))

        # --- Bandas de política superpuestas ---
        for gc in self.group_constraints:
            if gc.label not in cw.columns:
                continue
            for bound, dash in [(gc.min_weight, "dot"), (gc.max_weight, "dash")]:
                fig.add_hline(
                    y=bound * 100, line=dict(color=CLASS_COLORS.get(gc.label, "gray"), dash=dash, width=1),
                    annotation_text=f"{gc.label} {bound:.0%}", annotation_position="right",
                )
        fig.update_layout(
            title="Asignación Estratégica por Clase de Activo — Renta Fija vs Renta Variable",
            xaxis_title="Fecha de rebalanceo", yaxis_title="Peso (%)", yaxis_range=[0, 100],
        )
        return fig

    # ------------------------------------------------------------------ #
    def plot_fixed_income_breakdown(self, normalize: bool = False) -> go.Figure:
        """
        Composición interna de la Renta Fija por vía de implementación:
        Nodos TES (A), ETFs/FICs (B) y bonos individuales seleccionados (C).
        """
        if self.subclass_weights_history_ is None:
            raise RuntimeError("Ejecute run() antes de graficar.")

        # --- Datos de las sub-clases de RF (opcionalmente normalizados) ---
        rf_subs = [s.value for s in (AssetSubClass.RF_NODO_TES, AssetSubClass.RF_ETF, AssetSubClass.RF_BONO)]
        cols = [c for c in rf_subs if c in self.subclass_weights_history_.columns]
        data = self.subclass_weights_history_[cols]
        if normalize:
            total = data.sum(axis=1).replace(0.0, np.nan)
            data = data.div(total, axis=0).fillna(0.0)

        # --- Barras apiladas ---
        fig = go.Figure()
        for col in cols:
            fig.add_trace(go.Bar(
                x=data.index, y=data[col] * 100, name=col,
                marker_color=SUBCLASS_COLORS.get(col),
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m}}: %{{y:.2f}}%<extra></extra>",
            ))
        suffix = " (% dentro de la RF)" if normalize else " (% del portafolio total)"
        fig.update_layout(
            barmode="stack",
            title="Composición Interna de la Renta Fija — Nodos TES / ETFs / Bonos" + suffix,
            xaxis_title="Fecha de rebalanceo", yaxis_title="Peso (%)",
        )
        return fig

    # ------------------------------------------------------------------ #
    def plot_fixed_income_assets(self) -> go.Figure:
        """Detalle título a título de la pata de Renta Fija."""
        if self.rf_weights_history_ is None:
            raise RuntimeError("Ejecute run() antes de graficar.")
        # --- Orden por sub-clase ---
        rfw = self.rf_weights_history_
        ordered = sorted(rfw.columns, key=lambda c: (self.asset_subgroups.get(c, "Z"), c))

        # --- Áreas apiladas por instrumento ---
        fig = go.Figure()
        for col in ordered:
            sub = self.asset_subgroups.get(col, "SIN_CLASE")
            spec = self.asset_specs.get(col)
            label = spec.name if spec else col
            fig.add_trace(go.Scatter(
                x=rfw.index, y=rfw[col] * 100, mode="lines", stackgroup="rf",
                name=col, legendgroup=sub, line=dict(width=0.5),
                hovertemplate=(f"<b>{label}</b><br>{sub}<br>"
                               "%{x|%Y-%m}: %{y:.2f}%<extra></extra>"),
            ))
        fig.update_layout(
            title="Renta Fija — Detalle por Instrumento (nodos TES, ETFs y bonos aprobados)",
            xaxis_title="Fecha de rebalanceo", yaxis_title="Peso sobre el portafolio total (%)",
        )
        return fig


# ==============================================================================
# 11. POLÍTICA DE ASIGNACIÓN Y REPORTE INTERACTIVO
# ==============================================================================

@dataclass
class AllocationPolicy:
    """
    Declaración de política de inversión: bandas estratégicas por clase de
    activo y topes de concentración diferenciados por tipo de instrumento.

    Los defaults provienen del bloque de PARÁMETROS EDITABLES de la cabecera.
    """

    banda_rv: Tuple[float, float] = field(default_factory=lambda: BANDA_RV)
    banda_rf: Tuple[float, float] = field(default_factory=lambda: BANDA_RF)

    max_peso_accion: float = field(default_factory=lambda: MAX_PESO_ACCION_INDIVIDUAL)
    max_peso_etf_rv: float = field(default_factory=lambda: MAX_PESO_ETF_RV)
    max_peso_etf_rf: float = field(default_factory=lambda: MAX_PESO_ETF_RF)
    max_peso_nodo_tes: float = field(default_factory=lambda: MAX_PESO_NODO_TES)
    max_peso_bono: float = field(default_factory=lambda: MAX_PESO_BONO_INDIVIDUAL)

    rf_tenor_years: float = field(default_factory=lambda: RF_TENOR_YEARS)

    # ------------------------------------------------------------------ #
    def group_constraints(self) -> List[GroupConstraint]:
        return [
            GroupConstraint(AssetClass.RENTA_VARIABLE.value, *self.banda_rv),
            GroupConstraint(AssetClass.RENTA_FIJA.value, *self.banda_rf),
        ]

    # ------------------------------------------------------------------ #
    def _cap_for(self, spec: AssetSpec) -> float:
        return {
            AssetSubClass.RV_ACCION_LOCAL: self.max_peso_accion,
            AssetSubClass.RV_ETF_LOCAL: self.max_peso_etf_rv,
            AssetSubClass.RV_ETF_GLOBAL: self.max_peso_etf_rv,
            AssetSubClass.RF_NODO_TES: self.max_peso_nodo_tes,
            AssetSubClass.RF_ETF: self.max_peso_etf_rf,
            AssetSubClass.RF_BONO: self.max_peso_bono,
        }[spec.sub_class]

    def max_weights(self, specs: Mapping[str, AssetSpec]) -> Dict[str, float]:
        """Tope individual de cada activo según su tipo de instrumento."""
        return {tk: self._cap_for(spec) for tk, spec in specs.items()}

    # ------------------------------------------------------------------ #
    def validate(self, specs: Mapping[str, AssetSpec]) -> None:
        """
        Comprueba que cada banda sea alcanzable con los topes vigentes antes de
        llegar al solver, y avisa si el mínimo exige concentrar el universo.
        """
        # --- Capacidad de cada clase frente a su banda ---
        caps = self.max_weights(specs)
        for cls, (lo, hi) in ((AssetClass.RENTA_VARIABLE, self.banda_rv),
                              (AssetClass.RENTA_FIJA, self.banda_rf)):
            miembros = [tk for tk, sp in specs.items() if sp.asset_class is cls]
            capacidad = sum(caps[tk] for tk in miembros)
            if capacidad < lo - 1e-9:
                raise ValueError(
                    f"Política infactible: la banda mínima de {cls.value} es {lo:.0%} pero sus "
                    f"{len(miembros)} activos sólo admiten {capacidad:.0%} con los topes vigentes. "
                    f"Suba los topes o amplíe el universo de {cls.value}."
                )
            if capacidad < hi:
                logger.warning(
                    "La banda máxima de %s (%.0f%%) supera la capacidad del universo (%.0f%%): "
                    "el tope efectivo será %.0f%%.", cls.value, hi * 100, capacidad * 100,
                    capacidad * 100,
                )

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        return (f"RV {self.banda_rv[0]:.0%}–{self.banda_rv[1]:.0%} | "
                f"RF {self.banda_rf[0]:.0%}–{self.banda_rf[1]:.0%} | "
                f"topes: acción {self.max_peso_accion:.0%}, ETF RV {self.max_peso_etf_rv:.0%}, "
                f"ETF RF {self.max_peso_etf_rf:.0%}, nodo TES {self.max_peso_nodo_tes:.0%}, "
                f"bono {self.max_peso_bono:.0%}")


class InteractiveReport:
    """Construye el reporte HTML interactivo (Plotly) en una sola página."""

    _CSS = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
         padding:24px 32px 64px;background:#fafafa;color:#1a1a1a;}
    h1{margin-bottom:4px;} .subtitle{color:#666;margin-top:0;}
    section{margin-top:48px;} h2{border-bottom:2px solid #ddd;padding-bottom:8px;}
    .note{background:#fff8e6;border-left:4px solid #e0a800;padding:12px 16px;
          margin:16px 0;font-size:14px;line-height:1.5;}
    table{border-collapse:collapse;font-size:13px;background:#fff;margin-top:8px;}
    th,td{border:1px solid #e3e3e3;padding:6px 10px;text-align:right;}
    th{background:#f0f2f5;font-weight:600;text-align:center;}
    td:first-child,th:first-child{text-align:left;}
    tr:nth-child(even) td{background:#fbfbfc;}
    .wrap{overflow-x:auto;}
    """

    def __init__(self, title: str, subtitle: str) -> None:
        self.title = title
        self.subtitle = subtitle
        self._blocks: List[str] = []
        self._plotly_included = False

    # ------------------------------------------------------------------ #
    def add_chart(self, title: str, fig: go.Figure) -> "InteractiveReport":
        include_js = "cdn" if not self._plotly_included else False
        self._plotly_included = True
        self._blocks.append(
            f"<section><h2>{title}</h2>"
            + pio.to_html(fig, include_plotlyjs=include_js, full_html=False)
            + "</section>"
        )
        return self

    # ------------------------------------------------------------------ #
    def add_table(self, title: str, df: pd.DataFrame, float_format: str = "{:.4f}") -> "InteractiveReport":
        html = df.to_html(float_format=lambda v: float_format.format(v), border=0, na_rep="—")
        self._blocks.append(f"<section><h2>{title}</h2><div class='wrap'>{html}</div></section>")
        return self

    # ------------------------------------------------------------------ #
    def add_note(self, text: str) -> "InteractiveReport":
        self._blocks.append(f"<div class='note'>{text}</div>")
        return self

    # ------------------------------------------------------------------ #
    def write(self, path: str) -> str:
        # --- Ensamble del documento HTML ---
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{self.title}</title><style>{self._CSS}</style></head><body>"
            f"<h1>{self.title}</h1><p class='subtitle'>{self.subtitle}</p>"
            + "\n".join(self._blocks)
            + "</body></html>"
        )

        # --- Escritura a disco ---
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path


# ==============================================================================
# 12. EJECUCIÓN PRINCIPAL — DEMOSTRACIÓN INTEGRAL DEL MOTOR MULTI-ACTIVO
# ==============================================================================

def main() -> None:
    os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)

    logger.info("=" * 78)
    logger.info("QUANT PM — MOTOR DE ASSET ALLOCATION MULTI-ACTIVO (RV + RF)")
    logger.info("=" * 78)

    universe = AssetUniverse()
    policy = AllocationPolicy()
    start_date, end_date = FECHA_INICIO, FECHA_FIN

    # ---------------------------------------------------------------- #
    # 1) PRECIOS DE MERCADO (RV + ETFs de RF — Enfoque B)
    # ---------------------------------------------------------------- #
    # --- Descarga de precios y retornos ---
    logger.info("Descargando/generando precios para %d tickers de mercado.", len(universe.all_tickers))
    pipeline = MarketDataPipeline(
        universe.all_tickers + [universe.benchmark], start_date, end_date, seed=SEMILLA_ALEATORIA
    )
    prices = pipeline.fetch_prices()
    returns = pipeline.compute_returns(prices, method="log")
    logger.info("Precios obtenidos: %d filas x %d activos.", *prices.shape)

    # --- Selección del universo (datos reales vs. modo demostración) ---
    # Sólo se decide aquí lo que no depende de la fecha: tener precio real. La
    # historia, la cotización reciente y la liquidez se evalúan en cada
    # rebalanceo con la ventana de estimación (`tradability_issues_at`).
    def _reales(tickers: Sequence[str]) -> List[str]:
        return [tk for tk in tickers if tk not in pipeline.synthetic_tickers_]

    modo_demo = len(_reales(universe.rv_tickers)) < MIN_ACTIVOS_RV_REALES
    if modo_demo:
        logger.warning(
            "MODO DEMOSTRACIÓN: sólo %d tickers de RV con precio real (< %d). Se optimiza sobre "
            "series sintéticas calibradas; los resultados no son inferencia de mercado.",
            len(_reales(universe.rv_tickers)), MIN_ACTIVOS_RV_REALES,
        )
    if EXCLUIR_ACTIVOS_SIN_PRECIO_REAL and not modo_demo:
        rv_tickers, rf_etf_tickers = _reales(universe.rv_tickers), _reales(universe.rf_etf_tickers)
    else:
        rv_tickers, rf_etf_tickers = list(universe.rv_tickers), list(universe.rf_etf_tickers)
    tickers_mercado = rv_tickers + rf_etf_tickers

    # --- Registro de los activos sin precio real ---
    liquidez = pipeline.liquidity_report()
    descartados = sorted(set(universe.rv_tickers + universe.rf_etf_tickers) - set(tickers_mercado))
    motivos_exclusion = {tk: "sin precio real observable (serie sintética)" for tk in descartados}
    if descartados:
        logger.warning("Excluidos del universo invertible: %s",
                       "; ".join(f"{tk} ({motivos_exclusion[tk]})" for tk in descartados))

    # --- Tickers listados después del inicio (no recortan la matriz) ---
    min_start_buffer = pd.Timestamp(start_date) + pd.Timedelta(days=HOLGURA_COBERTURA_DIAS)
    listados_tarde = [tk for tk in tickers_mercado
                      if prices[tk].first_valid_index() is None
                      or prices[tk].first_valid_index() > min_start_buffer]
    if not rf_etf_tickers:
        logger.info("Sin ETFs/FICs de RF con precio observable: el Enfoque B queda vacío en esta corrida.")

    # ---------------------------------------------------------------- #
    # 2) CURVA TES — TASA LIBRE DE RIESGO Y BASE DE LA RENTA FIJA
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("CURVA TES — TASA LIBRE DE RIESGO Y NODOS DE DURACIÓN")

    # --- Histórico diario de la curva (Banrep) ---
    curve_history: Optional[pd.DataFrame] = None
    plazos_curva = sorted(set(float(t) for t in CURVA_TES_PLAZOS) | set(float(t) for t in universe.nodos_tes))
    if CURVA_TES_FUENTE == "banrep":
        try:
            nodos_banrep = BanrepTESHistory(RUTA_CACHE_CURVA_TES).fetch(
                pd.Timestamp(start_date) - pd.Timedelta(days=15), end_date
            )
            curve_history = BanrepTESHistory.complete_curve(nodos_banrep, plazos_curva, CURVA_TES_TAU_NS)
        except Exception as exc:
            logger.warning(
                "No se pudo obtener la curva TES histórica de Banrep (%s). Se usa la curva simulada: "
                "los resultados de renta fija no reflejan el mercado.", exc,
            )
    elif CURVA_TES_FUENTE != "simulada":
        raise ValueError(f"CURVA_TES_FUENTE debe ser 'banrep' o 'simulada', no '{CURVA_TES_FUENTE}'.")

    # --- Curva base al inicio de la ventana ---
    if curve_history is not None:
        fila_inicio =curve_history.loc[:start_date]
        fila_inicio = fila_inicio.iloc[-1] if not fila_inicio.empty else curve_history.iloc[0]
        curve = TESYieldCurve.from_levels(plazos_curva, fila_inicio.values)
        logger.info("Curva TES: histórico real de Banrep (1, 5 y 10 años, completado con Nelson-Siegel "
                    "τ=%.2f), %d días.", CURVA_TES_TAU_NS, len(curve_history))
    else:
        curve = TESYieldCurve.from_config()
        logger.info("Curva TES: curva fija de configuración con choques simulados.")

    # ---------------------------------------------------------------- #
    # 3) MOTOR DE RENTA FIJA — ENFOQUES A + B + C
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("RENTA FIJA — NODOS TES (A) + ETFs/FICs (B) + BONOS FILTRADOS (C)")

    # --- Construcción del motor de renta fija ---
    criteria = BondScreeningCriteria()
    fi_engine = FixedIncomeEngine(
        curve=curve,
        node_tenors=universe.nodos_tes,
        criteria=criteria,
        as_of=start_date,
    )
    fi_bundle = fi_engine.build(
        dates=prices.index,
        market_returns=returns[universe.benchmark],
        curve_history=curve_history,
    )

    # --- Tasa libre de riesgo ---
    rf_serie = fi_engine.shocks.levels_at([policy.rf_tenor_years]).iloc[:, 0]
    rf_tes = float(rf_serie.iloc[-1])
    rf_medio = float(rf_serie.mean())
    logger.info(
        "Tasa libre de riesgo: TES a %.2g año(s) = %.4f%% E.A. al cierre (%.4f%% al inicio, "
        "%.4f%% en promedio)", policy.rf_tenor_years, rf_tes * 100, rf_serie.iloc[0] * 100, rf_medio * 100,
    )

    # --- Salida en consola: screening y analítica de RF ---
    print("\n--- Screening de bonos individuales (Enfoque C) ---")
    print(fi_bundle.screening_report[
        ["emisor", "rating", "plazo_anios", "ytm", "duracion_mod", "spread_bp", "aprobado", "motivo_rechazo"]
    ].round(4).to_string())
    print("\n--- Analítica de activos de RF modelada (Enfoques A y C) ---")
    print(fi_bundle.analytics.round(4).to_string())

    # ---------------------------------------------------------------- #
    # 4) MATRIZ UNIFICADA MULTI-ACTIVO
    # ---------------------------------------------------------------- #
    # --- Fichas de todos los activos (mercado + RF modelada) ---
    specs: Dict[str, AssetSpec] = {}
    market_specs = universe.market_specs()
    for tk in tickers_mercado:
        specs[tk] = market_specs[tk]
    specs.update(fi_bundle.specs)

    # --- Precios y retornos unificados ---
    multi_prices = MarketDataPipeline.build_multi_asset_prices(
        prices[tickers_mercado], fi_bundle.prices, late_start=listados_tarde
    )
    multi_returns = MarketDataPipeline.compute_returns(multi_prices, method="log")
    n_rv = sum(1 for s in specs.values() if s.asset_class is AssetClass.RENTA_VARIABLE)
    n_rf = len(specs) - n_rv
    logger.info(
        "Matriz unificada: %d días x %d activos (%d RV / %d RF).",
        *multi_returns.shape, n_rv, n_rf,
    )

    # ---------------------------------------------------------------- #
    # 5) MÉTRICAS DE RIESGO MICRO
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("MÉTRICAS DE RIESGO MICRO POR ACTIVO")

    # --- Reporte de riesgo por activo ---
    risk_report = RiskMetrics.asset_risk_report(
        pd.concat([prices, fi_bundle.prices], axis=1, sort=True).ffill(),
        pd.concat([multi_returns, returns[[universe.benchmark]]], axis=1, sort=True)
        .loc[multi_returns.index[0]:].dropna(subset=[universe.benchmark]),
        universe.benchmark, rf_medio, specs,
    )
    print("\n" + risk_report.round(4).to_string())

    # ---------------------------------------------------------------- #
    # 6) OPTIMIZACIÓN ESTRATÉGICA CON BANDAS POR CLASE DE ACTIVO
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("OPTIMIZACIÓN MULTI-ACTIVO — MÁXIMO SHARPE CON BANDAS (%s)", policy.describe())
    policy.validate(specs)

    # --- μ analítico de la RF al cierre ---
    fecha_cierre = multi_returns.index[-1]
    mu_rf_hoy = np.log1p(fi_engine.expected_returns_at(fecha_cierre))

    # --- Universo invertible al cierre: bonos y activos de mercado ---
    bonos_no_elegibles = fi_engine.excluded_bonds_at(fecha_cierre)
    if bonos_no_elegibles:
        logger.info("Bonos no elegibles al %s (plazo, spread o emisor): %s",
                    fecha_cierre.date(), sorted(bonos_no_elegibles))
    motivos_cierre = pipeline.tradability_issues_at(fecha_cierre, multi_returns.index[0], tickers_mercado)
    if motivos_cierre:
        logger.info("Activos de mercado no invertibles al %s (ventana %s–%s): %s",
                    fecha_cierre.date(), multi_returns.index[0].date(), fecha_cierre.date(),
                    "; ".join(f"{tk} ({m})" for tk, m in sorted(motivos_cierre.items())))

    # --- Retornos pro-forma del universo invertible ---
    no_invertibles_cierre = (bonos_no_elegibles | set(motivos_cierre)
                             | set(multi_returns.columns[multi_returns.isna().any()]))
    retornos_estrategicos = fi_engine.proforma_log_returns(
        fecha_cierre,
        multi_returns.drop(columns=sorted(no_invertibles_cierre & set(multi_returns.columns))),
    )

    # --- Optimización de Máximo Sharpe ---
    optimizer = PortfolioOptimizer(
        retornos_estrategicos,
        rf=rf_tes,
        max_weight=policy.max_weights(specs),
        shrinkage=USAR_SHRINKAGE_LEDOIT_WOLF,
        asset_groups={tk: s.asset_class.value for tk, s in specs.items()},
        group_constraints=policy.group_constraints(),
        expected_returns=mu_rf_hoy,
    )
    weights_opt = optimizer.max_sharpe()
    ret_opt, vol_opt = optimizer._portfolio_perf(weights_opt.values)

    # --- Tabla de pesos y cumplimiento de bandas ---
    tabla_pesos = pd.DataFrame({
        "peso_%": (weights_opt * 100).round(2),
        "clase": [specs[t].asset_class.value for t in weights_opt.index],
        "sub_clase": [specs[t].sub_class.value for t in weights_opt.index],
    }).sort_values("peso_%", ascending=False)
    print("\n--- Portafolio estratégico óptimo ---")
    print(tabla_pesos[tabla_pesos["peso_%"] > 0.01].to_string())
    print("\n--- Exposición por clase de activo ---")
    print((optimizer.group_exposure(weights_opt) * 100).round(2).to_string())
    print("\n--- Cumplimiento de bandas ---")
    print(optimizer.constraints_check(weights_opt).to_string())

    # --- Diagnóstico de shrinkage y métricas del óptimo ---
    logger.info(
        "Intensidad de shrinkage Ledoit-Wolf: δ = %.4f (0 = covarianza muestral pura, 1 = target)",
        optimizer.shrinkage_intensity_,
    )
    for clase, phi in optimizer.mu_shrinkage_.items():
        logger.info("Bayes-Stein sobre μ de %s: φ = %.3f (0 = media muestral, 1 = media común).",
                    clase, phi)
    logger.info(
        "Óptimo — retorno esp.: %.2f%% | vol: %.2f%% | Sharpe: %.4f | activos con peso > 0: %d",
        ret_opt * 100, vol_opt * 100, (ret_opt - optimizer.rf_log_) / vol_opt, int((weights_opt > 1e-4).sum()),
    )

    # ---------------------------------------------------------------- #
    # 7) BACKTEST WALK-FORWARD MULTI-ACTIVO
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("BACKTEST WALK-FORWARD FUERA DE MUESTRA CON BANDAS RV/RF")

    # --- Universo invertible en cada rebalanceo (sólo datos de la ventana) ---
    motivos_por_fecha: Dict[pd.Timestamp, Dict[str, str]] = {}

    def no_invertibles(fecha: pd.Timestamp, inicio_ventana: pd.Timestamp) -> Set[str]:
        motivos = pipeline.tradability_issues_at(fecha, inicio_ventana, tickers_mercado)
        motivos_por_fecha[pd.Timestamp(fecha)] = motivos
        return set(motivos) | fi_engine.excluded_bonds_at(fecha)

    # --- Configuración y ejecución del backtest ---
    backtester = WalkForwardBacktester(
        multi_prices,
        lookback_days=LOOKBACK_DIAS,
        rebalance_freq=FRECUENCIA_REBALANCEO,
        rf=rf_tes,
        max_weight=policy.max_weights(specs),
        shrinkage=USAR_SHRINKAGE_LEDOIT_WOLF,
        asset_specs=specs,
        group_constraints=policy.group_constraints(),
        expected_returns_fn=lambda d: np.log1p(fi_engine.expected_returns_at(d)),
        rf_fn=lambda d: fi_engine.risk_free_at(d, policy.rf_tenor_years),
        exclude_fn=no_invertibles,
        window_fn=fi_engine.proforma_log_returns,
    )
    weights_history = backtester.run()
    perf_summary = backtester.performance_summary()

    # --- Elegibilidad por activo a lo largo del backtest ---
    def _tabla_elegibilidad(tickers: Sequence[str]) -> pd.DataFrame:
        elegibles = pd.DataFrame({
            d: {tk: tk not in excl for tk in tickers} for d, excl in backtester.excluded_history_.items()
        }).T
        fechas_elegible = {tk: elegibles.index[elegibles[tk]] for tk in tickers}
        return pd.DataFrame({
            "rebalanceos_elegible": {tk: f"{int(elegibles[tk].sum())}/{len(elegibles)}" for tk in tickers},
            "primer_rebalanceo": {tk: (f[0].date() if len(f) else "—") for tk, f in fechas_elegible.items()},
            "ultimo_rebalanceo": {tk: (f[-1].date() if len(f) else "—") for tk, f in fechas_elegible.items()},
            "peso_medio_%_cuando_elegible": {
                tk: (float(weights_history.loc[f, tk].mean() * 100) if len(f) and tk in weights_history else 0.0)
                for tk, f in fechas_elegible.items()
            },
        })

    bonos = [tk for tk, sp in specs.items() if sp.sub_class is AssetSubClass.RF_BONO]
    tabla_elegibilidad = _tabla_elegibilidad(bonos)
    tabla_elegibilidad.insert(0, "vencimiento", {tk: specs[tk].metadata["vencimiento"] for tk in bonos})

    # --- Universo de mercado: motivo de exclusión más frecuente y al cierre ---
    def _categoria(motivo: str) -> str:
        return motivo.split(":")[0].split(" desde")[0]

    tabla_universo = _tabla_elegibilidad(tickers_mercado)
    ultima_fecha = max(motivos_por_fecha)
    tabla_universo["motivo_principal"] = {
        tk: (pd.Series([_categoria(m[tk]) for m in motivos_por_fecha.values() if tk in m]).mode().iloc[0]
             if any(tk in m for m in motivos_por_fecha.values()) else "—")
        for tk in tickers_mercado
    }
    tabla_universo["estado_ultimo_rebalanceo"] = {
        tk: motivos_por_fecha[ultima_fecha].get(tk, "elegible") for tk in tickers_mercado
    }
    for tk in tickers_mercado:
        motivos_exclusion.setdefault(tk, f"elegible en {tabla_universo.loc[tk, 'rebalanceos_elegible']} rebalanceos")

    # --- Salida en consola del backtest ---
    print("\n--- Asignación por clase de activo en cada rebalanceo (%) ---")
    print((backtester.class_weights_history_ * 100).round(2).to_string())
    print("\n--- Composición interna de la Renta Fija (%) ---")
    rf_subs = [s.value for s in (AssetSubClass.RF_NODO_TES, AssetSubClass.RF_ETF, AssetSubClass.RF_BONO)]
    print((backtester.subclass_weights_history_[
        [c for c in rf_subs if c in backtester.subclass_weights_history_.columns]
    ] * 100).round(2).to_string())
    print("\n--- Cumplimiento de bandas a lo largo del backtest ---")
    print(backtester.band_compliance().to_string())
    print("\n--- Desempeño fuera de muestra ---")
    print(pd.Series(perf_summary, name="valor").round(4).to_string())
    print("\n--- Elegibilidad de bonos en el backtest (screening dinámico) ---")
    print(tabla_elegibilidad.round(2).to_string())
    print("\n--- Universo de mercado en el backtest (liquidez y cobertura en la ventana de cada rebalanceo) ---")
    print(tabla_universo.round(2).to_string())

    # ---------------------------------------------------------------- #
    # 8) REPORTE HTML INTERACTIVO
    # ---------------------------------------------------------------- #

    # --- Cabecera y notas metodológicas ---
    report = InteractiveReport(
        "Quant PM — Motor de Asset Allocation Multi-Activo (RV + RF)",
        f"Universo: {n_rv} activos de Renta Variable y {n_rf} de Renta Fija | "
        f"Rango: {start_date} a {end_date} | Política: {policy.describe()}",
    )
    n_nodos = sum(1 for sp in specs.values() if sp.sub_class is AssetSubClass.RF_NODO_TES)
    n_bonos = sum(1 for sp in specs.values() if sp.sub_class is AssetSubClass.RF_BONO)
    nota_b = (f"<b>(B)</b> {len(rf_etf_tickers)} ETF/FIC con precio líquido de mercado"
              if rf_etf_tickers else
              "<b>(B)</b> sin ETFs/FICs en esta corrida — ningún vehículo del universo tiene "
              "precio observable en la fuente de datos")
    report.add_note(
        "<b>Curva TES.</b> "
        + ("Histórico diario real de las tasas cero cupón TES en pesos a 1, 5 y 10 años del Banco "
           "de la República (SUAMECA), completado a todos los plazos con Nelson-Siegel "
           f"(τ = {CURVA_TES_TAU_NS:g} años). Los spreads de crédito de los bonos siguen siendo "
           "simulados: no hay una fuente pública de precios históricos por título."
           if fi_engine.shocks.source_ == "histórica" else
           "<b>Simulada</b> alrededor de una curva fija de configuración: los resultados de renta "
           "fija no reflejan episodios reales del mercado.")
    )
    report.add_note(
        "<b>Arquitectura de Renta Fija.</b> La pata de RF combina tres vías: "
        f"<b>(A)</b> {n_nodos} nodos sintéticos de la curva TES, cuyo retorno diario se modela por "
        "duración modificada y convexidad — ΔP/P ≈ −D·Δy + ½·C·(Δy)²; "
        f"{nota_b}; y "
        f"<b>(C)</b> {n_bonos} bonos individuales que superan el filtro de rating, plazo, liquidez "
        "y spread. El μ de (A) y (C) es analítico (YTM + roll-down), no una media muestral, y se "
        "recalcula con la curva vigente en cada rebalanceo. Los bonos envejecen: se revalúan a "
        "diario con su plazo residual, el screening se repite en cada rebalanceo (un título entra "
        "o sale según su plazo y spread vigentes) y su covarianza se estima con retornos pro-forma "
        "que aplican los choques históricos a su duración actual."
    )
    report.add_note(
        "<b>Integridad de datos.</b> "
        + ("Se excluyeron del universo los activos sin precio real: "
           + "; ".join(f"<b>{tk}</b>" for tk in descartados) + ". " if descartados else "")
        + "La historia, la cotización reciente y la liquidez se evalúan en cada rebalanceo sólo con "
        f"los precios de su ventana de estimación ({LOOKBACK_DIAS} días): un título entra o sale "
        "según su comportamiento hasta esa fecha, sin usar información posterior. El motor no "
        "sustituye precios faltantes por series simuladas dentro del optimizador, ni admite títulos "
        "cuyo precio congelado haría parecer su riesgo menor al real."
    )

    # --- Sección: curva TES y renta fija ---
    curva_cierre = fi_engine.shocks.curve_at(multi_returns.index[-1])
    report.add_chart(f"Curva TES — Estructura Temporal (ETTI) al {multi_returns.index[-1].date()}",
                     curva_cierre.plot_curve())
    report.add_chart("Curva TES — Evolución por nodo", fi_engine.shocks.plot_curve_history())
    report.add_chart("Enfoque C — Screening de bonos individuales", fi_engine.screener.plot_screening())
    report.add_table(
        "Analítica de la Renta Fija modelada (μ, YTM, duración, convexidad)",
        fi_bundle.analytics.drop(columns=[c for c in ("nombre",) if c in fi_bundle.analytics.columns]),
    )
    report.add_table(
        "Screening de bonos — veredicto por título (al inicio de la ventana)",
        fi_bundle.screening_report[
            ["emisor", "rating", "plazo_anios", "ytm", "duracion_mod", "spread_bp",
             "liquidez", "aprobado", "motivo_rechazo"]
        ],
    )
    report.add_table(
        "Screening dinámico — elegibilidad de cada bono en los rebalanceos",
        tabla_elegibilidad, "{:.2f}",
    )
    report.add_table(
        "Universo de mercado — elegibilidad por liquidez y cobertura en los rebalanceos",
        tabla_universo, "{:.2f}",
    )

    # --- Sección: optimización estratégica ---
    report.add_chart("Frontera Eficiente Multi-Activo", optimizer.plot_efficient_frontier(n_portfolios=3000))
    report.add_table("Portafolio estratégico óptimo (peso > 0.01%)",
                     tabla_pesos[tabla_pesos["peso_%"] > 0.01], "{:.2f}")
    report.add_table("Cumplimiento de bandas — portafolio estratégico",
                     optimizer.constraints_check(weights_opt), "{:.4f}")

    # --- Sección: backtest walk-forward ---
    report.add_chart("Asignación Estratégica: Total RF vs Total RV",
                     backtester.plot_asset_class_evolution())
    report.add_chart("Composición Interna de la Renta Fija (Nodos TES / ETFs / Bonos)",
                     backtester.plot_fixed_income_breakdown())
    report.add_chart("Renta Fija — Peso relativo dentro de la clase",
                     backtester.plot_fixed_income_breakdown(normalize=True))
    report.add_chart("Renta Fija — Detalle por instrumento", backtester.plot_fixed_income_assets())
    report.add_chart("Evolución de Pesos por Activo", backtester.plot_weights_evolution())
    report.add_chart("Curva de Equity — Backtest Walk-Forward", backtester.plot_equity_curve())
    report.add_table("Desempeño fuera de muestra", pd.DataFrame(perf_summary, index=["valor"]).T)

    # --- Sección: riesgo micro y liquidez ---
    report.add_table("Métricas de riesgo micro por activo", risk_report)
    if not liquidez.empty:
        tabla_liquidez = liquidez.assign(
            primer_dato=liquidez["primer_dato"].dt.date,
            ultimo_dato=liquidez["ultimo_dato"].dt.date,
            estado=[motivos_exclusion.get(tk, "—") for tk in liquidez.index],
        ).sort_values("pct_sin_movimiento", ascending=False)
        report.add_table("Liquidez y cobertura de precios por ticker (ventana completa)",
                         tabla_liquidez, "{:.3f}")

    # --- Escritura del reporte ---
    report_path = report.write(os.path.join(DIRECTORIO_SALIDA, "reporte_interactivo.html"))

    logger.info("=" * 78)
    logger.info("Ejecución completa. Reporte interactivo: %s", report_path)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
