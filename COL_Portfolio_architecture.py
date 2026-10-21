from __future__ import annotations

# ==============================================================================
# IMPORTACIONES Y CONFIGURACIÓN GLOBAL
# ==============================================================================

# --- Librería estándar --------------------------------------------------------
import logging
import os
import ssl
import sys
import warnings
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

# --- Cómputo numérico y estadístico ------------------------------------------
import numpy as np
import pandas as pd
from scipy import interpolate, optimize, signal, sparse
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
#                  ⚙  CONFIGURACIÓN (SET UP) — TODO LO EDITABLE VIVE AQUÍ
# ------------------------------------------------------------------------------
# Todo lo que se puede ajustar está en este bloque; el resto del archivo es el
# motor y no necesita tocarse. Las secciones van en el orden en que se usan:
#
#   1. Modo de ejecución y capital
#   2. Ventana de análisis (fechas)
#   3. Política de datos reales y alertas de calidad
#   4. Universo de Renta Variable
#   5. Universo de Renta Fija: ETFs y nodos TES
#   6. Universo de Renta Fija: BONOS INDIVIDUALES (los llenas tú)
#   7. Curva TES (tasa libre de riesgo y base de la renta fija)
#   8. Criterio de optimización y objetivos (Sharpe, varianza, CVaR)
#   9. Restricciones del portafolio (libres por defecto)
#  10. Estimación de retornos y riesgo
#  11. Backtest (sólo modo "validacion")
#  12. Liquidez y cobertura de precios
#  13. Parámetros del modelo de simulación
#  14. Salida
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. MODO DE EJECUCIÓN Y CAPITAL
# ------------------------------------------------------------------------------
# "recomendacion": calcula el portafolio óptimo con los datos más recientes y
#                  lo muestra con montos en pesos. Es el modo para invertir hoy.
# "validacion":    además de lo anterior, corre el backtest walk-forward para
#                  comprobar cómo se habría comportado el método en el pasado.
MODO_EJECUCION: str = "recomendacion"

# Capital a invertir (COP) para convertir los pesos del portafolio en montos.
# None = sólo muestra porcentajes.
CAPITAL_INVERSION_COP: Optional[float] = 100_000_000.0

# ------------------------------------------------------------------------------
# 2. VENTANA DE ANÁLISIS
# ------------------------------------------------------------------------------
# Fecha del último dato. None = hoy. Para reproducir un resultado pasado, fija
# una fecha ("2024-12-31"): las tasas de tus bonos (sección 6) deben ser las de
# ESA fecha.
FECHA_FIN: Optional[str] = None

# Años de historia hacia atrás. Sólo se usa si FECHA_INICIO es None. Debe cubrir al
# menos la ventana del CVaR (CVAR_VENTANA_ANIOS, sección 8); en modo "validacion" lo que
# sobra es el periodo del backtest.
HISTORIA_ANIOS: float = 8.0

# Fecha de inicio fija ("2021-01-01"). None = FECHA_FIN menos HISTORIA_ANIOS.
FECHA_INICIO: Optional[str] = None

# ------------------------------------------------------------------------------
# 3. POLÍTICA DE DATOS REALES Y ALERTAS DE CALIDAD
# ------------------------------------------------------------------------------
# False (recomendado para invertir): si no hay precios reales de suficientes
#   acciones, o no se logra descargar la curva TES del Banco de la República, el
#   programa se DETIENE con un mensaje claro en vez de optimizar con datos
#   inventados.
# True: permite el "modo demostración" con series simuladas (sólo para probar el
#   código sin internet; los resultados NO sirven para invertir).
PERMITIR_DATOS_SINTETICOS: bool = False

# Días calendario máximos entre FECHA_FIN y el último dato disponible (precio
# de mercado o curva TES). Si se supera, el programa avisa; con datos reales y
# PERMITIR_DATOS_SINTETICOS = False, una curva TES más vieja que esto detiene la
# corrida. Cubre fines de semana, festivos y el rezago de publicación de Banrep.
MAX_DIAS_DATOS_DESACTUALIZADOS: int = 7

# Un ticker sin precio real se excluye de toda la corrida.
EXCLUIR_ACTIVOS_SIN_PRECIO_REAL: bool = True

# Mínimo de acciones/ETFs de RV con precio real para considerar los datos válidos.
MIN_ACTIVOS_RV_REALES: int = 5

# ------------------------------------------------------------------------------
# 4. UNIVERSO DE RENTA VARIABLE
# ------------------------------------------------------------------------------
# Tickers de Yahoo Finance. Agrega o quita los que quieras; los que no tengan
# precio real, historia suficiente o liquidez se descartan solos y el reporte
# explica por qué.
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

# Índice de referencia para calcular betas (no se invierte en él).
BENCHMARK: str = "^GSPC"

# ------------------------------------------------------------------------------
# 5. UNIVERSO DE RENTA FIJA: ETFs Y NODOS TES
# ------------------------------------------------------------------------------
# ETFs / fondos de renta fija con precio de mercado en Yahoo Finance.
ETFS_RF_LOCALES: List[str] = []
ETFS_RF_GLOBALES: List[str] = ["TLT"]

# Nodos sintéticos de la curva TES: bonos "par" de plazo constante (en años) que
# representan comprar TES a ese plazo. Son replicables con TES reales.
NODOS_TES: List[float] = [1.0, 3.0, 5.0, 10.0]

# ------------------------------------------------------------------------------
# 6. UNIVERSO DE RENTA FIJA: BONOS INDIVIDUALES  (TÚ LOS LLENAS)
# ------------------------------------------------------------------------------
# Lista vacía = el portafolio se arma sólo con acciones, ETFs y nodos TES.
#
# GUÍA PARA ESCOGER QUÉ BONOS INGRESAR
# Un bono entra al optimizador sólo si cumple los criterios de más abajo. Para no
# perder tiempo, busca bonos que cumplan TODO esto:
#   # Tasa fija en pesos (indexación "TF"). La curva TES del motor es nominal en
#     pesos: un bono en UVR/IPC/IBR no es comparable con ella.
#   # Calificación local AAA o AA+ (BONOS_MIN_RATING). Un TES se ingresa como "AAA".
#   # Plazo residual entre 1 y 10 años desde FECHA_FIN (BONOS_PLAZO_*).
#   # Que tenga una tasa (YTM) o un precio de mercado reciente: Precia, PiP, BVC o
#     tu comisionista. Debe ser la del día FECHA_FIN.
#   # Que se pueda comprar de verdad: emisión grande y que cotice seguido
#     (puntaje de liquidez >= BONOS_MIN_LIQUIDEZ).
#   # Spread sobre la curva TES de hasta 450 pb (BONOS_MAX_SPREAD_BP): spreads más
#     altos indican riesgo de crédito que el calificador no está viendo.
#   # Máximo 2 bonos del mismo emisor (BONOS_MAX_POR_EMISOR) para diversificar.
#
# CAMPOS DE CADA BONO (uno por diccionario)
#   "isin"        OBLIGATORIO. Identificador único del título. En Colombia se usa el
#                 ISIN (p. ej. "COL17CT02622"); si tienes el CUSIP, también sirve:
#                 sólo se usa para identificar y no puede repetirse.
#   "emisor"      OBLIGATORIO. Nombre del emisor ("Bancolombia", "ISA", ...). Se usa
#                 para limitar cuántos bonos del mismo emisor entran.
#   "rating"      OBLIGATORIO. Calificación local: "AAA", "AA+", "AA", "AA-", "A+", ...
#   "cupon"       OBLIGATORIO. Tasa cupón efectiva anual en decimal (7,25% = 0.0725).
#   "vencimiento" OBLIGATORIO. Fecha de vencimiento, formato "AAAA-MM-DD".
#   --- Precio de mercado: pon UNO de estos tres (se usa el primero que aparezca) ---
#   "ytm"         Tasa de mercado (YTM) efectiva anual en decimal (10,5% = 0.105).
#                 Es la mejor opción: es lo que publican los proveedores de precios.
#   "precio"      Precio SUCIO (con cupón corrido) por cada 100 de valor nominal.
#   "spread_bp"   Spread en puntos básicos sobre la curva TES al mismo plazo.
#   --- Opcionales ---
#   "frecuencia"  Pagos de cupón por año: 1 (anual), 2 (semestral), 4 (trimestral).
#                 Por defecto 1.
#   "indexacion"  "TF" (tasa fija), "UVR", "IPC", "IBR". Por defecto "TF".
#   "liquidez"    Puntaje 0 a 1 que asignas tú. Referencia: 0.8-1.0 TES de referencia
#                 o emisión muy grande que cotiza a diario; 0.5-0.7 corporativo AAA
#                 que cotiza varias veces por semana; 0.3-0.5 cotiza poco; < 0.3
#                 casi no se negocia. Por defecto 0.5.
#   "sector"      Texto libre ("Soberano", "Financiero", "Utilities", ...). Sólo informativo.
#
# EJEMPLO. El primer bono es real en lo que se pudo verificar en fuentes del Ministerio
# de Hacienda: TES en pesos emitido el 27-feb-2025 con vencimiento el 27-feb-2030 y
# cupón de 12,50%. NO se pudo verificar su ISIN ni la tasa de hoy: el "isin" es un
# marcador que debes reemplazar por el ISIN real, y la "ytm" (14,79%) es el corte de
# la subasta de mayo de 2026, no la de tu fecha de datos. Actualiza ambos antes de usarlo.
# La plantilla de abajo (comentada) sirve para agregar más bonos; los datos de ISIN,
# cupón, vencimiento y tasa los consigues en el informe de la subasta de TES del
# Ministerio de Hacienda, en la BVC o con tu comisionista (Precia / PiP).
BONOS_UNIVERSO: List[Dict[str, object]] = [
    {
        "isin": "REEMPLAZAR-ISIN-TES-2030-02-27",
        "emisor": "Ministerio de Hacienda (TES)",
        "rating": "AAA",
        "cupon": 0.1250,
        "vencimiento": "2030-02-27",
        "ytm": 0.1479,
        "frecuencia": 1,
        "indexacion": "TF",
        "liquidez": 0.95,
        "sector": "Soberano",
    },
    # {
    #     "isin": "ISIN-DEL-BONO",
    #     "emisor": "Nombre del emisor",
    #     "rating": "AAA",
    #     "cupon": 0.0000,            # decimal, tasa cupón efectiva anual
    #     "vencimiento": "AAAA-MM-DD",
    #     "ytm": 0.0000,              # o "precio" (sucio, por 100) o "spread_bp"
    #     "frecuencia": 1,
    #     "indexacion": "TF",
    #     "liquidez": 0.5,
    #     "sector": "Financiero",
    # },
]

# --- Criterios de selección (filtro de política) ------------------------------
# Un bono que no los cumpla queda fuera del optimizador y el reporte dice por qué.
# Los de plazo, spread y emisor se vuelven a evaluar en cada fecha de decisión.
BONOS_MIN_RATING: str = "AA+"          # Calificación mínima. AA o inferior se rechaza.
BONOS_PLAZO_MIN_ANIOS: float = 1.0     # Plazo residual mínimo; antes de esto el bono "sale" para no llegar al vencimiento.
BONOS_PLAZO_MAX_ANIOS: float = 10.0    # Plazo residual máximo; los bonos más largos tienen mucha duración (riesgo de tasa).
BONOS_MIN_LIQUIDEZ: float = 0.35       # Puntaje de liquidez mínimo (campo "liquidez" de cada bono).
BONOS_MAX_SPREAD_BP: Optional[float] = 450.0  # Spread máximo sobre la curva TES, en pb. None = sin límite.
BONOS_MAX_POR_EMISOR: int = 2          # Máximo de bonos por emisor (si hay más, se queda el de mayor YTM ajustada por liquidez). 0 = sin límite.
BONOS_INDEXACIONES_PERMITIDAS: Optional[Set[str]] = {"TF"}  # Indexaciones aceptadas. None = todas (no recomendado).
BONOS_EMISORES_EXCLUIDOS: Set[str] = set()  # Emisores prohibidos por política, p. ej. {"Avianca"}.

# Volatilidad diaria (pb) del spread de crédito, que se simula porque no hay
# precios históricos por bono; se escala por iliquidez (menos líquido = más
# volátil). Sube el valor si quieres suponer bonos corporativos más riesgosos.
BONOS_VOL_SPREAD_BP_DIARIA: float = 2.0

# ------------------------------------------------------------------------------
# 7. CURVA TES (TASA LIBRE DE RIESGO Y BASE DE LA RENTA FIJA)
# ------------------------------------------------------------------------------
# "banrep":   histórico real de tasas cero cupón TES a 1, 5 y 10 años del Banco de
#             la República, completado a los demás plazos con Nelson-Siegel.
# "simulada": curva fija de abajo con choques simulados (sólo para pruebas).
CURVA_TES_FUENTE: str = "banrep"
CURVA_TES_TAU_NS: float = 1.37
RUTA_CACHE_CURVA_TES: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datos", "tes_cero_cupon_banrep.csv"
)

# Plazo del TES que se toma como tasa libre de riesgo (años).
RF_TENOR_YEARS: float = 1.0

# Curva fija para el modo "simulada" (o CSV de Banrep con columnas plazo_anios, tasa).
CURVA_TES_PLAZOS: List[float] = [0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]
CURVA_TES_TASAS: List[float] = [
    0.0980, 0.0975, 0.0965, 0.0950, 0.0940, 0.0935, 0.0955, 0.0975, 0.0995, 0.1010, 0.1015
]
RUTA_CURVA_TES_CSV: Optional[str] = None

# Incluir el roll-down (ganancia por "rodar" hacia plazos más cortos de la curva)
# en el retorno esperado de los nodos TES y de los bonos.
INCLUIR_ROLLDOWN: bool = True

# ------------------------------------------------------------------------------
# 8. CRITERIO DE OPTIMIZACIÓN Y OBJETIVOS
# ------------------------------------------------------------------------------
# Qué significa "mejor portafolio". Todos son long-only y 100% invertidos, y
# respetan las restricciones de la sección 9 si las activas:
#   "max_sharpe"        Mayor retorno por unidad de riesgo (volatilidad).
#   "min_varianza"      El portafolio menos volátil posible.
#   "max_retorno_vol"   El mayor retorno esperado sin pasar de una volatilidad anual
#                       objetivo (equivale a fijar una varianza objetivo).
#   "min_cvar"          La menor pérdida promedio en el peor (1 - CVAR_NIVEL) de los meses.
#   "max_retorno_cvar"  El mayor retorno esperado sin que el CVaR mensual pase de un objetivo.
CRITERIO_OPTIMIZACION: str = "max_sharpe"

# Objetivos de riesgo (sólo se usan con el criterio que los necesita). None = el
# programa muestra el rango alcanzable y toma el punto medio.
VOLATILIDAD_OBJETIVO_ANUAL: Optional[float] = None   # p. ej. 0.08 = 8% anual
CVAR_OBJETIVO_MENSUAL: Optional[float] = None        # p. ej. 0.04 = pérdida promedio de 4% en el peor 5% de los meses

# Antes de optimizar, el programa muestra el rango alcanzable de volatilidad y
# CVaR y te pregunta por el criterio y los objetivos (Enter = deja lo configurado
# aquí). Sin terminal interactiva (cron, ejecución automática) no pregunta.
PREGUNTAR_ANTES_DE_OPTIMIZAR: bool = True

# --- CVaR (pérdida esperada en la cola) ---------------------------------------
CVAR_NIVEL: float = 0.95               # 0.95 = promedio del peor 5% de los casos.
HORIZONTE_DIAS: int = 21               # Horizonte del portafolio en días hábiles (21 ≈ 1 mes).

# Historia con que se estima la cola. Es más larga que la ventana de retorno y
# covarianza (LOOKBACK_DIAS, sección 10) porque la cola del 5% necesita muchas
# observaciones. Con 5 años hay ~1.260 retornos diarios y ~63 caen en la cola; se
# recomienda al menos CVAR_MIN_OBS_COLA observaciones en la cola (30 -> 2,4 años
# diarios; con retornos semanales harían falta más de 11 años). Los activos que no
# tengan esta historia quedan fuera del universo (el reporte los lista).
CVAR_VENTANA_ANIOS: float = 5.0
CVAR_MIN_OBS_COLA: int = 30
FRECUENCIA_RETORNOS_CVAR: str = "diaria"  # Retornos continuos de los escenarios: "diaria" o "semanal" (poco confiable: 5 años = 260 datos).

# Los escenarios mensuales se simulan con bootstrap por bloques: se toman bloques de
# retornos diarios reales (conservan correlaciones y rachas de volatilidad) y se
# encadenan hasta completar el horizonte. Así el CVaR se apoya en toda la historia
# y no en ~60 meses solapados.
CVAR_ESCENARIOS: int = 3000            # Trayectorias simuladas (más = menos ruido de simulación, más lento).
CVAR_BLOQUE_DIAS: int = 5              # Largo medio de cada bloque, en días.
CVAR_REMUESTREOS_IC: int = 200         # Remuestreos para el intervalo de confianza del CVaR en el reporte.
CVAR_CENTRAR_EN_MU: bool = True        # True: los retornos se recentran en el retorno esperado usado por el
                                       # optimizador, para no heredar la tendencia pasada (p. ej. una acción que
                                       # subió mucho). False: historia tal cual.

# ------------------------------------------------------------------------------
# 9. RESTRICCIONES DEL PORTAFOLIO (LIBRES POR DEFECTO)
# ------------------------------------------------------------------------------
# Por defecto el optimizador decide libremente, sin ventas en corto y 100%
# invertido: escoge los activos que mejor sirvan al criterio de optimización y la
# proporción RV/RF sale del resultado (se muestra en el reporte). Sin topes el
# óptimo suele concentrarse en pocos activos; si quieres forzar diversificación,
# activa alguno de estos límites.

# Banda de peso total por clase de activo: (mínimo, máximo) en decimal, p. ej.
# {"RV": (0.40, 0.60), "RF": (0.40, 0.60)}. {} = sin bandas.
BANDAS_POR_CLASE: Dict[str, Tuple[float, float]] = {}

# Tope de peso por activo individual, en decimal (0.15 = 15%). None = sin tope.
MAX_PESO_POR_ACTIVO: Optional[float] = None

# Tope por tipo de instrumento (tiene prioridad sobre MAX_PESO_POR_ACTIVO).
# Claves: "accion", "etf_rv", "nodo_tes", "etf_rf", "bono". None = usa el tope general.
TOPES_POR_TIPO: Dict[str, Optional[float]] = {
    "accion": None,
    "etf_rv": None,
    "nodo_tes": None,
    "etf_rf": None,
    "bono": None,
}

# --- Número de activos del portafolio ------------------------------------------
# Un óptimo sin límites puede dejar posiciones diminutas (decenas de activos con 0,5%) que no
# se pueden operar de forma práctica, o al revés, concentrarse en 1 o 2 activos. Estos
# parámetros controlan cuántos activos entran.
MIN_ACTIVOS: Optional[int] = 5                 # Mínimo de activos: ningún activo pasará de 1/MIN_ACTIVOS (5 -> 20%). None = sin mínimo.
MAX_ACTIVOS: Optional[int] = 12                # Máximo de activos (10 a 15 es razonable). None = sin límite.
PESO_MIN_POSICION: Optional[float] = 0.03      # Peso mínimo de cada activo elegido (0.03 = 3%). None = sin mínimo.

# Cantidad óptima automática: de los portafolios con entre MIN_ACTIVOS y MAX_ACTIVOS activos se
# toma el que tenga MENOS activos y cuyo objetivo (Sharpe, volatilidad, CVaR o retorno, según el
# criterio) quede a menos de CARDINALIDAD_TOLERANCIA del mejor. El reporte grafica el objetivo
# contra el número de activos para que veas cuánto cuesta cada activo de menos. False = usa el
# portafolio con más activos permitidos (el de mejor objetivo).
# El máximo y el peso mínimo se resuelven con una heurística (se resuelve el óptimo libre y se
# van eliminando los activos de menor peso, re-optimizando): buenas soluciones, pero sin
# garantía de óptimo global, porque el problema es combinatorio.
CARDINALIDAD_AUTOMATICA: bool = True
CARDINALIDAD_TOLERANCIA: float = 0.02          # 0.02 = acepta perder hasta 2% del objetivo por tener menos activos.

# Peso mínimo (%) para que un activo aparezca en las tablas de resultados.
PESO_MIN_REPORTE_PCT: float = 0.1

# ------------------------------------------------------------------------------
# 10. ESTIMACIÓN DE RETORNOS Y RIESGO
# ------------------------------------------------------------------------------
LOOKBACK_DIAS: int = 504              # Días hábiles de historia para estimar retorno y covarianza (504 ≈ 2 años).
                                      # La cola del CVaR usa una ventana más larga: ver CVAR_VENTANA_ANIOS.
USAR_SHRINKAGE_LEDOIT_WOLF: bool = True  # Suaviza la matriz de covarianza (recomendado).
ESTIMADOR_MU: str = "bayes_stein"     # Retorno esperado de acciones/ETFs: "bayes_stein" o "muestral".

# ------------------------------------------------------------------------------
# 11. BACKTEST (sólo en modo "validacion")
# ------------------------------------------------------------------------------
FRECUENCIA_REBALANCEO: str = "ME"     # "ME" = último día hábil de cada mes.
PUNTOS_FRONTERA: int = 25             # Puntos con que se traza la frontera eficiente en el reporte.

# ------------------------------------------------------------------------------
# 12. LIQUIDEZ Y COBERTURA DE PRECIOS
# ------------------------------------------------------------------------------
MAX_PCT_DIAS_SIN_MOVIMIENTO: float = 0.25   # Un activo con más de este % de días sin variación de precio es ilíquido.
MAX_DIAS_PRECIO_CONGELADO: int = 30         # ...o con el precio igual más de estos días seguidos.
MAX_DIAS_RELLENO_PRECIO: int = 5            # Días máximos que se arrastra el último precio (festivos).
HOLGURA_COBERTURA_DIAS: int = 45            # Tolerancia para considerar que un activo "listó tarde".

# ------------------------------------------------------------------------------
# 13. PARÁMETROS DEL MODELO DE SIMULACIÓN
# ------------------------------------------------------------------------------
SEMILLA_ALEATORIA: int = 42

# Curva TES simulada (3 factores: nivel, pendiente, curvatura; pb diarios). Sólo
# se usa con CURVA_TES_FUENTE = "simulada" o PERMITIR_DATOS_SINTETICOS = True.
CURVA_VOL_NIVEL_BP: float = 5.5
CURVA_VOL_PENDIENTE_BP: float = 3.5
CURVA_VOL_CURVATURA_BP: float = 2.5
CURVA_BETA_RV_BP: float = -1.2
CURVA_SIM_TAU: float = 2.5
CURVA_SIM_REVERSION: float = 0.015
CURVA_SIM_PISO: float = 0.005

# ------------------------------------------------------------------------------
# 14. SALIDA
# ------------------------------------------------------------------------------
DIRECTORIO_SALIDA: str = os.environ.get(
    "OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
)
NOMBRE_REPORTE_HTML: str = "reporte_interactivo.html"

# ==============================================================================
#                          FIN DE LA CONFIGURACIÓN
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
    de la CONFIGURACIÓN (set up) en la cabecera del archivo; se pueden sobrescribir
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
                        end=self.end + pd.Timedelta(days=1),  # yfinance trata `end` como exclusivo
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
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Calcula retornos logarítmicos diarios a partir de precios."""
        return np.log(prices).diff().dropna(how="all")

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
        Curva construida desde la CONFIGURACIÓN de la cabecera: usa el CSV de
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
        times = BondAnalytics.coupon_times(maturity_years, freq)
        return float(BondAnalytics.par_coupon(self.get_rate(times), times, freq))

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
        import urllib.error
        import urllib.parse
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
        host = urllib.parse.urlparse(url).hostname
        context = self._context_with_intermediate(host)
        with urllib.request.urlopen(req, timeout=self.timeout, context=context) as resp:
            return resp.read()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _base_ssl_context() -> "ssl.SSLContext":
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    # ------------------------------------------------------------------ #
    def _context_with_intermediate(self, host: str) -> "ssl.SSLContext":
        import re
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
    def duration_convexity(times: np.ndarray, cashflows: np.ndarray, ytm):
        """
        Duración de Macaulay, Duración Modificada y Convexidad bajo tasa E.A.

            D_mod = D_mac / (1 + y)
            C     = Σ t·(t+1)·PV_t / (P · (1+y)²)

        Con `ytm` escalar devuelve tres floats; con `ytm` de forma (m,) y
        `cashflows` de forma (m, T) (un bono por fila) devuelve tres arreglos.
        Donde el precio no es positivo el resultado es NaN.
        """
        ytm = np.asarray(ytm, dtype=float)
        pv = cashflows * (1.0 + ytm[..., None]) ** (-times)
        price = pv.sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            macaulay = (times * pv).sum(axis=-1) / price
            convexity = (times * (times + 1.0) * pv).sum(axis=-1) / (price * (1.0 + ytm) ** 2)
        valid = price > 0
        macaulay, modified, convexity = (
            np.where(valid, x, np.nan) for x in (macaulay, macaulay / (1.0 + ytm), convexity)
        )
        if ytm.ndim == 0:
            return float(macaulay), float(modified), float(convexity)
        return macaulay, modified, convexity

    # ------------------------------------------------------------------ #
    @staticmethod
    def coupon_times(maturity_years: float, freq: int = 1) -> np.ndarray:
        """Fechas de cupón (años) de un bono par al plazo dado: 1/freq, 2/freq, …, T."""
        n = max(int(round(maturity_years * freq)), 1)
        return np.arange(1, n + 1, dtype=float) / freq

    # ------------------------------------------------------------------ #
    @staticmethod
    def par_coupon(zero_rates: np.ndarray, times: np.ndarray, freq: int = 1) -> Union[float, np.ndarray]:
        """
        Cupón par c = (1 − DF(T)) / (Σ_i DF(t_i) / freq) a partir de las tasas
        cero en `times` (última dimensión; admite una curva o una por fecha).
        """
        dfs = (1.0 + np.asarray(zero_rates, dtype=float)) ** (-times)
        return (1.0 - dfs[..., -1]) / (dfs.sum(axis=-1) / freq)

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
    tau: float = field(default_factory=lambda: CURVA_SIM_TAU)
    mean_reversion: float = field(default_factory=lambda: CURVA_SIM_REVERSION)
    equity_beta_bp: float = field(default_factory=lambda: CURVA_BETA_RV_BP)
    floor_rate: float = field(default_factory=lambda: CURVA_SIM_PISO)
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
        innovations = rng.normal(0.0, factor_vols, size=(n, 3))
        factors = signal.lfilter([1.0], [1.0, -(1.0 - cfg.mean_reversion)], innovations, axis=0)
        delta_factors = np.diff(factors, prepend=0.0, axis=0)
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
        interp = self._interp_rows(hist.to_numpy(), available, self.tenors)

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
        vals = self._interp_rows(frame.to_numpy(), self.tenors, mats)
        return pd.DataFrame(vals, index=frame.index, columns=mats)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _interp_rows(values: np.ndarray, src_tenors: np.ndarray, new_tenors: np.ndarray) -> np.ndarray:
        """Interpolación lineal de cada fila de `values` a `new_tenors`, con extrapolación plana."""
        clipped = np.clip(new_tenors, src_tenors.min(), src_tenors.max())
        return interpolate.interp1d(src_tenors, values, axis=1)(clipped)

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
        prev = lambda x: np.r_[x[0], x[:-1]]

        for spec in node_specs:
            # --- Cupón par, duración y convexidad rezagados un día ---
            tenor = float(spec.metadata["plazo_anios"])
            par = self._par_yield_history(shocks, tenor)
            dmod, conv = self._par_bond_risk(par, tenor)
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
        times = BondAnalytics.coupon_times(tenor, self.coupon_freq)
        return BondAnalytics.par_coupon(shocks.levels_at(times).to_numpy(), times, self.coupon_freq)

    # ------------------------------------------------------------------ #
    def _par_bond_risk(self, par: np.ndarray, tenor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Duración modificada y convexidad de un bono par (cupón = tasa) por fecha."""
        times = BondAnalytics.coupon_times(tenor, self.coupon_freq)
        cfs = np.repeat(par[:, None] / self.coupon_freq, len(times), axis=1)
        cfs[:, -1] += 1.0
        _, dmod, conv = BondAnalytics.duration_convexity(times, cfs, par)
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

    @classmethod
    def from_config(
        cls, entry: Mapping[str, object], as_of: Union[str, datetime]
    ) -> "BondSpec":
        """
        Construye un bono desde un diccionario de `BONOS_UNIVERSO` (campos en
        español, ver la sección 6 de la configuración) y valida lo que el
        usuario escribe a mano para fallar con un mensaje claro y no más
        adelante con un error críptico.
        """
        required = ("isin", "emisor", "rating", "cupon", "vencimiento")
        missing = [k for k in required if entry.get(k) in (None, "")]
        label = entry.get("isin") or entry.get("emisor") or "?"
        if missing:
            raise ValueError(f"Bono '{label}' en BONOS_UNIVERSO: faltan los campos {missing}.")
        quotes = [k for k in ("ytm", "precio", "spread_bp") if entry.get(k) is not None]
        if not quotes:
            raise ValueError(
                f"Bono '{label}' en BONOS_UNIVERSO: falta el precio de mercado; "
                "ingresa 'ytm', 'precio' o 'spread_bp'."
            )

        rating = str(entry["rating"]).upper().strip()
        if rating not in RATING_SCALE:
            raise ValueError(
                f"Bono '{label}': rating '{entry['rating']}' desconocido; "
                f"usa uno de {list(RATING_SCALE)}."
            )
        maturity = pd.Timestamp(entry["vencimiento"])
        if maturity <= pd.Timestamp(as_of):
            raise ValueError(
                f"Bono '{label}': vence el {maturity.date()}, que no es posterior a la fecha de "
                f"los datos ({pd.Timestamp(as_of).date()})."
            )
        coupon = float(entry["cupon"])
        if not 0.0 <= coupon < 1.0:
            raise ValueError(
                f"Bono '{label}': 'cupon' debe ir en decimal (7,25% = 0.0725), no {coupon}."
            )
        if entry.get("ytm") is not None and not -0.05 < float(entry["ytm"]) < 1.0:
            raise ValueError(f"Bono '{label}': 'ytm' debe ir en decimal (10,5% = 0.105).")
        freq = int(entry.get("frecuencia", 1))
        if freq not in (1, 2, 4, 12):
            raise ValueError(f"Bono '{label}': 'frecuencia' debe ser 1, 2, 4 o 12, no {freq}.")
        liquidez = float(entry.get("liquidez", 0.5))
        if not 0.0 <= liquidez <= 1.0:
            raise ValueError(f"Bono '{label}': 'liquidez' debe estar entre 0 y 1.")

        return cls(
            isin=str(entry["isin"]).strip(),
            emisor=str(entry["emisor"]).strip(),
            rating=rating,
            coupon_rate=coupon,
            maturity_date=maturity.strftime("%Y-%m-%d"),
            ytm=float(entry["ytm"]) if entry.get("ytm") is not None else None,
            price=float(entry["precio"]) if entry.get("precio") is not None else None,
            spread_bp=float(entry["spread_bp"]) if entry.get("spread_bp") is not None else None,
            freq=freq,
            indexacion=str(entry.get("indexacion", "TF")).upper().strip(),
            liquidez=liquidez,
            sector=str(entry.get("sector", "Corporativo")),
        )

    @staticmethod
    def list_from_config(
        entries: Sequence[Mapping[str, object]], as_of: Union[str, datetime]
    ) -> List["BondSpec"]:
        """Convierte `BONOS_UNIVERSO` en bonos validados; rechaza ISIN repetidos."""
        bonds = [BondSpec.from_config(e, as_of) for e in entries]
        repeated = sorted({b.isin for b in bonds if sum(x.isin == b.isin for x in bonds) > 1})
        if repeated:
            raise ValueError(f"BONOS_UNIVERSO tiene identificadores repetidos: {repeated}.")
        return bonds


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
        if self.criteria.max_por_emisor <= 0 or candidates.empty:
            return pd.Index([])
        score = candidates["ytm"] * candidates["liquidez"]
        rank = score.groupby(candidates["emisor"]).rank(method="first", ascending=False)
        return candidates.index[rank > self.criteria.max_por_emisor]

    # ------------------------------------------------------------------ #
    # SCREENING
    # ------------------------------------------------------------------ #
    def screen(self) -> pd.DataFrame:
        """
        Aplica los filtros y construye el reporte de screening. Retorna un
        DataFrame con la analítica de cada título y su veredicto.
        """
        # --- Sin bonos en BONOS_UNIVERSO: reporte vacío con las columnas esperadas ---
        if not self.bonds:
            self.report_ = pd.DataFrame(columns=[
                "emisor", "sector", "rating", "indexacion", "cupon", "liquidez", "plazo_anios",
                "ytm", "precio", "duracion_mod", "convexidad", "spread_bp", "aprobado",
                "motivo_rechazo",
            ]).astype({"aprobado": bool}).rename_axis("isin")
            logger.info("BONOS_UNIVERSO está vacío: el Enfoque C (bonos individuales) no aporta activos.")
            return self.report_

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
        dropped = self._cap_per_issuer(passed)
        return [tk for tk in passed.index if tk not in dropped]

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
        spread_vol_bp: Optional[float] = None,
        seed: int = 77,
    ) -> pd.DataFrame:
        """
        Retornos simples diarios de los bonos aprobados por revaluación
        completa: cada día se descuentan los flujos que le quedan al título a
        su YTM vigente,

            YTM_t = z_t(plazo_t) + spread_t,     plazo_t = plazo_0 − τ_t

        donde z_t es la curva de ese día y el spread sigue una caminata
        aleatoria escalada por la (i)liquidez del título, anclada para que en
        la fecha de cotización (`as_of`) valga exactamente el spread que el
        usuario ingresó; hacia atrás en el tiempo es una reconstrucción
        simulada, ya que no hay precios históricos por bono. El retorno total
        incluye el cupón cobrado en el día:

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
        vol_bp = BONOS_VOL_SPREAD_BP_DIARIA if spread_vol_bp is None else spread_vol_bp
        levels = shocks.levels_
        tenors = np.asarray(levels.columns, dtype=float)
        level_rows = levels.to_numpy()
        tau = self._years_since_as_of(levels.index)
        d_tau = np.diff(tau, prepend=tau[0])
        reinvest = shocks.levels_at([self.reinvest_tenor]).iloc[:, 0].to_numpy()
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
            spread_shock = rng.normal(0.0, (vol_bp / 1e4) * (1.5 - liq), size=len(levels))
            walk = np.cumsum(spread_shock)
            spread = float(spec.metadata["spread_bp"]) / 1e4 + (walk - walk[-1])

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
        if report.empty:
            fig.update_layout(title="Screening de bonos individuales — BONOS_UNIVERSO está vacío")
            return fig
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
        # `as_of` es la fecha de las cotizaciones de los bonos (el cierre de los datos)
        # y `curve` debe ser la curva TES de esa misma fecha.
        # --- Enfoque A: constructor de nodos TES ---
        self.curve = curve
        self.node_builder = TESNodeBuilder(curve, node_tenors, include_rolldown=include_rolldown)
        self.include_rolldown = include_rolldown

        # --- Generador de choques de curva ---
        tenors = sorted(set([float(t) for t in node_tenors]) | set(curve.tenors.tolist()))
        self.shocks = CurveShockGenerator(curve, tenors, shock_config)

        # --- Enfoque C: screener de bonos individuales ---
        self.screener = BondScreener(list(bonds or []), criteria, curve, as_of)
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
        return float(np.expm1(returns.mean() * RiskMetrics.TRADING_DAYS)) if len(returns) else 0.0

    @staticmethod
    def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Beta sectorial/de mercado: Cov(asset, market) / Var(market)."""
        aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if aligned.shape[0] < 2:
            return float("nan")
        return float(aligned.cov().iloc[0, 1] / aligned.iloc[:, 1].var())

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


CRITERIOS_OPTIMIZACION: Dict[str, str] = {
    "max_sharpe": "Máximo Sharpe",
    "min_varianza": "Mínima varianza",
    "max_retorno_vol": "Máximo retorno con volatilidad objetivo",
    "min_cvar": "Mínimo CVaR",
    "max_retorno_cvar": "Máximo retorno con CVaR objetivo",
}


@dataclass
class ObjectiveSpec:
    """
    Qué optimizar y con qué objetivos de riesgo. Los defaults provienen de la
    sección 8 de la configuración. Un objetivo en None se reemplaza por el punto
    medio del rango alcanzable en el momento de optimizar.
    """

    criterion: str = field(default_factory=lambda: CRITERIO_OPTIMIZACION)
    vol_target: Optional[float] = field(default_factory=lambda: VOLATILIDAD_OBJETIVO_ANUAL)
    cvar_target: Optional[float] = field(default_factory=lambda: CVAR_OBJETIVO_MENSUAL)
    cvar_level: float = field(default_factory=lambda: CVAR_NIVEL)
    horizon_days: int = field(default_factory=lambda: HORIZONTE_DIAS)
    return_freq: str = field(default_factory=lambda: FRECUENCIA_RETORNOS_CVAR)
    center_on_mu: bool = field(default_factory=lambda: CVAR_CENTRAR_EN_MU)
    n_scenarios: int = field(default_factory=lambda: CVAR_ESCENARIOS)
    block_days: int = field(default_factory=lambda: CVAR_BLOQUE_DIAS)
    seed: int = field(default_factory=lambda: SEMILLA_ALEATORIA)
    # --- Número de activos ---
    max_assets: Optional[int] = field(default_factory=lambda: MAX_ACTIVOS)
    min_position: Optional[float] = field(default_factory=lambda: PESO_MIN_POSICION)
    auto_assets: bool = field(default_factory=lambda: CARDINALIDAD_AUTOMATICA)
    auto_tolerance: float = field(default_factory=lambda: CARDINALIDAD_TOLERANCIA)
    min_assets: Optional[int] = field(default_factory=lambda: MIN_ACTIVOS)

    def __post_init__(self) -> None:
        if self.max_assets is not None and self.max_assets < 1:
            raise ValueError("MAX_ACTIVOS debe ser al menos 1 o None.")
        if self.min_assets is not None and self.min_assets < 1:
            raise ValueError("MIN_ACTIVOS debe ser al menos 1 o None.")
        if self.min_assets and self.max_assets and self.min_assets > self.max_assets:
            raise ValueError(f"MIN_ACTIVOS ({self.min_assets}) no puede superar MAX_ACTIVOS ({self.max_assets}).")
        if self.min_position is not None and not 0.0 < self.min_position < 1.0:
            raise ValueError("PESO_MIN_POSICION debe estar en (0, 1) o ser None.")
        if self.n_scenarios < 200:
            raise ValueError("CVAR_ESCENARIOS debe ser al menos 200.")
        if self.criterion not in CRITERIOS_OPTIMIZACION:
            raise ValueError(f"CRITERIO_OPTIMIZACION debe ser uno de {list(CRITERIOS_OPTIMIZACION)}, "
                             f"no '{self.criterion}'.")
        if not 0.5 <= self.cvar_level < 1.0:
            raise ValueError(f"CVAR_NIVEL debe estar en [0.5, 1), no {self.cvar_level}.")
        if self.return_freq not in ("diaria", "semanal"):
            raise ValueError(f"FRECUENCIA_RETORNOS_CVAR debe ser 'diaria' o 'semanal', no '{self.return_freq}'.")
        if self.horizon_days < 1:
            raise ValueError("HORIZONTE_DIAS debe ser al menos 1.")

    @property
    def label(self) -> str:
        return CRITERIOS_OPTIMIZACION[self.criterion]

    @property
    def limits_assets(self) -> bool:
        return (self.max_assets is not None or self.min_position is not None
                or self.min_assets is not None)

    @property
    def needs_vol(self) -> bool:
        return self.criterion == "max_retorno_vol"

    @property
    def needs_cvar(self) -> bool:
        return self.criterion == "max_retorno_cvar"


class PortfolioOptimizer:
    """
    Optimización de portafolios por Máximo Sharpe (Markowitz clásico) bajo
    restricción de no-cortaje (long-only, w_i >= 0), suma de pesos = 1,
    límite máximo por activo y **bandas por clase de activo**:

        Σ w = 1
        0 ≤ w_i ≤ w_max,i
        min_g ≤ Σ_{i ∈ g} w_i ≤ max_g       (p. ej. 40% ≤ Σ w_RV ≤ 60%)

    El Sharpe no es cóncavo en w, pero con el cambio de variable y = w/(μ−rf)'w
    el problema se vuelve un QP convexo (ver `max_sharpe`), que se resuelve
    con `scipy.optimize.minimize` (SLSQP) hasta el óptimo global.

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
        tail_returns: Optional[pd.DataFrame] = None,
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
        # Historia más larga para la cola del CVaR (si no se da, la misma ventana de estimación)
        self.tail_returns = (tail_returns[self.assets].dropna(how="any")
                             if tail_returns is not None else self.returns)
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
        self._band_matrix, self._band_lo, self._band_hi = self._build_band_system()

        # --- Momentos: covarianza ---
        self.cov_, self.shrinkage_intensity_ = MarketDataPipeline.covariance_matrix(
            self.returns, shrinkage=shrinkage, trading_days=trading_days
        )
        self.cov_df_ = pd.DataFrame(self.cov_, index=self.assets, columns=self.assets)

        # --- Momentos: retorno esperado (μ) ---
        override = (pd.Series(expected_returns, dtype=float).reindex(self.assets).dropna()
                    if expected_returns is not None else pd.Series(dtype=float))
        self.mu_sample_series_ = (self.returns.mean() * trading_days).rename("mu_muestral")
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
    def _build_band_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Matriz de pertenencia (bandas x activos) y límites inferior/superior de cada banda."""
        active = [gc for gc in self.group_constraints if len(self._group_index[gc.label])]
        matrix = np.zeros((len(active), self.n))
        for row, gc in enumerate(active):
            matrix[row, self._group_index[gc.label]] = 1.0
        return (matrix,
                np.array([gc.min_weight for gc in active], dtype=float),
                np.array([gc.max_weight for gc in active], dtype=float))

    # ------------------------------------------------------------------ #
    # PUNTOS INICIALES FACTIBLES
    # ------------------------------------------------------------------ #
    def _initial_weights(self) -> np.ndarray:
        """
        Portafolio factible más cercano al equiponderado (proyección euclídea
        sobre presupuesto, topes y bandas). Sirve de punto inicial de la mínima
        varianza y de respaldo final cuando el solver no converge.
        """
        equal = np.full(self.n, 1.0 / self.n)
        w = self._minimize_quadratic(np.eye(self.n), equal, start=equal)
        if w is None:
            w = np.minimum(equal, self.max_weights)
            w = w / w.sum()
        return w

    # ------------------------------------------------------------------ #
    def _minimize_quadratic(
        self, quad: np.ndarray, center: np.ndarray, start: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Resuelve min (w−center)'·quad·(w−center) sobre el conjunto factible
        (presupuesto, topes y bandas): QP convexo. Devuelve None si SLSQP no
        converge a un punto factible.
        """
        constraints = [optimize.LinearConstraint(np.ones((1, self.n)), 1.0, 1.0)]
        if len(self._band_matrix):
            constraints.append(optimize.LinearConstraint(self._band_matrix, self._band_lo, self._band_hi))
        result = optimize.minimize(
            lambda w: (w - center) @ quad @ (w - center),
            start,
            jac=lambda w: 2.0 * quad @ (w - center),
            method="SLSQP",
            bounds=optimize.Bounds(self.min_weight, self.max_weights),
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w = np.clip(result.x, 0.0, None)
        if result.success and w.sum() > 0 and self.is_feasible(w / w.sum()):
            return w / w.sum()
        return None

    # ------------------------------------------------------------------ #
    # a) MARKOWITZ / MÁXIMO SHARPE
    # ------------------------------------------------------------------ #
    def _portfolio_perf(self, w: np.ndarray) -> Tuple[float, float]:
        ret = float(w @ self.mu_)
        vol = float(np.sqrt(w @ self.cov_ @ w))
        return ret, vol

    def is_feasible(self, w: np.ndarray, tol: float = 1e-6) -> bool:
        """Verifica presupuesto, topes individuales y bandas por clase."""
        if abs(float(np.sum(w)) - 1.0) > 1e-6:
            return False
        if np.any(w < -tol) or np.any(w > self.max_weights + tol):
            return False
        exposure = self._band_matrix @ w
        return bool(np.all(exposure >= self._band_lo - tol) and np.all(exposure <= self._band_hi + tol))

    def max_sharpe(self) -> pd.Series:
        """
        Portafolio de máximo Sharpe (tangencia) long-only, con tope por activo y
        bandas por clase de activo. La tasa libre de riesgo (rf) debe derivarse
        externamente (ej. de la curva TES o la IBR overnight).

        El Sharpe no es cóncavo en w, pero con u = c·w/(μ−rf)'w (c > 0
        arbitrario) el problema es un QP convexo (Cornuejols y Tütüncü, 2006):

            min  u'Σu
            s.t. (μ−rf)'u = c,  u ≥ 0
                 min·Σu ≤ u_i ≤ max_i·Σu
                 lo_g·Σu ≤ Σ_{i ∈ g} u_i ≤ hi_g·Σu

        y w = u/Σu. Al ser convexo, SLSQP llega al óptimo global desde un solo
        punto inicial: el portafolio factible de mayor exceso de retorno (LP).
        Si ningún portafolio factible supera a rf no existe tangencia (el
        cociente sería negativo para todos) y se devuelve el de mínima varianza.
        """
        excess = self.mu_ - self.rf_log_
        has_bands = len(self._band_matrix) > 0

        # --- Punto inicial: portafolio factible de máximo exceso de retorno (LP) ---
        lp = optimize.linprog(
            -excess,
            A_ub=np.vstack([self._band_matrix, -self._band_matrix]) if has_bands else None,
            b_ub=np.concatenate([self._band_hi, -self._band_lo]) if has_bands else None,
            A_eq=np.ones((1, self.n)), b_eq=[1.0],
            bounds=np.column_stack([np.full(self.n, self.min_weight), self.max_weights]),
        )
        max_excess = -lp.fun if lp.success else 0.0

        # --- QP convexo en u: topes, mínimos y bandas escalados por Σu ---
        best_w: Optional[np.ndarray] = None
        if max_excess <= 1e-12:
            logger.warning("Max Sharpe: ningún portafolio factible supera la tasa libre de riesgo; "
                           "se usa el de mínima varianza.")
            best_w = self._min_variance_weights()
        else:
            ones, eye = np.ones((1, self.n)), np.eye(self.n)
            homogeneous = np.vstack([
                eye - self.min_weight * ones,
                self.max_weights[:, None] * ones - eye,
                self._band_matrix - self._band_lo[:, None] * ones,
                self._band_hi[:, None] * ones - self._band_matrix,
            ])
            result = optimize.minimize(
                lambda u: u @ self.cov_ @ u,
                lp.x,
                jac=lambda u: 2.0 * self.cov_ @ u,
                method="SLSQP",
                bounds=optimize.Bounds(0.0, np.inf),
                constraints=[
                    optimize.LinearConstraint(excess[None, :], max_excess, max_excess),
                    optimize.LinearConstraint(homogeneous, 0.0, np.inf),
                ],
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            self.result_ = result
            u = np.clip(result.x, 0.0, None)
            if result.success and u.sum() > 0 and self.is_feasible(u / u.sum()):
                best_w = u / u.sum()

        # --- Respaldo: punto medio factible de las bandas ---
        if best_w is None:
            logger.warning("Max Sharpe: SLSQP no convergió. Usando el punto medio factible de las bandas.")
            w0 = self._initial_weights()
            best_w = w0 / w0.sum()

        return pd.Series(best_w, index=self.assets, name="max_sharpe")

    # ------------------------------------------------------------------ #
    def _min_variance_weights(self) -> Optional[np.ndarray]:
        """Portafolio de mínima varianza dentro de topes y bandas (QP convexo); None si falla."""
        return self._minimize_quadratic(self.cov_, np.zeros(self.n), start=self._initial_weights())

    # ------------------------------------------------------------------ #
    # ESCENARIOS Y CVaR (BOOTSTRAP POR BLOQUES SOBRE LA HISTORIA DE LA COLA)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bootstrap_paths(
        X: np.ndarray, steps: int, mean_block: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Bootstrap estacionario (Politis-Romano): cada trayectoria encadena
        `steps` retornos de las filas de X; en cada paso sigue con la fila
        siguiente (continúa el bloque) o, con probabilidad 1/mean_block,
        salta a una fila al azar. Devuelve la suma de cada trayectoria
        (n_paths x activos). Conserva correlaciones entre activos y rachas.
        """
        T = X.shape[0]
        idx = np.empty((n_paths, steps), dtype=np.int64)
        idx[:, 0] = rng.integers(0, T, n_paths)
        for t in range(1, steps):
            jump = rng.random(n_paths) < 1.0 / max(mean_block, 1)
            idx[:, t] = np.where(jump, rng.integers(0, T, n_paths), (idx[:, t - 1] + 1) % T)
        return X[idx].sum(axis=1)

    # ------------------------------------------------------------------ #
    def _tail_matrix(self, spec: ObjectiveSpec) -> Tuple[np.ndarray, int, int]:
        """Retornos de la ventana de la cola en la frecuencia pedida: (matriz, periodos/año, bloque)."""
        r = self.tail_returns
        periods = self.trading_days
        if spec.return_freq == "semanal":
            r = r.resample("W-FRI").sum()
            periods = 52
        X = r.to_numpy()
        if spec.center_on_mu:
            X = X - X.mean(axis=0) + self.mu_ / periods
        block = max(int(round(spec.block_days * periods / self.trading_days)), 1)
        return X, periods, block

    # ------------------------------------------------------------------ #
    def scenarios(self, spec: ObjectiveSpec) -> np.ndarray:
        """
        Matriz (escenarios x activos) de retornos simples al horizonte del
        portafolio, simulada por bootstrap por bloques de los retornos
        continuos de la ventana de la cola (diarios o semanales), recentrados
        en μ si `center_on_mu`. Es determinista (semilla fija).
        """
        key = (spec.return_freq, spec.horizon_days, spec.center_on_mu, spec.n_scenarios, spec.block_days, spec.seed)
        if getattr(self, "_scenario_key", None) == key:
            return self._scenario_cache
        X, periods, block = self._tail_matrix(spec)
        steps = max(int(round(spec.horizon_days * periods / self.trading_days)), 1)
        paths = self._bootstrap_paths(X, steps, block, spec.n_scenarios, np.random.default_rng(spec.seed))
        self._scenario_key, self._scenario_cache = key, np.expm1(paths)
        return self._scenario_cache

    # ------------------------------------------------------------------ #
    def tail_diagnostics(self, spec: ObjectiveSpec) -> Dict[str, float]:
        """Cuánta información respalda la cola: observaciones, años y observaciones en la cola."""
        periods = 52 if spec.return_freq == "semanal" else self.trading_days
        n_obs = len(self.tail_returns) if spec.return_freq == "diaria" else len(
            self.tail_returns.resample("W-FRI").sum())
        tail = n_obs * (1.0 - spec.cvar_level)
        return {
            "observaciones": float(n_obs), "anios": n_obs / periods, "obs_en_la_cola": tail,
            "anios_minimos": (CVAR_MIN_OBS_COLA / (1.0 - spec.cvar_level)) / periods,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cvar_from_losses(losses: np.ndarray, level: float) -> Tuple[float, float]:
        var = float(np.quantile(losses, level))
        return float(losses[losses >= var].mean()), var

    def cvar_var(self, w: np.ndarray, spec: ObjectiveSpec) -> Tuple[float, float]:
        """(CVaR, VaR) de pérdida al horizonte, como fracción positiva del portafolio."""
        return self._cvar_from_losses(-(self.scenarios(spec) @ np.asarray(w, dtype=float)), spec.cvar_level)

    # ------------------------------------------------------------------ #
    def cvar_interval(
        self, w: np.ndarray, spec: ObjectiveSpec, n_boot: Optional[int] = None, confidence: float = 0.90
    ) -> Tuple[float, float]:
        """
        Intervalo de confianza del CVaR por remuestreo: se remuestrea la propia
        historia de la cola (bootstrap por bloques), se vuelve a simular el
        CVaR con cada remuestreo y se toman los percentiles. Un intervalo ancho
        indica que la historia no basta para fijar la cola.
        """
        n_boot = CVAR_REMUESTREOS_IC if n_boot is None else n_boot
        w = np.asarray(w, dtype=float)
        X, periods, block = self._tail_matrix(spec)
        steps = max(int(round(spec.horizon_days * periods / self.trading_days)), 1)
        rng = np.random.default_rng(spec.seed + 1)
        values = []
        for _ in range(n_boot):
            rows = self._bootstrap_rows(X.shape[0], block, rng)
            paths = self._bootstrap_paths(X[rows], steps, block, 1000, rng)
            values.append(self._cvar_from_losses(-(np.expm1(paths) @ w), spec.cvar_level)[0])
        lo, hi = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2])
        return float(lo), float(hi)

    @staticmethod
    def _bootstrap_rows(T: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
        """Índices de una historia remuestreada (bootstrap estacionario) del mismo largo T."""
        idx = np.empty(T, dtype=np.int64)
        idx[0] = rng.integers(0, T)
        jumps = rng.random(T) < 1.0 / max(mean_block, 1)
        starts = rng.integers(0, T, T)
        for t in range(1, T):
            idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % T
        return idx

    # ------------------------------------------------------------------ #
    def _lp_constraints(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Bandas por clase en forma A_ub·w ≤ b_ub (None si no hay bandas)."""
        if not len(self._band_matrix):
            return None, None
        return (np.vstack([self._band_matrix, -self._band_matrix]),
                np.concatenate([self._band_hi, -self._band_lo]))

    # ------------------------------------------------------------------ #
    def _cvar_lp(
        self, spec: ObjectiveSpec, maximize_return: bool, cvar_cap: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Programa lineal de Rockafellar-Uryasev sobre los escenarios simulados.
        Variables x = [w (n), α (1), z (S)] con CVaR = α + Σz / ((1−β)·S):

            min  CVaR                       (maximize_return = False)
            max  μ'w  s.t.  CVaR ≤ cvar_cap (maximize_return = True)
            s.t. z_s ≥ −R_s·w − α,  z ≥ 0,  Σw = 1, topes y bandas.

        Matrices dispersas: con miles de escenarios el bloque de pérdidas es casi vacío.
        """
        R = self.scenarios(spec)
        S, n = R.shape
        tail = 1.0 / ((1.0 - spec.cvar_level) * S)
        cvar_row = np.concatenate([np.zeros(n), [1.0], np.full(S, tail)])

        blocks = [sparse.hstack([sparse.csr_matrix(-R), sparse.csr_matrix(-np.ones((S, 1))),
                                 -sparse.identity(S, format="csr")], format="csr")]
        rhs = [np.zeros(S)]
        band_a, band_b = self._lp_constraints()
        if band_a is not None:
            blocks.append(sparse.hstack([sparse.csr_matrix(band_a), sparse.csr_matrix((len(band_a), 1 + S))],
                                        format="csr"))
            rhs.append(band_b)
        if maximize_return:
            blocks.append(sparse.csr_matrix(cvar_row[None, :]))
            rhs.append(np.array([cvar_cap]))
            cost = np.concatenate([-self.mu_, np.zeros(1 + S)])
        else:
            cost = cvar_row

        res = optimize.linprog(
            cost, A_ub=sparse.vstack(blocks, format="csr"), b_ub=np.concatenate(rhs),
            A_eq=sparse.csr_matrix(np.concatenate([np.ones(n), np.zeros(1 + S)])[None, :]), b_eq=[1.0],
            bounds=[(self.min_weight, float(m)) for m in self.max_weights] + [(None, None)] + [(0, None)] * S,
            method="highs",
        )
        if not res.success:
            return None
        w = np.clip(res.x[:n], 0.0, None)
        return w / w.sum() if w.sum() > 0 else None

    # ------------------------------------------------------------------ #
    def _max_return_weights(self) -> np.ndarray:
        """Portafolio factible de mayor retorno esperado (LP): el extremo de mayor riesgo."""
        band_a, band_b = self._lp_constraints()
        lp = optimize.linprog(
            -self.mu_, A_ub=band_a, b_ub=band_b,
            A_eq=np.ones((1, self.n)), b_eq=[1.0],
            bounds=np.column_stack([np.full(self.n, self.min_weight), self.max_weights]),
        )
        if lp.success:
            return np.clip(lp.x, 0.0, None) / np.clip(lp.x, 0.0, None).sum()
        return self._initial_weights()

    # ------------------------------------------------------------------ #
    def _max_return_vol_weights(self, vol_cap: float) -> Optional[np.ndarray]:
        """Máximo retorno con volatilidad ≤ vol_cap: QP convexo, punto inicial = mínima varianza."""
        start = self._min_variance_weights()
        if start is None:
            return None
        constraints = [
            optimize.LinearConstraint(np.ones((1, self.n)), 1.0, 1.0),
            optimize.NonlinearConstraint(lambda w: w @ self.cov_ @ w, 0.0, vol_cap ** 2,
                                         jac=lambda w: 2.0 * self.cov_ @ w),
        ]
        if len(self._band_matrix):
            constraints.append(optimize.LinearConstraint(self._band_matrix, self._band_lo, self._band_hi))
        res = optimize.minimize(
            lambda w: -(w @ self.mu_), start, jac=lambda w: -self.mu_, method="SLSQP",
            bounds=optimize.Bounds(self.min_weight, self.max_weights), constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w = np.clip(res.x, 0.0, None)
        if res.success and w.sum() > 0:
            w = w / w.sum()
            if self.is_feasible(w) and np.sqrt(w @ self.cov_ @ w) <= vol_cap * (1 + 1e-4):
                return w
        return start

    # ------------------------------------------------------------------ #
    # MEDIDAS DE UN PORTAFOLIO Y RANGO ALCANZABLE
    # ------------------------------------------------------------------ #
    def measures(self, w: Union[pd.Series, np.ndarray], spec: ObjectiveSpec) -> Dict[str, float]:
        """Retorno, volatilidad, Sharpe, CVaR/VaR al horizonte y concentración de un portafolio."""
        w = np.asarray(w, dtype=float)
        ret, vol = self._portfolio_perf(w)
        cvar, var = self.cvar_var(w, spec)
        hhi = float((w ** 2).sum())
        return {
            "retorno_esperado_anual": float(np.expm1(ret)),
            "volatilidad_anual": vol,
            "sharpe": (ret - self.rf_log_) / vol if vol > 0 else 0.0,
            "cvar_horizonte": cvar,
            "var_horizonte": var,
            "num_activos": float((w > 1e-4).sum()),
            "activos_efectivos": 1.0 / hhi if hhi > 0 else 0.0,
            "mayor_peso": float(w.max()),
        }

    # ------------------------------------------------------------------ #
    def reference_portfolios(self, spec: ObjectiveSpec) -> Dict[str, np.ndarray]:
        """Portafolios de referencia que delimitan lo alcanzable."""
        w_minvar = self._min_variance_weights()
        refs = {
            "Mínima varianza": w_minvar if w_minvar is not None else self._initial_weights(),
            "Mínimo CVaR": self._cvar_lp(spec, maximize_return=False),
            "Máximo Sharpe": self.max_sharpe().to_numpy(),
            "Máximo retorno": self._max_return_weights(),
        }
        return {k: v for k, v in refs.items() if v is not None}

    # ------------------------------------------------------------------ #
    def achievable_ranges(self, spec: ObjectiveSpec) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """
        Tabla de los portafolios de referencia con sus medidas y rangos
        alcanzables de volatilidad anual y CVaR al horizonte: del mínimo (el
        portafolio que lo minimiza) al valor del portafolio de máximo retorno.
        """
        with self.diversified(spec):
            refs = self.reference_portfolios(spec)
        self.reference_weights_ = refs
        table = pd.DataFrame({name: self.measures(w, spec) for name, w in refs.items()}).T
        ranges = {
            "vol": (float(table.loc["Mínima varianza", "volatilidad_anual"]),
                    float(table.loc["Máximo retorno", "volatilidad_anual"])),
            "cvar": (float(table.loc["Mínimo CVaR", "cvar_horizonte"]),
                     float(table.loc["Máximo retorno", "cvar_horizonte"])),
        }
        return table, ranges

    # ------------------------------------------------------------------ #
    def _range_for(self, key: str, spec: ObjectiveSpec) -> Tuple[float, float]:
        """Rango alcanzable de 'vol' o 'cvar' (sólo lo necesario, más barato que `achievable_ranges`)."""
        w_max = self._max_return_weights()
        if key == "vol":
            w_min = self._min_variance_weights()
            w_min = w_min if w_min is not None else self._initial_weights()
            return self._portfolio_perf(w_min)[1], self._portfolio_perf(w_max)[1]
        w_min = self._cvar_lp(spec, maximize_return=False)
        w_min = w_min if w_min is not None else self._initial_weights()
        return self.cvar_var(w_min, spec)[0], self.cvar_var(w_max, spec)[0]

    # ------------------------------------------------------------------ #
    def optimize(
        self, spec: ObjectiveSpec, ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[pd.Series, Dict[str, object]]:
        """
        Resuelve el criterio pedido sobre el universo vigente (sin límite de
        número de activos; para eso está `solve`). Si un objetivo cae fuera del
        rango alcanzable se usa el valor factible más cercano (y se informa); si
        no se dio objetivo, el punto medio del rango. Retorna (pesos, info).
        """
        info: Dict[str, object] = {"criterio": spec.label, "nota": "", "resuelto": True}
        w: Optional[np.ndarray] = None

        if spec.criterion == "max_sharpe":
            w = self.max_sharpe().to_numpy()
        elif spec.criterion == "min_varianza":
            w = self._min_variance_weights()
        elif spec.criterion == "min_cvar":
            w = self._cvar_lp(spec, maximize_return=False)
        else:
            key, target = ("vol", spec.vol_target) if spec.needs_vol else ("cvar", spec.cvar_target)
            lo, hi = ranges[key] if ranges is not None else self._range_for(key, spec)
            used = (lo + hi) / 2.0 if target is None else float(np.clip(target, lo, hi))
            unit = "volatilidad anual" if key == "vol" else "CVaR"
            if target is None:
                info["nota"] = f"Sin {unit} objetivo: se usó el punto medio del rango alcanzable ({used:.2%})."
            elif abs(used - target) > 1e-9:
                info["nota"] = (f"El objetivo de {unit} ({target:.2%}) está fuera del rango alcanzable "
                                f"[{lo:.2%}, {hi:.2%}]: se usó {used:.2%}.")
            info["objetivo_usado"] = used
            if key == "vol":
                w = self._max_return_vol_weights(used)
            else:
                w = self._cvar_lp(spec, maximize_return=True, cvar_cap=used * (1 + 1e-9))

        if w is None:
            logger.warning("%s: el solver no encontró solución; se usa la mínima varianza.", spec.label)
            info["nota"] = (info["nota"] + " El solver no convergió: se usó la mínima varianza.").strip()
            info["resuelto"] = False
            w = self._min_variance_weights()
            if w is None:
                w = self._initial_weights()
        w = np.clip(w, 0.0, None) / np.clip(w, 0.0, None).sum()
        return pd.Series(w, index=self.assets, name=spec.criterion), info

    # ------------------------------------------------------------------ #
    # NÚMERO DE ACTIVOS (HEURÍSTICA DE CARDINALIDAD)
    # ------------------------------------------------------------------ #
    def score(self, w: np.ndarray, spec: ObjectiveSpec) -> float:
        """Valor del objetivo del criterio (mayor es mejor)."""
        ret, vol = self._portfolio_perf(w)
        if spec.criterion == "max_sharpe":
            return (ret - self.rf_log_) / vol if vol > 0 else 0.0
        if spec.criterion == "min_varianza":
            return -vol
        if spec.criterion == "min_cvar":
            return -self.cvar_var(w, spec)[0]
        return ret

    # ------------------------------------------------------------------ #
    def solve(
        self, spec: ObjectiveSpec, ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[pd.Series, Dict[str, object]]:
        """`optimize` más los límites de número de activos y de peso mínimo, si están activos."""
        if not spec.limits_assets:
            return self.optimize(spec, ranges)
        with self.diversified(spec):
            return self._solve_cardinality(spec, ranges)

    # ------------------------------------------------------------------ #
    @contextmanager
    def diversified(self, spec: ObjectiveSpec):
        """
        Aplica el mínimo de activos como tope por activo de 1/N mientras dure el
        bloque (al menos N activos <=> ninguno pasa de 1/N) y luego restaura los topes.
        """
        original = self.max_weights.copy()
        try:
            if spec.min_assets:
                self.max_weights = np.minimum(self.max_weights, 1.0 / spec.min_assets)
                if self.max_weights.sum() < 1.0 - 1e-9:
                    raise ValueError(f"MIN_ACTIVOS = {spec.min_assets} no se puede cumplir con sólo "
                                     f"{self.n} activos invertibles.")
            yield
        finally:
            self.max_weights = original

    # ------------------------------------------------------------------ #
    def _solve_cardinality(
        self, spec: ObjectiveSpec, ranges: Optional[Dict[str, Tuple[float, float]]]
    ) -> Tuple[pd.Series, Dict[str, object]]:
        """
        Heurística de eliminación hacia atrás: se resuelve el óptimo libre y,
        mientras haya más activos que `max_assets` o posiciones menores a
        `min_position`, se elimina el de menor peso (varios a la vez si faltan
        muchos) y se re-optimiza sin él. En modo automático se sigue
        eliminando hasta `min_assets` para trazar cómo cambia el objetivo
        con el número de activos, y se elige el portafolio con menos activos
        cuyo objetivo está a menos de `auto_tolerance` del mejor.

        No garantiza el óptimo global (el problema es combinatorio), pero cada
        paso es un problema convexo resuelto con los mismos solvers.
        """
        eps = 1e-4
        caps = self.max_weights.copy()
        max_k = spec.max_assets or self.n
        w_min = spec.min_position or 0.0
        floor_k = max(spec.min_assets or 1, 1) if spec.auto_assets else 1
        path: Dict[int, Tuple[np.ndarray, bool]] = {}
        protected: Set[int] = set()
        notes: List[str] = []

        def record(w: np.ndarray) -> None:
            sel = w > eps
            k = int(sel.sum())
            path[k] = (w.copy(), k <= max_k and bool(np.all(w[sel] >= w_min - 1e-6)))

        try:
            w_series, info = self.optimize(spec, ranges)
            w = w_series.to_numpy()
            record(w)
            while True:
                sel = np.flatnonzero(w > eps)
                k = len(sel)
                too_small = bool(w_min) and bool(np.any(w[sel] < w_min - 1e-6))
                if k <= 1 or not (k > max_k or too_small or (spec.auto_assets and k > floor_k)):
                    break
                order = [i for i in sel[np.argsort(w[sel])] if i not in protected]
                if not order:
                    break
                excess = k - max_k
                dropped = order[: max(1, excess // 2) if excess > 4 else 1]
                saved = self.max_weights.copy()
                self.max_weights[dropped] = 0.0
                w_new, info_new = self.optimize(spec, None)
                w_new = w_new.to_numpy()
                if not info_new["resuelto"] or not self.is_feasible(w_new):
                    self.max_weights = saved          # sin ese activo no hay solución (p. ej. una banda): se protege
                    protected.update(dropped)
                    continue
                w = w_new
                record(w)
        finally:
            self.max_weights = caps

        valid = {k: w for k, (w, ok) in path.items() if ok}
        curve = pd.DataFrame({
            k: {"score": self.score(w, spec), **self.measures(w, spec)} for k, w in sorted(valid.items())
        }).T
        curve.index.name = "activos"
        if not valid:
            chosen_k = min(path)
            chosen = path[chosen_k][0]
            notes.append(f"No se logró cumplir los límites de activos ({max_k}) y peso mínimo "
                         f"({w_min:.0%}); se dejó el portafolio con {chosen_k} activos.")
        elif spec.auto_assets:
            best = float(curve["score"].max())
            ok_k = [k for k, sc in curve["score"].items() if sc >= best - spec.auto_tolerance * abs(best) - 1e-12]
            chosen_k = int(min(ok_k))
            chosen = valid[chosen_k]
            notes.append(f"Cantidad automática de activos: {chosen_k} (con {int(curve.index.max())} "
                         f"el objetivo mejora menos de {spec.auto_tolerance:.0%}).")
        else:
            chosen_k = int(max(valid))
            chosen = valid[chosen_k]

        info["nota"] = " ".join(x for x in [str(info["nota"])] + notes if x).strip()
        info["curva_activos"] = curve
        info["activos_elegidos"] = chosen_k
        return pd.Series(chosen, index=self.assets, name=spec.criterion), info

    # ------------------------------------------------------------------ #
    # DIAGNÓSTICO DE LA SOLUCIÓN
    # ------------------------------------------------------------------ #
    def risk_contribution(self, weights: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Contribución porcentual de cada activo a la varianza del portafolio."""
        w = np.asarray(weights, dtype=float)
        marginal = self.cov_ @ w
        total = float(w @ marginal)
        return pd.Series(w * marginal / total if total > 0 else 0.0, index=self.assets)

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
        return pd.DataFrame(rows).set_index("clase") if rows else pd.DataFrame()

    # ------------------------------------------------------------------ #
    # VISUALIZACIONES
    # ------------------------------------------------------------------ #
    def plot_efficient_frontier(
        self,
        chosen: pd.Series,
        chosen_label: str,
        references: Optional[Mapping[str, np.ndarray]] = None,
        n_points: Optional[int] = None,
    ) -> go.Figure:
        """
        Frontera eficiente (máximo retorno para cada nivel de volatilidad, con
        las restricciones vigentes), los activos individuales, los portafolios
        de referencia y el portafolio elegido.
        """
        n_points = PUNTOS_FRONTERA if n_points is None else n_points
        fig = go.Figure()

        # --- Activos individuales ---
        vols = np.sqrt(np.diag(self.cov_))
        fig.add_trace(go.Scatter(
            x=vols, y=np.expm1(self.mu_), mode="markers", name="Activos individuales",
            marker=dict(size=6, color="lightgray", line=dict(width=0.5, color="gray")),
            text=self.assets, hovertemplate="%{text}<br>vol %{x:.1%} | retorno %{y:.1%}<extra></extra>",
        ))

        # --- Frontera: máximo retorno para volatilidades crecientes ---
        w_minvar = self._min_variance_weights()
        w_maxret = self._max_return_weights()
        points: List[Tuple[float, float]] = []
        if w_minvar is not None:
            lo, hi = self._portfolio_perf(w_minvar)[1], self._portfolio_perf(w_maxret)[1]
            for cap in np.linspace(lo, hi, n_points):
                w = self._max_return_vol_weights(cap)
                if w is not None:
                    ret, vol = self._portfolio_perf(w)
                    points.append((vol, np.expm1(ret)))
            if points:
                xs, ys = zip(*sorted(points))
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Frontera eficiente",
                                         line=dict(width=2.5, color="#2E6F9E")))

        # --- Portafolios de referencia y elegido ---
        for name, w in (references or {}).items():
            ret, vol = self._portfolio_perf(np.asarray(w, dtype=float))
            fig.add_trace(go.Scatter(
                x=[vol], y=[np.expm1(ret)], mode="markers", name=name,
                marker=dict(size=10, symbol="diamond", line=dict(width=1, color="white")),
            ))
        ret, vol = self._portfolio_perf(chosen.to_numpy())
        fig.add_trace(go.Scatter(
            x=[vol], y=[np.expm1(ret)], mode="markers", name=f"Elegido: {chosen_label}",
            marker=dict(symbol="star", size=20, color="red", line=dict(width=1, color="white")),
        ))

        bands = " | ".join(
            f"{gc.label}: {gc.min_weight:.0%}–{gc.max_weight:.0%}" for gc in self.group_constraints
        )
        # Vista inicial centrada en la zona útil; el resto de la frontera se ve con zoom.
        near = [vol] + [self._portfolio_perf(np.asarray(w, dtype=float))[1]
                        for name, w in (references or {}).items() if name != "Máximo retorno"]
        x_max = 2.5 * max(near)
        y_seen = ([y for x, y in points if x <= x_max] + [np.expm1(m) for v, m in zip(vols, self.mu_) if v <= x_max]
                  + [np.expm1(ret)])
        pad = 0.08 * (max(y_seen) - min(y_seen) or 0.05)
        fig.update_layout(
            title="Frontera eficiente" + (f"<br><sup>Bandas: {bands}</sup>" if bands else ""),
            xaxis_title="Volatilidad anual (zoom para ver el resto)", yaxis_title="Retorno esperado anual (E.A.)",
            xaxis_tickformat=".0%", yaxis_tickformat=".0%",
            xaxis_range=[0, x_max], yaxis_range=[min(y_seen) - pad, max(y_seen) + pad],
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
        objective: Optional[ObjectiveSpec] = None,
        cvar_lookback_days: Optional[int] = None,
    ) -> None:
        # --- Datos y parámetros del backtest ---
        self.objective = objective or ObjectiveSpec()
        # Ventana larga (cola del CVaR); la de retorno y covarianza es `lookback_days`.
        self.cvar_lookback_days = max(cvar_lookback_days or 0, lookback_days)
        self.prices = prices.dropna(how="all")
        self.returns = MarketDataPipeline.compute_returns(self.prices)
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
        if len(idx) < self.cvar_lookback_days:
            return []
        dates = pd.Series(idx, index=idx).resample(self.rebalance_freq).last().dropna()
        return [pd.Timestamp(d) for d in dates[dates >= idx[self.cvar_lookback_days - 1]]]

    # ------------------------------------------------------------------ #
    def _in_sample(self, reb_date: pd.Timestamp) -> pd.DataFrame:
        """Ventana larga (cola del CVaR) con sólo los activos invertibles en `reb_date`."""
        window = self.returns.loc[:reb_date].tail(self.cvar_lookback_days)
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

        # --- Optimizador: retorno y covarianza con la ventana corta, cola con la larga ---
        return PortfolioOptimizer(
            window_returns.tail(self.lookback_days),
            tail_returns=window_returns,
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
        weights, _ = opt.solve(self.objective)
        return weights, opt.rf

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
        return pd.DataFrame(rows).set_index("clase") if rows else pd.DataFrame()

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
            title=f"Curva de Equity — Backtest Walk-Forward ({self.objective.label})",
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
    Restricciones del portafolio. Por defecto no hay ninguna más allá de "sin
    ventas en corto y 100% invertido": el optimizador elige libremente y la
    proporción RV/RF sale del resultado. Se pueden activar bandas por clase de
    activo y topes de peso (general o por tipo de instrumento).

    Los defaults provienen de la sección 8 del bloque de configuración.
    """

    bandas: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(BANDAS_POR_CLASE))
    max_peso_activo: Optional[float] = field(default_factory=lambda: MAX_PESO_POR_ACTIVO)
    topes_por_tipo: Dict[str, Optional[float]] = field(default_factory=lambda: dict(TOPES_POR_TIPO))

    rf_tenor_years: float = field(default_factory=lambda: RF_TENOR_YEARS)

    _TIPO_POR_SUBCLASE = {
        AssetSubClass.RV_ACCION_LOCAL: "accion",
        AssetSubClass.RV_ETF_LOCAL: "etf_rv",
        AssetSubClass.RV_ETF_GLOBAL: "etf_rv",
        AssetSubClass.RF_NODO_TES: "nodo_tes",
        AssetSubClass.RF_ETF: "etf_rf",
        AssetSubClass.RF_BONO: "bono",
    }

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        valid_classes = {c.value for c in AssetClass}
        unknown = set(self.bandas) - valid_classes
        if unknown:
            raise ValueError(f"BANDAS_POR_CLASE: clases desconocidas {sorted(unknown)}; "
                             f"usa {sorted(valid_classes)}.")
        unknown = set(self.topes_por_tipo) - set(self._TIPO_POR_SUBCLASE.values())
        if unknown:
            raise ValueError(f"TOPES_POR_TIPO: tipos desconocidos {sorted(unknown)}; "
                             f"usa {sorted(set(self._TIPO_POR_SUBCLASE.values()))}.")
        for label, cap in [("MAX_PESO_POR_ACTIVO", self.max_peso_activo), *self.topes_por_tipo.items()]:
            if cap is not None and not 0.0 < cap <= 1.0:
                raise ValueError(f"{label} debe estar en (0, 1] o ser None, no {cap}.")

    # ------------------------------------------------------------------ #
    def group_constraints(self) -> List[GroupConstraint]:
        return [GroupConstraint(cls, lo, hi) for cls, (lo, hi) in self.bandas.items()]

    # ------------------------------------------------------------------ #
    def _cap_for(self, spec: AssetSpec) -> float:
        cap = self.topes_por_tipo.get(self._TIPO_POR_SUBCLASE[spec.sub_class])
        if cap is None:
            cap = self.max_peso_activo
        return 1.0 if cap is None else float(cap)

    def max_weights(self, specs: Mapping[str, AssetSpec]) -> Dict[str, float]:
        """Tope individual de cada activo según su tipo de instrumento (1.0 = sin tope)."""
        return {tk: self._cap_for(spec) for tk, spec in specs.items()}

    # ------------------------------------------------------------------ #
    def validate(self, specs: Mapping[str, AssetSpec]) -> None:
        """
        Comprueba que cada banda sea alcanzable con los topes vigentes antes de
        llegar al solver, y avisa si el máximo supera la capacidad del universo.
        """
        # --- Capacidad de cada clase frente a su banda ---
        caps = self.max_weights(specs)
        for cls_label, (lo, hi) in self.bandas.items():
            miembros = [tk for tk, sp in specs.items() if sp.asset_class.value == cls_label]
            capacidad = sum(caps[tk] for tk in miembros)
            if capacidad < lo - 1e-9:
                raise ValueError(
                    f"Política infactible: la banda mínima de {cls_label} es {lo:.0%} pero sus "
                    f"{len(miembros)} activos sólo admiten {capacidad:.0%} con los topes vigentes. "
                    f"Suba los topes o amplíe el universo de {cls_label}."
                )
            if capacidad < hi:
                logger.warning(
                    "La banda máxima de %s (%.0f%%) supera la capacidad del universo (%.0f%%): "
                    "el tope efectivo será %.0f%%.", cls_label, hi * 100, capacidad * 100,
                    capacidad * 100,
                )

        # --- Un tope general demasiado bajo no deja invertir el 100% ---
        if sum(caps.values()) < 1.0 - 1e-9:
            raise ValueError(
                f"Política infactible: con los topes vigentes los {len(caps)} activos admiten sólo "
                f"{sum(caps.values()):.0%} del portafolio. Suba los topes o amplíe el universo."
            )

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        bands = " | ".join(f"{c} {lo:.0%}–{hi:.0%}" for c, (lo, hi) in self.bandas.items())
        caps = [f"{t} {c:.0%}" for t, c in self.topes_por_tipo.items() if c is not None]
        if self.max_peso_activo is not None:
            caps.insert(0, f"general {self.max_peso_activo:.0%}")
        return ((f"bandas: {bands}" if bands else "sin bandas por clase")
                + " | " + (f"topes: {', '.join(caps)}" if caps else "sin topes por activo"))


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
    .kpis{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0 8px;}
    .kpi{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:12px 18px;min-width:150px;}
    .kpi .v{font-size:22px;font-weight:600;} .kpi .l{font-size:12px;color:#666;margin-top:2px;}
    details{margin-top:28px;background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:8px 16px;}
    summary{cursor:pointer;font-weight:600;padding:6px 0;}
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
    def add_kpis(self, items: Sequence[Tuple[str, str]]) -> "InteractiveReport":
        """Tarjetas con las cifras clave (etiqueta, valor ya formateado)."""
        cards = "".join(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{label}</div></div>"
                        for label, v in items)
        self._blocks.append(f"<div class='kpis'>{cards}</div>")
        return self

    # ------------------------------------------------------------------ #
    def add_details(self, title: str, html: str) -> "InteractiveReport":
        """Bloque plegado (se abre con un clic) para el detalle que no es central."""
        self._blocks.append(f"<details><summary>{title}</summary>{html}</details>")
        return self

    # ------------------------------------------------------------------ #
    @staticmethod
    def table_html(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
        return ("<div class='wrap'>"
                + df.to_html(float_format=lambda v: float_format.format(v), border=0, na_rep="—")
                + "</div>")

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
# 12. EJECUCIÓN PRINCIPAL
# ==============================================================================

MODOS_EJECUCION = ("recomendacion", "validacion")


def _resolver_ventana() -> Tuple[str, str]:
    """Fechas de inicio y fin de los datos según la sección 2 de la configuración."""
    end = pd.Timestamp(FECHA_FIN) if FECHA_FIN else pd.Timestamp.today().normalize()
    if FECHA_INICIO:
        start = pd.Timestamp(FECHA_INICIO)
    else:
        start = end - pd.Timedelta(days=int(round(HISTORIA_ANIOS * 365.25)))
    if start >= end:
        raise ValueError(
            f"FECHA_INICIO ({start.date()}) debe ser anterior a FECHA_FIN ({end.date()})."
        )
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _curva_historica_real(
    start_date: str, end_date: str, plazos: Sequence[float], alertas: List[str]
) -> Optional[pd.DataFrame]:
    """
    Histórico diario de la curva TES hasta `end_date`, o None si se usa la curva
    simulada. Con PERMITIR_DATOS_SINTETICOS = False (modo para invertir) una
    curva que no se pueda descargar, o que termine antes de la fecha de cierre,
    detiene la corrida: la tasa libre de riesgo y todo el retorno de la renta
    fija salen de ella y no deben inventarse.
    """
    if CURVA_TES_FUENTE == "simulada":
        if not PERMITIR_DATOS_SINTETICOS:
            raise ValueError(
                "CURVA_TES_FUENTE = 'simulada' sólo sirve para pruebas: pon 'banrep' o "
                "PERMITIR_DATOS_SINTETICOS = True."
            )
        return None
    if CURVA_TES_FUENTE != "banrep":
        raise ValueError(f"CURVA_TES_FUENTE debe ser 'banrep' o 'simulada', no '{CURVA_TES_FUENTE}'.")

    # --- Descarga (con caché local) ---
    try:
        nodos = BanrepTESHistory(RUTA_CACHE_CURVA_TES).fetch(
            pd.Timestamp(start_date) - pd.Timedelta(days=15), end_date
        )
        if nodos.empty:
            raise ValueError("la serie de Banrep no trae datos en la ventana pedida")
    except Exception as exc:
        if not PERMITIR_DATOS_SINTETICOS:
            raise RuntimeError(
                f"No se pudo obtener la curva TES real del Banco de la República ({exc}). Revisa la "
                "conexión a internet y vuelve a correr. La corrida se detiene porque con "
                "PERMITIR_DATOS_SINTETICOS = False no se usa una curva simulada para invertir."
            ) from exc
        logger.warning(
            "No se pudo obtener la curva TES histórica de Banrep (%s). Se usa la curva simulada: "
            "los resultados de renta fija no reflejan el mercado.", exc,
        )
        alertas.append("Curva TES simulada: los resultados de renta fija NO reflejan el mercado.")
        return None

    # --- La curva debe llegar hasta la fecha de cierre ---
    rezago = (pd.Timestamp(end_date) - nodos.index[-1]).days
    if rezago > MAX_DIAS_DATOS_DESACTUALIZADOS:
        msg = (f"La curva TES de Banrep llega hasta {nodos.index[-1].date()}, {rezago} días antes de "
               f"FECHA_FIN ({end_date}); el máximo permitido es {MAX_DIAS_DATOS_DESACTUALIZADOS}.")
        if not PERMITIR_DATOS_SINTETICOS:
            raise RuntimeError(msg + " Vuelve a correr cuando Banrep publique datos más recientes.")
        logger.warning(msg)
        alertas.append(msg)
    return BanrepTESHistory.complete_curve(nodos, plazos, CURVA_TES_TAU_NS)


def _descripcion_activo(spec: AssetSpec) -> str:
    """Nombre legible de un activo para las tablas de resultados."""
    if spec.sub_class is AssetSubClass.RF_BONO:
        return f"{spec.name} — ISIN {spec.metadata['isin']}, vence {spec.metadata['vencimiento']}"
    return spec.name


def _formatear_medidas(tabla: pd.DataFrame) -> pd.DataFrame:
    """Medidas de uno o varios portafolios en formato legible (porcentajes)."""
    pct = "{:.2%}".format
    out = pd.DataFrame(index=tabla.index)
    out["retorno_esperado_E.A."] = tabla["retorno_esperado_anual"].map(pct)
    out["volatilidad_anual"] = tabla["volatilidad_anual"].map(pct)
    out["sharpe"] = tabla["sharpe"].map("{:.2f}".format)
    out["CVaR_horizonte"] = tabla["cvar_horizonte"].map(pct)
    out["VaR_horizonte"] = tabla["var_horizonte"].map(pct)
    out["activos"] = tabla["num_activos"].astype(int)
    out["activos_efectivos"] = tabla["activos_efectivos"].map("{:.1f}".format)
    out["mayor_peso"] = tabla["mayor_peso"].map(pct)
    return out


def _pedir_porcentaje(mensaje: str, minimo: float, maximo: float, defecto: Optional[float]) -> Optional[float]:
    """Pide un porcentaje por consola (escrito como 8 u 8%) dentro de [minimo, maximo]; Enter = defecto."""
    while True:
        texto = input(mensaje).strip().replace("%", "").replace(",", ".")
        if not texto:
            return defecto
        try:
            valor = float(texto) / 100.0
        except ValueError:
            print("  Escribe un número, por ejemplo 8 o 8%.")
            continue
        if minimo - 1e-9 <= valor <= maximo + 1e-9:
            return valor
        print(f"  Fuera del rango alcanzable [{minimo:.2%}, {maximo:.2%}]. Intenta de nuevo.")


def _elegir_objetivo(spec: ObjectiveSpec, rangos: Dict[str, Tuple[float, float]]) -> ObjectiveSpec:
    """
    Pregunta por consola el criterio y el objetivo de riesgo (Enter conserva lo
    configurado). No pregunta si PREGUNTAR_ANTES_DE_OPTIMIZAR es False o si no
    hay terminal interactiva.
    """
    if not (PREGUNTAR_ANTES_DE_OPTIMIZAR and sys.stdin is not None and sys.stdin.isatty()):
        return spec

    claves = list(CRITERIOS_OPTIMIZACION)
    print("\nCriterios de optimización:")
    for i, clave in enumerate(claves, 1):
        print(f"  {i}. {CRITERIOS_OPTIMIZACION[clave]}" + ("   <- configurado" if clave == spec.criterion else ""))
    while True:
        texto = input(f"Criterio [1-{len(claves)}] (Enter = conservar): ").strip()
        if not texto:
            break
        if texto.isdigit() and 1 <= int(texto) <= len(claves):
            spec = replace(spec, criterion=claves[int(texto) - 1])
            break
        print("  Opción no válida.")

    if spec.needs_vol:
        lo, hi = rangos["vol"]
        medio = (lo + hi) / 2.0
        defecto = spec.vol_target if spec.vol_target is not None else medio
        valor = _pedir_porcentaje(
            f"Volatilidad anual objetivo en % [{lo:.2%} a {hi:.2%}] (Enter = {defecto:.2%}): ", lo, hi, defecto)
        spec = replace(spec, vol_target=valor)
    elif spec.needs_cvar:
        lo, hi = rangos["cvar"]
        medio = (lo + hi) / 2.0
        defecto = spec.cvar_target if spec.cvar_target is not None else medio
        valor = _pedir_porcentaje(
            f"CVaR {spec.cvar_level:.0%} a {spec.horizon_days} días objetivo en % [{lo:.2%} a {hi:.2%}] "
            f"(Enter = {defecto:.2%}): ", lo, hi, defecto)
        spec = replace(spec, cvar_target=valor)
    return spec


def _fig_pesos_riesgo(tabla: pd.DataFrame) -> go.Figure:
    """Barras horizontales: peso de cada activo frente a su contribución al riesgo."""
    t = tabla.sort_values("peso_%")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=t.index, x=t["peso_%"], orientation="h", name="Peso (%)",
                         marker_color="#2E6F9E"))
    fig.add_trace(go.Bar(y=t.index, x=t["contribucion_riesgo_%"], orientation="h",
                         name="Contribución a la varianza (%)", marker_color="#C77B30"))
    fig.update_layout(barmode="group", xaxis_title="%", height=max(340, 46 * len(t)),
                      legend=dict(orientation="h", y=1.08))
    return fig


def _fig_distribucion(dist_subclase: pd.DataFrame) -> go.Figure:
    """Anillo RV/RF con su desglose por sub-clase."""
    sub = dist_subclase["peso_%"][dist_subclase["peso_%"] > 0.05]
    clase_de = {nombre: nombre.split(" — ")[0] for nombre in sub.index}
    por_clase = sub.groupby(pd.Series(clase_de)).sum()
    fig = go.Figure(go.Sunburst(
        labels=list(por_clase.index) + list(sub.index),
        parents=[""] * len(por_clase) + [clase_de[n] for n in sub.index],
        values=list(por_clase.values) + list(sub.values),
        branchvalues="total", texttemplate="%{label}<br>%{value:.1f}%",
    ))
    fig.update_layout(height=420, margin=dict(t=20, b=20))
    return fig


def _fig_curva_activos(curva: pd.DataFrame, elegido: int, spec: ObjectiveSpec) -> go.Figure:
    """Cómo cambia el objetivo del criterio al variar el número de activos; marca el elegido."""
    columna, titulo, formato = {
        "max_sharpe": ("sharpe", "Sharpe esperado", ".2f"),
        "min_varianza": ("volatilidad_anual", "Volatilidad anual", ".2%"),
        "min_cvar": ("cvar_horizonte", f"CVaR {spec.cvar_level:.0%} a {spec.horizon_days} días", ".2%"),
    }.get(spec.criterion, ("retorno_esperado_anual", "Retorno esperado (E.A.)", ".2%"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curva.index, y=curva[columna], mode="lines+markers", name=titulo,
                             line=dict(color="#2E6F9E")))
    if elegido in curva.index:
        fig.add_trace(go.Scatter(x=[elegido], y=[curva.loc[elegido, columna]], mode="markers",
                                 name=f"Elegido: {elegido} activos",
                                 marker=dict(symbol="star", size=18, color="red")))
    fig.update_layout(xaxis_title="Número de activos", yaxis_title=titulo, yaxis_tickformat=formato,
                      xaxis=dict(dtick=1), height=380)
    return fig


def main() -> None:
    # ---------------------------------------------------------------- #
    # 0) CONFIGURACIÓN: MODO, VENTANA, BONOS Y POLÍTICA
    # ---------------------------------------------------------------- #
    if MODO_EJECUCION not in MODOS_EJECUCION:
        raise ValueError(f"MODO_EJECUCION debe ser uno de {MODOS_EJECUCION}, no '{MODO_EJECUCION}'.")
    modo_validacion = MODO_EJECUCION == "validacion"
    os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)

    logger.info("=" * 78)
    logger.info("QUANT PM — MOTOR DE ASSET ALLOCATION MULTI-ACTIVO (RV + RF) — modo: %s", MODO_EJECUCION)
    logger.info("=" * 78)

    start_date, end_date = _resolver_ventana()
    alertas: List[str] = []
    hoy = pd.Timestamp.today().normalize()
    if not modo_validacion and (hoy - pd.Timestamp(end_date)).days > MAX_DIAS_DATOS_DESACTUALIZADOS:
        msg = (f"FECHA_FIN = {end_date} es anterior a hoy ({hoy.date()}): el portafolio refleja el "
               "mercado de esa fecha, no el actual. Pon FECHA_FIN = None para invertir hoy.")
        logger.warning(msg)
        alertas.append(msg)
    logger.info("Ventana de datos: %s a %s.", start_date, end_date)
    anios_datos = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    if anios_datos < CVAR_VENTANA_ANIOS:
        raise ValueError(
            f"La ventana de datos ({anios_datos:.1f} años) es menor que la ventana del CVaR "
            f"({CVAR_VENTANA_ANIOS:g} años): sube HISTORIA_ANIOS o baja CVAR_VENTANA_ANIOS."
        )
    if modo_validacion and anios_datos < CVAR_VENTANA_ANIOS + 1:
        logger.warning("Quedan menos de 1 año de datos para el backtest (%.1f años de datos, %g de ventana).",
                       anios_datos, CVAR_VENTANA_ANIOS)

    # --- Bonos individuales ingresados por el usuario (se validan antes de descargar nada) ---
    bonos = BondSpec.list_from_config(BONOS_UNIVERSO, end_date)
    logger.info("BONOS_UNIVERSO: %d bono(s) ingresado(s).", len(bonos))

    universe = AssetUniverse()
    policy = AllocationPolicy()

    # ---------------------------------------------------------------- #
    # 1) PRECIOS DE MERCADO (RV + ETFs de RF — Enfoque B)
    # ---------------------------------------------------------------- #
    # --- Descarga de precios y retornos ---
    logger.info("Descargando/generando precios para %d tickers de mercado.", len(universe.all_tickers))
    pipeline = MarketDataPipeline(
        universe.all_tickers + [universe.benchmark], start_date, end_date, seed=SEMILLA_ALEATORIA
    )
    prices = pipeline.fetch_prices()
    returns = pipeline.compute_returns(prices)
    logger.info("Precios obtenidos: %d filas x %d activos.", *prices.shape)

    # --- Selección del universo (datos reales vs. modo demostración) ---
    # Sólo se decide aquí lo que no depende de la fecha: tener precio real. La
    # historia, la cotización reciente y la liquidez se evalúan en cada
    # rebalanceo con la ventana de estimación (`tradability_issues_at`).
    def _reales(tickers: Sequence[str]) -> List[str]:
        return [tk for tk in tickers if tk not in pipeline.synthetic_tickers_]

    modo_demo = len(_reales(universe.rv_tickers)) < MIN_ACTIVOS_RV_REALES
    if modo_demo and not PERMITIR_DATOS_SINTETICOS:
        raise RuntimeError(
            f"Sólo {len(_reales(universe.rv_tickers))} tickers de RV con precio real (mínimo "
            f"{MIN_ACTIVOS_RV_REALES}). Revisa la conexión a internet y que yfinance esté instalado "
            f"({'instalado' if _YFINANCE_AVAILABLE else 'NO instalado: pip install yfinance'}). "
            "Para probar el código sin internet pon PERMITIR_DATOS_SINTETICOS = True; esos "
            "resultados no sirven para invertir."
        )
    if modo_demo:
        logger.warning(
            "MODO DEMOSTRACIÓN: sólo %d tickers de RV con precio real (< %d). Se optimiza sobre "
            "series sintéticas calibradas; los resultados no son inferencia de mercado.",
            len(_reales(universe.rv_tickers)), MIN_ACTIVOS_RV_REALES,
        )
        alertas.append("MODO DEMOSTRACIÓN: series sintéticas, los resultados NO sirven para invertir.")
    if EXCLUIR_ACTIVOS_SIN_PRECIO_REAL and not modo_demo:
        rv_tickers, rf_etf_tickers = _reales(universe.rv_tickers), _reales(universe.rf_etf_tickers)
    else:
        rv_tickers, rf_etf_tickers = list(universe.rv_tickers), list(universe.rf_etf_tickers)
    tickers_mercado = rv_tickers + rf_etf_tickers

    # --- Frescura de los precios frente a la fecha de cierre ---
    ultimos_precios = {tk: s.index[-1] for tk, s in pipeline.raw_prices_.items()}
    if ultimos_precios:
        ultimo_precio = max(ultimos_precios.values())
        rezago = (pd.Timestamp(end_date) - ultimo_precio).days
        if rezago > MAX_DIAS_DATOS_DESACTUALIZADOS:
            msg = (f"El último precio de mercado es del {ultimo_precio.date()}, {rezago} días antes de "
                   f"FECHA_FIN ({end_date}); el máximo permitido es {MAX_DIAS_DATOS_DESACTUALIZADOS}.")
            if not PERMITIR_DATOS_SINTETICOS:
                raise RuntimeError(msg + " Revisa la fuente de datos (Yahoo Finance) y vuelve a correr.")
            logger.warning(msg)
            alertas.append(msg)

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
    plazos_curva = sorted(set(float(t) for t in CURVA_TES_PLAZOS) | set(float(t) for t in universe.nodos_tes))
    curve_history = _curva_historica_real(start_date, end_date, plazos_curva, alertas)

    # --- Curva base: la vigente en la fecha de cierre (donde se cotizan los bonos) ---
    if curve_history is not None:
        fila_cierre = curve_history.loc[:end_date]
        fila_cierre = fila_cierre.iloc[-1] if not fila_cierre.empty else curve_history.iloc[-1]
        curve = TESYieldCurve.from_levels(plazos_curva, fila_cierre.values)
        logger.info("Curva TES: histórico real de Banrep (1, 5 y 10 años, completado con Nelson-Siegel "
                    "τ=%.2f), %d días, hasta el %s.", CURVA_TES_TAU_NS, len(curve_history),
                    curve_history.index[-1].date())
    else:
        curve = TESYieldCurve.from_config()
        logger.info("Curva TES: curva fija de configuración con choques simulados.")

    # ---------------------------------------------------------------- #
    # 3) MOTOR DE RENTA FIJA — ENFOQUES A + B + C
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("RENTA FIJA — NODOS TES (A) + ETFs/FICs (B) + BONOS FILTRADOS (C)")

    # --- Construcción del motor de renta fija (bonos cotizados a la fecha de cierre) ---
    criteria = BondScreeningCriteria()
    fi_engine = FixedIncomeEngine(
        curve=curve,
        node_tenors=universe.nodos_tes,
        bonds=bonos,
        criteria=criteria,
        as_of=end_date,
        include_rolldown=INCLUIR_ROLLDOWN,
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
    if not fi_bundle.screening_report.empty:
        print("\n--- Screening de bonos individuales (Enfoque C) a la fecha de cierre ---")
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
    multi_returns = MarketDataPipeline.compute_returns(multi_prices)
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
    # 6) PORTAFOLIO ÓPTIMO A LA FECHA DE CIERRE
    # ---------------------------------------------------------------- #
    logger.info("-" * 78)
    logger.info("OPTIMIZACIÓN MULTI-ACTIVO (%s)", policy.describe())
    policy.validate(specs)

    # --- μ analítico de la RF al cierre ---
    fecha_cierre = multi_returns.index[-1]
    mu_rf_hoy = np.log1p(fi_engine.expected_returns_at(fecha_cierre))

    # --- Universo invertible al cierre: bonos y activos de mercado ---
    bonos_no_elegibles = fi_engine.excluded_bonds_at(fecha_cierre)
    if bonos_no_elegibles:
        logger.info("Bonos no elegibles al %s (plazo, spread o emisor): %s",
                    fecha_cierre.date(), sorted(bonos_no_elegibles))
    # --- Ventanas: la cola del CVaR usa historia larga; retorno y covarianza, la corta ---
    dias_cvar = int(round(CVAR_VENTANA_ANIOS * TRADING_DAYS))
    ventana_larga = multi_returns.tail(dias_cvar)
    if len(ventana_larga) < 0.98 * dias_cvar:
        raise ValueError(f"Sólo hay {len(ventana_larga)} días de retornos y la ventana del CVaR pide "
                         f"{dias_cvar}: sube HISTORIA_ANIOS o baja CVAR_VENTANA_ANIOS.")
    inicio_corta = ventana_larga.index[-min(LOOKBACK_DIAS, len(ventana_larga))]
    motivos_corta = pipeline.tradability_issues_at(fecha_cierre, inicio_corta, tickers_mercado)
    motivos_cierre = pipeline.tradability_issues_at(fecha_cierre, ventana_larga.index[0], tickers_mercado)
    if motivos_cierre:
        logger.info("Activos de mercado no invertibles al %s (ventana %s–%s): %s",
                    fecha_cierre.date(), ventana_larga.index[0].date(), fecha_cierre.date(),
                    "; ".join(f"{tk} ({m})" for tk, m in sorted(motivos_cierre.items())))
    corta = ventana_larga.tail(LOOKBACK_DIAS)
    sin_historia_cvar = sorted(
        (set(motivos_cierre) | set(ventana_larga.columns[ventana_larga.isna().any()]))
        - set(motivos_corta) - set(corta.columns[corta.isna().any()])
    )
    if sin_historia_cvar:
        alertas.append(
            f"{len(sin_historia_cvar)} activo(s) cumplen la ventana de estimación pero no la del CVaR "
            f"({CVAR_VENTANA_ANIOS:g} años: historia insuficiente o poco líquidos) y quedan fuera: "
            f"{', '.join(sin_historia_cvar)}. Baja CVAR_VENTANA_ANIOS si quieres incluirlos, a costa "
            "de una cola menos confiable."
        )

    # --- Retornos pro-forma del universo invertible ---
    no_invertibles_cierre = (bonos_no_elegibles | set(motivos_cierre)
                             | set(ventana_larga.columns[ventana_larga.isna().any()]))
    retornos_cola = fi_engine.proforma_log_returns(
        fecha_cierre,
        ventana_larga.drop(columns=sorted(no_invertibles_cierre & set(ventana_larga.columns))),
    )

    optimizer = PortfolioOptimizer(
        retornos_cola.tail(LOOKBACK_DIAS),
        tail_returns=retornos_cola,
        rf=rf_tes,
        max_weight=policy.max_weights(specs),
        shrinkage=USAR_SHRINKAGE_LEDOIT_WOLF,
        asset_groups={tk: s.asset_class.value for tk, s in specs.items()},
        group_constraints=policy.group_constraints(),
        expected_returns=mu_rf_hoy,
    )

    # --- Rango alcanzable y elección del criterio y los objetivos ---
    objetivo = ObjectiveSpec()
    tabla_rango, rangos = optimizer.achievable_ranges(objetivo)
    print(f"\n--- Rango alcanzable con los datos al {fecha_cierre.date()} "
          f"(CVaR {objetivo.cvar_level:.0%} a {objetivo.horizon_days} días) ---")
    print(_formatear_medidas(tabla_rango).to_string())
    objetivo = _elegir_objetivo(objetivo, rangos)
    logger.info("Criterio de optimización: %s", objetivo.label)

    diag = optimizer.tail_diagnostics(objetivo)
    logger.info("Cola del CVaR: %.1f años (%d observaciones), %.0f en la cola; mínimo recomendado %.1f años.",
                diag["anios"], diag["observaciones"], diag["obs_en_la_cola"], diag["anios_minimos"])
    if diag["obs_en_la_cola"] < CVAR_MIN_OBS_COLA:
        alertas.append(
            f"El CVaR se apoya en sólo {diag['obs_en_la_cola']:.0f} observaciones en la cola (mínimo "
            f"recomendado {CVAR_MIN_OBS_COLA}): se necesitan al menos {diag['anios_minimos']:.1f} años con "
            f"retornos {objetivo.return_freq}s (hay {diag['anios']:.1f}). Trátalo como una estimación aproximada."
        )

    weights_opt, info_opt = optimizer.solve(objetivo, rangos)
    if info_opt["nota"]:
        logger.warning(info_opt["nota"])
        alertas.append(str(info_opt["nota"]))
    medidas = optimizer.measures(weights_opt, objetivo)
    riesgo_contrib = optimizer.risk_contribution(weights_opt)

    # --- Tabla de pesos (con montos si hay capital) ---
    tabla_pesos = pd.DataFrame({
        "activo": [_descripcion_activo(specs[t]) for t in weights_opt.index],
        "clase": [specs[t].asset_class.value for t in weights_opt.index],
        "sub_clase": [specs[t].sub_class.value for t in weights_opt.index],
        "peso_%": weights_opt * 100,
        "contribucion_riesgo_%": riesgo_contrib * 100,
    }, index=weights_opt.index)
    if CAPITAL_INVERSION_COP:
        tabla_pesos["monto_COP"] = weights_opt * CAPITAL_INVERSION_COP
    tabla_pesos = tabla_pesos.sort_values("peso_%", ascending=False)
    tabla_pesos_visible = tabla_pesos[tabla_pesos["peso_%"] > PESO_MIN_REPORTE_PCT]

    # --- Distribución resultante por clase y sub-clase (RV vs RF sale del optimizador) ---
    def _distribucion(grupos: Mapping[str, str]) -> pd.DataFrame:
        dist = (weights_opt.groupby(pd.Series(grupos)).sum() * 100).rename("peso_%").to_frame()
        if CAPITAL_INVERSION_COP:
            dist["monto_COP"] = dist["peso_%"] / 100 * CAPITAL_INVERSION_COP
        return dist.sort_values("peso_%", ascending=False)

    dist_clase = _distribucion({t: specs[t].asset_class.value for t in weights_opt.index})
    dist_subclase = _distribucion({t: specs[t].sub_class.value for t in weights_opt.index})
    peso_rv = float(dist_clase["peso_%"].get(AssetClass.RENTA_VARIABLE.value, 0.0))
    peso_rf = float(dist_clase["peso_%"].get(AssetClass.RENTA_FIJA.value, 0.0))

    n_activos = int(medidas["num_activos"])
    if not policy.bandas and policy.max_peso_activo is None and not any(policy.topes_por_tipo.values()):
        if n_activos < 4 or medidas["mayor_peso"] > 0.5:
            alertas.append(
                f"Sin bandas ni topes el portafolio quedó concentrado ({n_activos} activos; "
                f"mayor peso {medidas['mayor_peso']:.0%} en {weights_opt.idxmax()}). Si quieres más "
                "diversificación, activa topes o bandas en la sección 9 de la configuración."
            )

    print(f"\n--- Portafolio óptimo al {fecha_cierre.date()} — {objetivo.label} ---")
    print(tabla_pesos_visible.round(2).to_string())
    print("\n--- Distribución por clase de activo (resultado del optimizador) ---")
    print(dist_clase.round(2).to_string())
    print("\n--- Medidas del portafolio ---")
    print(_formatear_medidas(pd.DataFrame({"Elegido": medidas}).T).to_string())
    if policy.bandas:
        print("\n--- Cumplimiento de bandas ---")
        print(optimizer.constraints_check(weights_opt).to_string())

    # --- Diagnóstico de shrinkage ---
    logger.info(
        "Intensidad de shrinkage Ledoit-Wolf: δ = %.4f (0 = covarianza muestral pura, 1 = target)",
        optimizer.shrinkage_intensity_,
    )
    for clase, phi in optimizer.mu_shrinkage_.items():
        logger.info("Bayes-Stein sobre μ de %s: φ = %.3f (0 = media muestral, 1 = media común).",
                    clase, phi)

    # --- Calidad y frescura de los datos al cierre ---
    estado_cierre = {tk: motivos_cierre.get(tk, "elegible") for tk in tickers_mercado}
    estado_cierre.update({tk: motivos_exclusion[tk] for tk in descartados})
    filas_calidad = {}
    for tk, estado in estado_cierre.items():
        if estado == "elegible":
            continue
        fila = {"estado_al_cierre": estado, "ultimo_dato": "—", "dias_de_rezago": np.nan}
        if tk in liquidez.index:
            fila["ultimo_dato"] = liquidez.loc[tk, "ultimo_dato"].date()
            fila["dias_de_rezago"] = float((fecha_cierre - liquidez.loc[tk, "ultimo_dato"]).days)
        filas_calidad[tk] = fila
    tabla_excluidos = pd.DataFrame.from_dict(filas_calidad, orient="index")

    # ---------------------------------------------------------------- #
    # 7) BACKTEST WALK-FORWARD (sólo en modo "validacion")
    # ---------------------------------------------------------------- #
    backtester = None
    if modo_validacion:
        logger.info("-" * 78)
        logger.info("BACKTEST WALK-FORWARD FUERA DE MUESTRA — %s", objetivo.label)

        # --- Universo invertible en cada rebalanceo (sólo datos de la ventana) ---
        def no_invertibles(fecha: pd.Timestamp, inicio_ventana: pd.Timestamp) -> Set[str]:
            motivos = pipeline.tradability_issues_at(fecha, inicio_ventana, tickers_mercado)
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
            objective=objetivo,
            cvar_lookback_days=dias_cvar,
        )
        backtester.run()
        perf_summary = backtester.performance_summary()

        # --- Salida en consola del backtest ---
        print("\n--- Asignación por clase de activo en cada rebalanceo (%) ---")
        print((backtester.class_weights_history_ * 100).round(2).to_string())
        if policy.bandas:
            print("\n--- Cumplimiento de bandas a lo largo del backtest ---")
            print(backtester.band_compliance().to_string())
        print("\n--- Desempeño fuera de muestra ---")
        print(pd.Series(perf_summary, name="valor").round(4).to_string())

    # ---------------------------------------------------------------- #
    # 8) REPORTE HTML INTERACTIVO (sólo lo relevante)
    # ---------------------------------------------------------------- #
    report = InteractiveReport(
        "Portafolio óptimo — Mercado colombiano",
        f"Datos al {fecha_cierre.date()} | Modo: {MODO_EJECUCION} | Criterio: {objetivo.label} | "
        f"Universo: {n_rv} activos de Renta Variable y {n_rf} de Renta Fija | "
        f"Ventana de estimación: {LOOKBACK_DIAS} días | Restricciones: {policy.describe()}",
    )
    for alerta in alertas:
        report.add_note(f"<b>⚠ Alerta.</b> {alerta}")

    # --- Medidas clave ---
    capital_txt = f"{CAPITAL_INVERSION_COP / 1e6:,.0f} M COP" if CAPITAL_INVERSION_COP else "—"
    cvar_lo, cvar_hi = optimizer.cvar_interval(weights_opt.to_numpy(), objetivo)
    report.add_kpis([
        ("Retorno esperado (E.A.)", f"{medidas['retorno_esperado_anual']:.1%}"),
        ("Volatilidad anual", f"{medidas['volatilidad_anual']:.1%}"),
        ("Sharpe esperado", f"{medidas['sharpe']:.2f}"),
        (f"CVaR {objetivo.cvar_level:.0%} a {objetivo.horizon_days} días (IC 90%: {cvar_lo:.1%} a {cvar_hi:.1%})",
         f"{medidas['cvar_horizonte']:.1%}"),
        (f"VaR {objetivo.cvar_level:.0%} a {objetivo.horizon_days} días", f"{medidas['var_horizonte']:.1%}"),
        ("Renta variable / fija", f"{peso_rv:.0f}% / {peso_rf:.0f}%"),
        ("Activos (efectivos)", f"{n_activos} ({medidas['activos_efectivos']:.1f})"),
        ("Tasa libre de riesgo", f"{rf_tes:.1%}"),
        ("Capital", capital_txt),
    ])

    # --- Portafolio: tabla y gráficos ---
    otros = tabla_pesos[tabla_pesos["peso_%"] <= PESO_MIN_REPORTE_PCT]
    tabla_html = tabla_pesos_visible.set_index("activo")[["clase", "peso_%", "contribucion_riesgo_%"]
                                                          + (["monto_COP"] if CAPITAL_INVERSION_COP else [])].copy()
    if len(otros) and otros["peso_%"].sum() > 0:
        fila_otros = {"clase": "", "peso_%": float(otros["peso_%"].sum()),
                      "contribucion_riesgo_%": float(otros["contribucion_riesgo_%"].sum())}
        if CAPITAL_INVERSION_COP:
            fila_otros["monto_COP"] = float(otros["monto_COP"].sum())
        tabla_html.loc[f"Otros ({int((otros['peso_%'] > 1e-4).sum())} activos, cada uno bajo "
                       f"{PESO_MIN_REPORTE_PCT:g}%)"] = fila_otros
    tabla_html["peso_%"] = tabla_html["peso_%"].map("{:.1f}%".format)
    tabla_html["contribucion_riesgo_%"] = tabla_html["contribucion_riesgo_%"].map("{:.1f}%".format)
    if CAPITAL_INVERSION_COP:
        tabla_html["monto_COP"] = tabla_html["monto_COP"].map("{:,.0f}".format)
    tabla_html.index.name = None
    cap_txt = f" — capital {CAPITAL_INVERSION_COP:,.0f} COP" if CAPITAL_INVERSION_COP else ""
    report.add_table(f"Portafolio recomendado al {fecha_cierre.date()}{cap_txt}", tabla_html)
    report.add_chart("Peso y contribución al riesgo por activo",
                     _fig_pesos_riesgo(tabla_pesos_visible))
    report.add_chart("Distribución por clase y sub-clase", _fig_distribucion(dist_subclase))

    if "curva_activos" in info_opt and len(info_opt["curva_activos"]) >= 3:
        report.add_chart(
            f"Objetivo según el número de activos (elegido: {info_opt['activos_elegidos']})",
            _fig_curva_activos(info_opt["curva_activos"], int(info_opt["activos_elegidos"]), objetivo),
        )

    # --- Rango alcanzable y frontera ---
    report.add_table(
        f"Rango alcanzable y portafolios de referencia (CVaR {objetivo.cvar_level:.0%} a "
        f"{objetivo.horizon_days} días"
        + (f"; mínimo {objetivo.min_assets} activos" if objetivo.min_assets else "") + ")",
        _formatear_medidas(tabla_rango),
    )
    with optimizer.diversified(objetivo):      # misma restricción de mínimo de activos que el resultado
        figura_frontera = optimizer.plot_efficient_frontier(weights_opt, objetivo.label,
                                                            optimizer.reference_weights_)
    report.add_chart("Frontera eficiente", figura_frontera)

    # --- Renta fija ---
    report.add_chart(f"Curva TES al {fecha_cierre.date()}",
                     fi_engine.shocks.curve_at(fecha_cierre).plot_curve())
    if not fi_bundle.screening_report.empty:
        scr = fi_bundle.screening_report[
            ["emisor", "rating", "plazo_anios", "ytm", "duracion_mod", "spread_bp",
             "aprobado", "motivo_rechazo"]
        ].copy()
        scr["ytm"] = scr["ytm"].map("{:.2%}".format)
        scr["plazo_anios"] = scr["plazo_anios"].map("{:.1f}".format)
        scr["duracion_mod"] = scr["duracion_mod"].map("{:.2f}".format)
        scr["spread_bp"] = scr["spread_bp"].map("{:.0f}".format)
        scr["aprobado"] = scr["aprobado"].map({True: "sí", False: "no"})
        scr.index.name = None
        report.add_table("Screening de bonos individuales (a la fecha de cierre)", scr)

    # --- Validación: backtest ---
    if backtester is not None:
        report.add_chart("Backtest — Curva de equity", backtester.plot_equity_curve())
        report.add_chart("Backtest — Asignación RV vs RF en el tiempo",
                         backtester.plot_asset_class_evolution())
        report.add_table("Backtest — Desempeño fuera de muestra",
                         pd.DataFrame(perf_summary, index=["valor"]).T)

    # --- Detalle plegado: calidad de datos y metodología ---
    if not tabla_excluidos.empty:
        report.add_details(
            f"Activos no invertibles al cierre ({len(tabla_excluidos)})",
            InteractiveReport.table_html(tabla_excluidos, "{:.0f}"),
        )
    confiabilidad = pd.DataFrame({"valor": {
        "Años de historia de la cola": f"{diag['anios']:.1f}",
        "Observaciones (retornos " + objetivo.return_freq + "s)": f"{diag['observaciones']:.0f}",
        "Observaciones en la cola": f"{diag['obs_en_la_cola']:.0f} (mínimo recomendado {CVAR_MIN_OBS_COLA})",
        "Escenarios simulados al horizonte": f"{objetivo.n_scenarios}",
        "CVaR del portafolio": f"{medidas['cvar_horizonte']:.2%}",
        "Intervalo de confianza 90% del CVaR": f"{cvar_lo:.2%} a {cvar_hi:.2%}",
    }})
    report.add_details("Confiabilidad del CVaR", InteractiveReport.table_html(confiabilidad))
    report.add_details(
        "Metodología y límites",
        "<p><b>Datos.</b> Precios de Yahoo Finance y curva cero cupón TES en pesos del Banco de la "
        "República (1, 5 y 10 años, completada con Nelson-Siegel). Los spreads de crédito de los bonos "
        "se anclan a tu cotización en la fecha de los datos; hacia atrás son simulados.</p>"
        f"<p><b>Estimación.</b> Ventana de {LOOKBACK_DIAS} días; covarianza con shrinkage Ledoit-Wolf; "
        "retorno esperado de acciones y ETFs con Bayes-Stein; el de la renta fija es analítico "
        "(YTM + roll-down).</p>"
        f"<p><b>CVaR.</b> Bootstrap por bloques (~{objetivo.block_days} días) de los retornos "
        f"{objetivo.return_freq}s continuos de los últimos {CVAR_VENTANA_ANIOS:g} años, encadenados hasta "
        f"{objetivo.horizon_days} días"
        + (" y recentrados en el retorno esperado" if objetivo.center_on_mu else "")
        + f". Se simulan {objetivo.n_scenarios} trayectorias; el intervalo de confianza refleja cuánto "
        "podría cambiar el CVaR con otra muestra de la historia.</p>"
        "<p><b>Número de activos.</b> Heurística de eliminación hacia atrás (no garantiza el óptimo "
        "global): se parte del óptimo libre y se quitan los activos de menor peso, re-optimizando."
        "</p>"
        "<p><b>Sin topes ni bandas</b> el óptimo tiende a concentrarse; el resultado es muy sensible a "
        "los retornos esperados. Revisa los pesos antes de operar.</p>",
    )

    # --- Escritura del reporte ---
    report_path = report.write(os.path.join(DIRECTORIO_SALIDA, NOMBRE_REPORTE_HTML))

    logger.info("=" * 78)
    for alerta in alertas:
        logger.warning("ALERTA: %s", alerta)
    logger.info("Ejecución completa. Reporte interactivo: %s", report_path)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
