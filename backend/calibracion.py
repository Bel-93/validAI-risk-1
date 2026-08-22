# ============================================================
# ValidAI Risk — Módulo de tests de CALIBRACIÓN (PD)
# ------------------------------------------------------------
# Implementa los tests estándar de validación de calibración de
# modelos de PD para riesgo de crédito (contexto SBS / Basilea):
#
#   - Comparación PD estimada vs. tasa de default observada (DR)
#   - Test binomial (una cola)                    -> por bucket/rating
#   - Test de Jeffreys (bayesiano)                -> por bucket/rating
#   - Semáforo (traffic light, tipo Basilea)      -> por bucket/rating
#   - Hosmer-Lemeshow                             -> global
#   - Spiegelhalter Z                             -> global
#   - Brier score + calibration-in-the-large      -> global
#
# Diseñado para enchufarse a la tool MCP `calcular_calibracion`.
# La data se carga desde Athena (ver cargar_desde_athena), pero el
# módulo funciona con cualquier DataFrame para poder probarlo local.
#
# Convención de p-values: p PEQUEÑO = problema (la DR observada es
# significativamente MAYOR que la PD estimada -> subestimación).
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------
# UMBRALES DEL SEMÁFORO  (reemplazar por los del manual interno)
# ------------------------------------------------------------
ALPHA_AMBAR = 0.05    # 95%   -> por encima de esto: verde
ALPHA_ROJO  = 0.001   # 99.9% -> por debajo de esto: rojo


# ============================================================
# TESTS POR BUCKET (rating grade)
# ============================================================
def test_binomial(n: int, defaults: int, pd_estimada: float) -> float:
    """H0: la PD del bucket es correcta. Una cola (superior):
    ¿los defaults observados exceden lo esperado por la PD?
    Devuelve el p-value P(X >= defaults | X~Binom(n, pd))."""
    if n == 0:
        return np.nan
    return stats.binomtest(defaults, n, pd_estimada, alternative="greater").pvalue


def test_jeffreys(n: int, defaults: int, pd_estimada: float) -> float:
    """Test de Jeffreys (BCBS): posterior Beta(defaults+0.5, n-defaults+0.5).
    Devuelve P(theta <= pd_estimada): si es pequeño, la tasa real está
    muy probablemente POR ENCIMA de la PD (subestimación)."""
    if n == 0:
        return np.nan
    a = defaults + 0.5
    b = n - defaults + 0.5
    return float(stats.beta.cdf(pd_estimada, a, b))


def semaforo(p_value: float) -> str:
    """Verde / Ámbar / Rojo a partir del p-value del test binomial."""
    if np.isnan(p_value):
        return "SIN DATOS"
    if p_value >= ALPHA_AMBAR:
        return "VERDE"
    if p_value >= ALPHA_ROJO:
        return "AMBAR"
    return "ROJO"


# ============================================================
# TESTS GLOBALES
# ============================================================
def hosmer_lemeshow(df_buckets: pd.DataFrame) -> dict:
    """Bondad de ajuste global. df_buckets con columnas:
    n, defaults, pd_media. dof = G - 2."""
    n = df_buckets["n"].to_numpy(float)
    obs = df_buckets["defaults"].to_numpy(float)
    p = df_buckets["pd_media"].to_numpy(float)
    esp = n * p
    denom = esp * (1 - p)
    denom = np.where(denom == 0, np.nan, denom)
    hl = np.nansum((obs - esp) ** 2 / denom)
    g = len(df_buckets)
    dof = max(g - 2, 1)
    p_value = float(stats.chi2.sf(hl, dof))
    return {"estadistico_HL": float(hl), "gl": dof, "p_value": p_value,
            "resultado": "OK" if p_value >= ALPHA_AMBAR else "REVISAR"}


def spiegelhalter_z(pd_ind: np.ndarray, y_ind: np.ndarray) -> dict:
    """Test Z de Spiegelhalter a nivel individual (obligor).
    pd_ind: PD estimada de cada obligor; y_ind: default 0/1. Dos colas."""
    pd_ind = np.asarray(pd_ind, float)
    y_ind = np.asarray(y_ind, float)
    num = np.sum((y_ind - pd_ind) * (1 - 2 * pd_ind))
    den = np.sqrt(np.sum((1 - 2 * pd_ind) ** 2 * pd_ind * (1 - pd_ind)))
    if den == 0:
        return {"Z": np.nan, "p_value": np.nan, "resultado": "SIN DATOS"}
    z = num / den
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"Z": float(z), "p_value": p_value,
            "resultado": "OK" if p_value >= ALPHA_AMBAR else "REVISAR"}


def brier_score(pd_ind: np.ndarray, y_ind: np.ndarray) -> float:
    """Brier score (menor = mejor). Requiere data a nivel obligor."""
    pd_ind = np.asarray(pd_ind, float)
    y_ind = np.asarray(y_ind, float)
    return float(np.mean((pd_ind - y_ind) ** 2))


def calibration_in_the_large(pd_ind: np.ndarray, y_ind: np.ndarray) -> dict:
    """PD media estimada vs. tasa de default global observada."""
    pd_media = float(np.mean(pd_ind))
    dr_obs = float(np.mean(y_ind))
    return {"pd_media": pd_media, "dr_observada": dr_obs,
            "diferencia": dr_obs - pd_media}


# ============================================================
# ORQUESTADOR — recibe data a nivel obligor y devuelve el informe
# ============================================================
def evaluar_calibracion(df: pd.DataFrame,
                        col_pd: str = "pd",
                        col_default: str = "default",
                        col_rating: str = "rating") -> dict:
    """df a nivel obligor con: col_pd (PD estimada), col_default (0/1)
    y col_rating (grado). Si no hay rating, se crea por deciles de PD.
    Devuelve dict con tabla por bucket + tests globales."""
    df = df.copy()
    if col_rating not in df.columns:
        df[col_rating] = pd.qcut(df[col_pd], 10, labels=False, duplicates="drop")

    filas = []
    for r, g in df.groupby(col_rating):
        n = len(g)
        defaults = int(g[col_default].sum())
        pd_media = float(g[col_pd].mean())
        dr_obs = defaults / n if n else np.nan
        p_bin = test_binomial(n, defaults, pd_media)
        p_jef = test_jeffreys(n, defaults, pd_media)
        filas.append({
            "rating": r, "n": n, "defaults": defaults,
            "pd_media": round(pd_media, 5), "dr_observada": round(dr_obs, 5),
            "p_binomial": round(p_bin, 4) if not np.isnan(p_bin) else np.nan,
            "p_jeffreys": round(p_jef, 4) if not np.isnan(p_jef) else np.nan,
            "semaforo": semaforo(p_bin),
        })
    tabla = pd.DataFrame(filas).sort_values("rating").reset_index(drop=True)

    hl = hosmer_lemeshow(tabla)
    sp = spiegelhalter_z(df[col_pd].to_numpy(), df[col_default].to_numpy())
    brier = brier_score(df[col_pd].to_numpy(), df[col_default].to_numpy())
    cil = calibration_in_the_large(df[col_pd].to_numpy(), df[col_default].to_numpy())

    resumen_semaforo = tabla["semaforo"].value_counts().to_dict()
    return {
        "tabla_por_bucket": tabla,
        "hosmer_lemeshow": hl,
        "spiegelhalter_z": sp,
        "brier_score": round(brier, 5),
        "calibration_in_the_large": cil,
        "resumen_semaforo": resumen_semaforo,
    }


# ============================================================
# CARGA DESDE ATHENA (esqueleto — se activa en AWS)
# ============================================================
def cargar_desde_athena(query: str,
                        database: str = "validairisk",
                        s3_output: str = "s3://validairisk-athena-results/") -> pd.DataFrame:
    """Ejecuta una consulta en Athena y devuelve un DataFrame a nivel
    obligor con columnas pd, default y (opcional) rating.
    Requiere awswrangler + credenciales AWS. Ejemplo de query:

        SELECT pd_estimada AS pd, flag_default AS default, rating
        FROM validairisk.scores_modelo
        WHERE cosecha = '2025Q4'
    """
    import awswrangler as wr  # se importa aquí para no exigirlo en local
    return wr.athena.read_sql_query(query, database=database, s3_output=s3_output)


# ============================================================
# WRAPPER MCP — así se expone como tool en validairisk-mcp
# ============================================================
def calcular_calibracion(query_athena: str) -> dict:
    """Tool MCP: carga scores desde Athena y devuelve el informe de
    calibración serializable (la tabla como lista de registros)."""
    df = cargar_desde_athena(query_athena)
    r = evaluar_calibracion(df)
    r["tabla_por_bucket"] = r["tabla_por_bucket"].to_dict(orient="records")
    return r
