# ============================================================
# ValidAI Risk — Copiloto AUTOMÁTICO (Modo B: replicar el modelo)
# ------------------------------------------------------------
# Idea: el validador da un puntero a la data (Athena) + la
# especificación del modelo. El copiloto:
#   1) baja la data cruda de Athena
#   2) REPLICA el modelo (recalcula la PD)
#   3) corre la calibración sobre la PD replicada
# Devuelve un informe listo para el reporte de hallazgos.
#
# Funciona con cualquier DataFrame para poder probarlo en local.
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 1) CONECTOR DE DATOS — Amazon Athena
# ============================================================
def obtener_datos_athena(query: str,
                         database: str = "validairisk",
                         s3_output: str = "s3://validairisk-athena-results/") -> pd.DataFrame:
    """Ejecuta una consulta en Athena y devuelve un DataFrame.
    En Modo B, la query trae la DATA CRUDA (variables del modelo +
    flag de default observado), no la PD ya calculada. Ejemplo:

        SELECT edad, ingreso, deuda_ratio, antiguedad, flag_default
        FROM validairisk.cartera_scoreable
        WHERE cosecha = '2025Q4'

    Requiere awswrangler + credenciales AWS (se importa aquí para no
    exigirlo en entornos sin AWS)."""
    import awswrangler as wr
    return wr.athena.read_sql_query(query, database=database, s3_output=s3_output)


# ============================================================
# 2) REPLICACIÓN DEL MODELO — recalcular la PD
# ============================================================
def _aplicar_woe(df: pd.DataFrame, woe_maps: dict) -> pd.DataFrame:
    """Transforma variables a WoE según el mapeo del modelo.
    woe_maps = {variable: {categoria_o_bin: valor_woe}}."""
    out = df.copy()
    for var, mapa in (woe_maps or {}).items():
        if var in out.columns:
            out[var] = out[var].map(mapa).fillna(0.0)
    return out


def replicar_pd_logistica(df_raw: pd.DataFrame, especificacion: dict) -> pd.DataFrame:
    """Replica un modelo de PD logístico / scorecard sobre la data cruda.

    especificacion = {
        "intercepto": float,
        "coeficientes": {variable: beta, ...},
        "woe_maps": {variable: {bin: woe}}   # opcional (si el modelo usa WoE)
        "col_default": "flag_default",       # nombre del default observado
        "escala_rating": [(limite_pd, "grado"), ...]  # opcional
    }

    Devuelve el df con columnas nuevas: pd (PD replicada), default y rating.
    """
    b0 = float(especificacion.get("intercepto", 0.0))
    coefs = especificacion.get("coeficientes", {})
    woe_maps = especificacion.get("woe_maps")
    col_default = especificacion.get("col_default", "default")

    X = _aplicar_woe(df_raw, woe_maps) if woe_maps else df_raw.copy()

    z = np.full(len(X), b0, dtype=float)
    for var, beta in coefs.items():
        if var not in X.columns:
            raise KeyError(f"La variable '{var}' del modelo no está en la data de Athena.")
        z = z + float(beta) * pd.to_numeric(X[var], errors="coerce").fillna(0.0).to_numpy(float)

    pd_rep = 1.0 / (1.0 + np.exp(-z))   # función logística

    res = pd.DataFrame({"pd": pd_rep})
    if col_default in df_raw.columns:
        res["default"] = pd.to_numeric(df_raw[col_default], errors="coerce").astype("Int64")
    else:
        raise KeyError(f"No se encontró la columna de default '{col_default}' en la data.")

    escala = especificacion.get("escala_rating")
    if escala:
        def _grado(p):
            for limite, nombre in escala:
                if p <= limite:
                    return nombre
            return escala[-1][1]
        res["rating"] = res["pd"].apply(_grado)
    return res


# ============================================================
# 3) CALIBRACIÓN (mismos tests ya validados)
# ============================================================
_A, _R = 0.05, 0.001
def _binom(n, d, p): return stats.binomtest(d, n, p, alternative="greater").pvalue if n else float("nan")
def _jeff(n, d, p):  return float(stats.beta.cdf(p, d + 0.5, n - d + 0.5)) if n else float("nan")
def _sem(p):
    if p != p: return "SIN DATOS"
    return "VERDE" if p >= _A else ("AMBAR" if p >= _R else "ROJO")


def evaluar_calibracion(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["pd", "default"]).copy()
    df["default"] = df["default"].astype(int)
    if "rating" not in df.columns:
        df["rating"] = pd.qcut(df["pd"], 10, labels=False, duplicates="drop")
    filas = []
    for r, g in df.groupby("rating"):
        n = len(g); d = int(g["default"].sum()); pm = float(g["pd"].mean())
        pb = _binom(n, d, pm)
        filas.append({"rating": r, "n": n, "defaults": d, "pd_media": round(pm, 5),
                      "dr_obs": round(d / n, 5), "p_binomial": round(pb, 4),
                      "p_jeffreys": round(_jeff(n, d, pm), 4), "semaforo": _sem(pb)})
    t = pd.DataFrame(filas).sort_values("rating").reset_index(drop=True)
    esp = t["n"] * t["pd_media"]; den = (esp * (1 - t["pd_media"])).replace(0, np.nan)
    HL = float(((t["defaults"] - esp) ** 2 / den).sum(skipna=True)); dof = max(len(t) - 2, 1)
    hl_p = float(stats.chi2.sf(HL, dof))
    p = df["pd"].to_numpy(float); y = df["default"].to_numpy(float)
    num = float(((y - p) * (1 - 2 * p)).sum()); d2 = float(((1 - 2 * p) ** 2 * p * (1 - p)).sum()) ** 0.5
    Z = num / d2 if d2 else float("nan"); z_p = float(2 * (1 - stats.norm.cdf(abs(Z)))) if d2 else float("nan")
    return {"tabla": t, "HL": HL, "HL_gl": dof, "HL_p": hl_p, "Z": Z, "Z_p": z_p,
            "pd_media": float(p.mean()), "dr_obs": float(y.mean()),
            "semaforo": t["semaforo"].value_counts().to_dict()}


# ============================================================
# 4) PIPELINE AUTOMÁTICO — lo que el copiloto ejecuta de una
# ============================================================
def validar_calibracion_auto(query_athena: str, especificacion: dict,
                             cargador=obtener_datos_athena) -> str:
    """Flujo completo Modo B: Athena -> replicar modelo -> calibración.
    'cargador' es inyectable para poder probar en local sin AWS."""
    df_raw = cargador(query_athena)
    scores = replicar_pd_logistica(df_raw, especificacion)
    r = evaluar_calibracion(scores)
    t = r["tabla"]
    out = ["INFORME DE CALIBRACION (PD replicada por el copiloto)", "",
           "Por bucket / grado de rating:",
           f"{'rating':>8} {'n':>7} {'def':>6} {'pd_media':>9} {'dr_obs':>9} {'p_binom':>8} {'semaforo':>9}"]
    for _, row in t.iterrows():
        out.append(f"{str(row['rating']):>8} {row['n']:>7} {row['defaults']:>6} "
                   f"{row['pd_media']:>9} {row['dr_obs']:>9} {row['p_binomial']:>8} {row['semaforo']:>9}")
    out += ["",
            f"Hosmer-Lemeshow: HL={r['HL']:.3f} (gl={r['HL_gl']}), p={r['HL_p']:.4f} -> {'OK' if r['HL_p']>=_A else 'REVISAR'}",
            f"Spiegelhalter Z: Z={r['Z']:.3f}, p={r['Z_p']:.4f} -> {'OK' if r['Z_p']>=_A else 'REVISAR'}",
            f"Calibration-in-the-large: PD media={r['pd_media']:.5f} vs DR observada={r['dr_obs']:.5f} (dif={r['dr_obs']-r['pd_media']:+.5f})",
            f"Resumen semaforo: {r['semaforo']}"]
    return "\n".join(out)
