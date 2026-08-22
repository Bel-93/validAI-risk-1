# ============================================================
# ValidAI Risk — Módulo de VALIDACIÓN METODOLÓGICA (PD)
# ------------------------------------------------------------
# Rol: VALIDADOR (no monitoring). Reta cómo se construyó el modelo
# y si se mantiene, con pruebas consistentes alineadas a SBS/Basilea.
#
# Pilares que cubre esta tool (una sola, mínima complejidad):
#   1. VARIABLES     : IV (Information Value), monotonía WoE, VIF,
#                      coherencia de signos de los coeficientes.
#   2. ESTABILIDAD   : PSI (población) y CSI (características).
#   3. BENCHMARK     : challenger (regresión logística de referencia)
#                      vs. el modelo -> Gini/KS comparados; y PD
#                      replicada vs. PD entregada si ambas existen.
#
# Complementa a:  calibracion.py (calibración)  y  copiloto_auto.py (replicación).
# Devuelve un dict con semáforo y severidad, listo para el reporte
# de hallazgos del copiloto. Depende de numpy/pandas/scipy/sklearn
# (ya presentes en el stack). Sin dependencias ni servidores nuevos.
#
# Convención de semáforo:  VERDE = ok · AMBAR = revisar · ROJO = hallazgo.
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# UMBRALES (ajustar al manual interno / criterio SBS)
# ------------------------------------------------------------
IV_INUTIL, IV_SOSPECHOSO = 0.02, 0.50     # IV<0.02 sin poder; IV>0.5 posible fuga
VIF_AMBAR, VIF_ROJO       = 5.0, 10.0
PSI_AMBAR, PSI_ROJO       = 0.10, 0.25
GINI_MIN                  = 0.30          # Gini mínimo aceptable (SBS/industria)
BENCH_DELTA_AMBAR         = 0.03          # el challenger supera al modelo por >3pp de Gini
BENCH_DELTA_ROJO          = 0.05          # ... por >5pp -> hallazgo


# ============================================================
# 1. VARIABLES — IV / WoE / VIF / signos
# ============================================================
def _bins_cuantiles(x, q=10):
    b = np.unique(np.nanquantile(x, np.linspace(0, 1, q + 1)))
    if len(b) < 3:
        b = np.array([np.nanmin(x), np.nanmedian(x), np.nanmax(x)])
    return b


def iv_woe(x, y, q=10) -> dict:
    """Information Value + tabla WoE + monotonía frente al target binario."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y); x, y = x[m], y[m]
    if x.size == 0 or y.sum() in (0, y.size):
        return {"IV": np.nan, "fuerza": "SIN DATOS", "monotonia_WoE": None, "tabla": []}
    b = _bins_cuantiles(x, q)
    idx = np.clip(np.digitize(x, b[1:-1]), 0, len(b) - 2)
    tb, tg = max(y.sum(), 1e-9), max((1 - y).sum(), 1e-9)
    iv, woes, tabla = 0.0, [], []
    for k in range(len(b) - 1):
        sel = idx == k; n = int(sel.sum())
        if n == 0:
            continue
        bad = float(y[sel].sum()); good = n - bad
        db, dg = max(bad, 0.5) / tb, max(good, 0.5) / tg
        w = float(np.log(dg / db)); iv += (dg - db) * w; woes.append(w)
        tabla.append({"rango": f"[{b[k]:.4g}, {b[k+1]:.4g}]", "n": n,
                      "tasa_default": round(bad / n, 4), "WoE": round(w, 4)})
    mono = _monotonia(woes)
    return {"IV": round(float(iv), 4), "fuerza": _fuerza_iv(iv),
            "monotonia_WoE": mono, "tabla": tabla}


def _monotonia(woes, umbral=0.8):
    """Monotonía por tendencia (Spearman entre orden de bin y WoE), robusta
    a pequeños quiebres por ruido en las colas. |rho|>=umbral -> monótona."""
    if len(woes) < 3:
        return None
    from scipy import stats as _st
    rho = _st.spearmanr(range(len(woes)), woes).correlation
    return bool(abs(rho) >= umbral) if rho == rho else None


def _fuerza_iv(iv):
    if iv < IV_INUTIL: return "inutil"
    if iv < 0.10:      return "debil"
    if iv < 0.30:      return "medio"
    if iv < IV_SOSPECHOSO: return "fuerte"
    return "sospechoso (posible fuga)"


def vif(df, variables) -> dict:
    """VIF_j = 1/(1-R2_j) regresando cada variable sobre las demás."""
    out = {}
    X = df[variables].apply(pd.to_numeric, errors="coerce").dropna()
    for v in variables:
        otros = [c for c in variables if c != v]
        if not otros or len(X) < 3:
            out[v] = {"VIF": 1.0, "estado": "VERDE"}; continue
        A = np.column_stack([np.ones(len(X)), X[otros].values]); yv = X[v].values
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        ss_res = np.sum((yv - A @ beta) ** 2); ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vv = 1 / (1 - r2) if r2 < 1 else np.inf
        estado = "ROJO" if vv >= VIF_ROJO else ("AMBAR" if vv >= VIF_AMBAR else "VERDE")
        out[v] = {"VIF": round(float(vv), 2), "estado": estado}
    return out


def coherencia_signos(df, variables, target, especificacion) -> dict:
    """Compara el signo del coeficiente con la relación univariada de la
    variable con el default. Signo contradictorio = posible problema."""
    coefs = (especificacion or {}).get("coeficientes", {})
    if not coefs:
        return {"nota": "Sin coeficientes en la especificación; no se evalúan signos."}
    y = pd.to_numeric(df[target], errors="coerce")
    out = {}
    for v in variables:
        if v not in coefs or v not in df.columns:
            continue
        x = pd.to_numeric(df[v], errors="coerce")
        corr = float(x.corr(y))
        signo_coef = np.sign(coefs[v]); signo_uni = np.sign(corr)
        coherente = bool(signo_coef == signo_uni) if not np.isnan(corr) else None
        out[v] = {"coef": coefs[v], "corr_con_default": round(corr, 4),
                  "coherente": coherente,
                  "estado": "VERDE" if coherente else ("AMBAR" if coherente is False else "SIN DATOS")}
    return out


# ============================================================
# 2. ESTABILIDAD — PSI / CSI
# ============================================================
def psi(base, actual, bins=10) -> float:
    base = np.asarray(base, float); actual = np.asarray(actual, float)
    base = base[~np.isnan(base)]; actual = actual[~np.isnan(actual)]
    if base.size == 0 or actual.size == 0:
        return np.nan
    b = np.unique(np.nanquantile(base, np.linspace(0, 1, bins + 1)))
    if len(b) < 3:
        return np.nan
    bi = np.clip(np.digitize(base, b[1:-1]), 0, len(b) - 2)
    ai = np.clip(np.digitize(actual, b[1:-1]), 0, len(b) - 2)
    v = 0.0
    for k in range(len(b) - 1):
        pb = max((bi == k).mean(), 1e-6); pa = max((ai == k).mean(), 1e-6)
        v += (pa - pb) * np.log(pa / pb)
    return round(float(v), 4)


def _estado_psi(v):
    if v is None or np.isnan(v): return "SIN DATOS"
    return "ROJO" if v >= PSI_ROJO else ("AMBAR" if v >= PSI_AMBAR else "VERDE")


def estabilidad(df, score, variables, periodo) -> dict:
    """PSI del score y CSI por variable entre periodo más antiguo y más reciente."""
    if periodo not in df.columns:
        return {"nota": f"Sin columna de periodo '{periodo}'; estabilidad omitida."}
    pers = sorted(df[periodo].dropna().unique())
    if len(pers) < 2:
        return {"nota": "Se requieren >=2 periodos para PSI/CSI."}
    base, act = df[df[periodo] == pers[0]], df[df[periodo] == pers[-1]]
    res = {"periodo_base": pers[0], "periodo_actual": pers[-1]}
    if score in df.columns:
        p = psi(base[score].values, act[score].values)
        res["PSI_score"] = p; res["estado_PSI"] = _estado_psi(p)
    res["CSI"] = {v: {"CSI": (c := psi(base[v].values, act[v].values)), "estado": _estado_psi(c)}
                  for v in variables if v in df.columns}
    return res


# ============================================================
# 3. BENCHMARK — challenger vs modelo  (Gini/KS comparados)
# ============================================================
def _auc_gini_ks(score, y) -> dict:
    score = np.asarray(score, float); y = np.asarray(y, float)
    m = ~np.isnan(score) & ~np.isnan(y); score, y = score[m], y[m]
    pos, neg = score[y == 1], score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return {"AUC": np.nan, "Gini": np.nan, "KS": np.nan}
    orden = score.argsort(); ranks = np.empty(len(score), float)
    ranks[orden] = np.arange(1, len(score) + 1)
    auc = (ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    auc = float(max(auc, 1 - auc))
    grid = np.unique(score)
    ks = float(np.max(np.abs(np.searchsorted(np.sort(pos), grid, "right") / pos.size
                              - np.searchsorted(np.sort(neg), grid, "right") / neg.size)))
    return {"AUC": round(auc, 4), "Gini": round(2 * auc - 1, 4), "KS": round(ks, 4)}


def benchmark(df, variables, target, score="pd", pd_replicada="pd_replicada") -> dict:
    """Ajusta un challenger (regresión logística estándar) sobre las variables
    y compara su Gini con el del modelo. Si el challenger supera al modelo,
    es un hallazgo (el modelo no aprovecha la información disponible).
    Si existe PD replicada y entregada, también las compara."""
    res = {}
    y = pd.to_numeric(df[target], errors="coerce")
    # 3a. Discriminación del modelo
    if score in df.columns:
        res["modelo"] = _auc_gini_ks(df[score].values, y.values)
        res["modelo"]["cumple_gini_min"] = bool(res["modelo"]["Gini"] >= GINI_MIN) \
            if not np.isnan(res["modelo"]["Gini"]) else None
    # 3b. Challenger logístico
    vars_ok = [v for v in variables if v in df.columns]
    if vars_ok:
        try:
            from sklearn.linear_model import LogisticRegression
            X = df[vars_ok].apply(pd.to_numeric, errors="coerce")
            d = pd.concat([X, y.rename("_y")], axis=1).dropna()
            if d["_y"].nunique() == 2 and len(d) > 30:
                lr = LogisticRegression(max_iter=1000)
                lr.fit(d[vars_ok].values, d["_y"].values)
                p_ch = lr.predict_proba(d[vars_ok].values)[:, 1]
                res["challenger"] = _auc_gini_ks(p_ch, d["_y"].values)
                if "modelo" in res and not np.isnan(res["modelo"]["Gini"]):
                    delta = res["challenger"]["Gini"] - res["modelo"]["Gini"]
                    res["delta_gini_challenger_vs_modelo"] = round(float(delta), 4)
                    res["estado_benchmark"] = ("ROJO" if delta > BENCH_DELTA_ROJO
                        else "AMBAR" if delta > BENCH_DELTA_AMBAR else "VERDE")
        except Exception as e:
            res["challenger"] = {"nota": f"No se pudo ajustar el challenger: {e}"}
    # 3c. Replicada vs entregada (Modo B)
    if pd_replicada in df.columns and score in df.columns:
        a = pd.to_numeric(df[pd_replicada], errors="coerce")
        b = pd.to_numeric(df[score], errors="coerce")
        dif = (a - b).abs()
        res["replicacion"] = {"correlacion": round(float(a.corr(b)), 4),
                              "max_abs_dif": round(float(dif.max()), 4),
                              "media_abs_dif": round(float(dif.mean()), 4),
                              "estado": "ROJO" if dif.max() > 0.05 else "VERDE"}
    return res


# ============================================================
# ORQUESTADOR — una sola tool para el agente
# ============================================================
def validar_metodologia(df, target="default", score="pd",
                        variables=None, periodo="periodo",
                        especificacion=None, pd_replicada="pd_replicada") -> dict:
    """Corre los pilares metodológicos (variables, estabilidad, benchmark)
    y devuelve un dict estructurado con 'resumen' tipo semáforo para el reporte."""
    if variables is None:
        variables = list((especificacion or {}).get("coeficientes", {}).keys())
    variables = [v for v in variables if v in df.columns]
    res = {"n": int(len(df)), "variables_analizadas": variables}

    # 1. Variables
    res["variables"] = {
        "poder_predictivo": {v: iv_woe(df[v].values, df[target].values) for v in variables}
                             if target in df.columns else {"nota": f"Sin target '{target}'."},
        "multicolinealidad": vif(df, variables) if len(variables) >= 2 else
                             {"nota": "Se requieren >=2 variables para VIF."},
        "coherencia_signos": coherencia_signos(df, variables, target, especificacion)
                             if target in df.columns else {"nota": "Sin target."},
    }
    # 2. Estabilidad
    res["estabilidad"] = estabilidad(df, score, variables, periodo)
    # 3. Benchmark
    res["benchmark"] = benchmark(df, variables, target, score, pd_replicada)

    res["resumen"] = _resumen(res)
    return res


def _resumen(res) -> list[dict]:
    filas = []
    var = res.get("variables", {})
    pp = var.get("poder_predictivo", {})
    if isinstance(pp, dict):
        inut = [v for v, d in pp.items() if isinstance(d, dict) and d.get("fuerza") == "inutil"]
        nomono = [v for v, d in pp.items() if isinstance(d, dict) and d.get("monotonia_WoE") is False]
        if inut:
            filas.append({"test": "Poder predictivo (IV)", "detalle": f"sin poder: {', '.join(inut)}", "estado": "AMBAR"})
        if nomono:
            filas.append({"test": "Monotonía WoE", "detalle": f"no monótonas: {', '.join(nomono)}", "estado": "AMBAR"})
    mc = var.get("multicolinealidad", {})
    if isinstance(mc, dict):
        peor = [d for d in mc.values() if isinstance(d, dict) and d.get("estado") in ("AMBAR", "ROJO")]
        if peor:
            est = "ROJO" if any(d["estado"] == "ROJO" for d in peor) else "AMBAR"
            filas.append({"test": "Multicolinealidad (VIF)", "detalle": f"VIF máx {max(d['VIF'] for d in peor)}", "estado": est})
    sg = var.get("coherencia_signos", {})
    if isinstance(sg, dict):
        incoh = [v for v, d in sg.items() if isinstance(d, dict) and d.get("coherente") is False]
        if incoh:
            filas.append({"test": "Coherencia de signos", "detalle": f"contradictorios: {', '.join(incoh)}", "estado": "ROJO"})
    est = res.get("estabilidad", {})
    if est.get("estado_PSI") in ("AMBAR", "ROJO"):
        filas.append({"test": "Estabilidad (PSI)", "detalle": f"PSI={est.get('PSI_score')}", "estado": est["estado_PSI"]})
    bm = res.get("benchmark", {})
    if bm.get("modelo", {}).get("cumple_gini_min") is False:
        filas.append({"test": "Gini mínimo", "detalle": f"Gini={bm['modelo']['Gini']} < {GINI_MIN}", "estado": "ROJO"})
    if bm.get("estado_benchmark") in ("AMBAR", "ROJO"):
        filas.append({"test": "Benchmark challenger", "detalle": f"Δ Gini={bm.get('delta_gini_challenger_vs_modelo')}", "estado": bm["estado_benchmark"]})
    if bm.get("replicacion", {}).get("estado") == "ROJO":
        filas.append({"test": "Replicación PD", "detalle": f"máx dif={bm['replicacion']['max_abs_dif']}", "estado": "ROJO"})
    return filas


# ------------------------------------------------------------
# Carga desde Athena (misma convención que calibracion.py)
# ------------------------------------------------------------
def cargar_desde_athena(query, database="validairisk", s3_output=None):
    import awswrangler as wr
    return wr.athena.read_sql_query(query, database=database, s3_output=s3_output)


def validar_metodologia_athena(query_athena: str, **kwargs) -> dict:
    """Wrapper tool: baja la data de Athena y corre el pilar metodológico."""
    return validar_metodologia(cargar_desde_athena(query_athena), **kwargs)
