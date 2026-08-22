# ============================================================
# ValidAI Risk — Backend: gráficos de validación (calibración y PSI)
# Devuelven PNG en base64 para que el frontend los muestre sin
# recalcular nada (toda la lógica vive en el backend).
# ============================================================
import base64
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_a_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grafico_calibracion(df: pd.DataFrame):
    """Barras PD media vs tasa de default observada por grado. -> base64 o None."""
    if not {"pd", "default"}.issubset(df.columns):
        return None
    d = df.copy()
    d["pd"] = pd.to_numeric(d["pd"], errors="coerce")
    d["default"] = pd.to_numeric(d["default"], errors="coerce")
    d = d.dropna(subset=["pd", "default"])
    if d.empty:
        return None
    if "rating" not in d.columns:
        d["rating"] = pd.qcut(d["pd"], 10, labels=False, duplicates="drop")
    g = d.groupby("rating").agg(pd_media=("pd", "mean"), dr=("default", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w / 2, g["pd_media"], w, label="PD media", color="#1565C0")
    ax.bar(x + w / 2, g["dr"], w, label="DR observada", color="#C62828")
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in g["rating"]])
    ax.set_title("Calibración por grado: PD media vs tasa de default observada")
    ax.set_ylabel("Probabilidad"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    return _fig_a_base64(fig)


def grafico_psi(df: pd.DataFrame):
    """Estabilidad PSI del score entre el primer y último periodo. -> base64 o None."""
    if "periodo" not in df.columns or "pd" not in df.columns:
        return None
    pers = sorted(df["periodo"].dropna().unique())
    if len(pers) < 2:
        return None
    base = pd.to_numeric(df[df.periodo == pers[0]]["pd"], errors="coerce").dropna().values
    act = pd.to_numeric(df[df.periodo == pers[-1]]["pd"], errors="coerce").dropna().values
    if base.size == 0 or act.size == 0:
        return None
    b = np.unique(np.nanquantile(base, np.linspace(0, 1, 11)))

    def _p(a):
        ai = np.clip(np.digitize(a, b[1:-1]), 0, len(b) - 2)
        return np.clip(np.array([(ai == k).mean() for k in range(len(b) - 1)]), 1e-6, None)

    pb, pa = _p(base), _p(act)
    val = float(((pa - pb) * np.log(pa / pb)).sum())
    est = "ROJO" if val >= 0.25 else ("AMBAR" if val >= 0.10 else "VERDE")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(base, bins=b, alpha=0.6, label=f"Base ({pers[0]})", color="#1565C0", density=True)
    ax.hist(act, bins=b, alpha=0.6, label=f"Actual ({pers[-1]})", color="#EF6C00", density=True)
    ax.set_title(f"Estabilidad PSI del score (PSI={val:.3f} · {est})")
    ax.set_xlabel("PD"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    return _fig_a_base64(fig)
