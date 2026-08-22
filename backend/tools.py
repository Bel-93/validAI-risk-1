# ============================================================
# ValidAI Risk — Tools del agente (backend)
# Cada tool devuelve texto/JSON citable para el agente validador.
# ============================================================
import json
import pandas as pd
from langchain_core.tools import tool

import rag
import calibracion
import validacion_metodologica as vm
from copiloto_auto import replicar_pd_logistica  # noqa: F401 (disponible para Modo B)


def _df(df_json: str) -> pd.DataFrame:
    try:
        data = json.loads(df_json)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


@tool
def buscar_evidencia_rag(pregunta: str) -> str:
    """Busca evidencia normativa (SBS) en la base de conocimiento propia (RAG).
    Úsala para sustentar criterios de validación con la resolución/artículo."""
    texto, _ = rag.buscar_evidencia(pregunta)
    return texto


@tool
def calcular_calibracion(df_json: str) -> str:
    """Evalúa la CALIBRACIÓN de un modelo de PD. Recibe scores a nivel obligor
    en JSON (columnas: pd, default, rating opcional). Devuelve por bucket la
    tasa observada vs PD, binomial, Jeffreys, semáforo, y global HL / Spiegelhalter."""
    df = _df(df_json)
    if df.empty or not {"pd", "default"}.issubset(df.columns):
        return "Faltan columnas: se requiere 'pd' y 'default' (rating opcional)."
    res = calibracion.evaluar_calibracion(df)
    return json.dumps(res, ensure_ascii=False, default=str, indent=2)


@tool
def validar_metodologia_tool(df_json: str, especificacion_json: str = "{}") -> str:
    """Valida la METODOLOGÍA de un modelo de PD (rol validador, no monitoring):
    IV/WoE + monotonía, VIF, coherencia de signos, estabilidad PSI/CSI y
    benchmark contra un challenger logístico; compara PD replicada vs entregada.
    Recibe data en JSON (variables + 'default'; opcional 'pd','periodo','pd_replicada')
    y, opcional, la especificación con 'coeficientes'."""
    df = _df(df_json)
    if df.empty:
        return "No se recibió data válida para la validación metodológica."
    try:
        espec = json.loads(especificacion_json) if especificacion_json else {}
    except Exception:
        espec = {}
    res = vm.validar_metodologia(df, especificacion=espec or None)
    return json.dumps(res, ensure_ascii=False, default=str, indent=2)


TOOLS_LOCALES = [buscar_evidencia_rag, calcular_calibracion, validar_metodologia_tool]
