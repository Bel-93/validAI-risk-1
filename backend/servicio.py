# ============================================================
# ValidAI Risk — Backend: orquestación de la revisión completa
# Reproduce, sin Streamlit, el flujo "Ejecutar revisión IA":
# lee insumos -> replica (Modo B) -> calibración + metodología ->
# gráficos -> evidencia RAG -> agente -> reporte preliminar.
# ============================================================
import json
import logging

import pandas as pd

import documentos as D
import graficos as G
import calibracion as C
import validacion_metodologica as VM
import rag
import memoria
import agent as A
from copiloto_auto import replicar_pd_logistica

logger = logging.getLogger("validai.servicio")


# ---------- formateadores de resultados de las tools ----------
def _fmt_calibracion(res: dict) -> str:
    tabla = res.get("tabla_por_bucket")
    L = ["INFORME DE CALIBRACION - modelo de PD", "", "Por bucket / grado:"]
    if isinstance(tabla, pd.DataFrame) and not tabla.empty:
        L.append(tabla.to_string(index=False))
    hl = res.get("hosmer_lemeshow", {})
    sp = res.get("spiegelhalter_z", {})
    cil = res.get("calibration_in_the_large", {})
    L.append("")
    L.append(f"Hosmer-Lemeshow: {hl}")
    L.append(f"Spiegelhalter Z: {sp}")
    L.append(f"Brier score: {res.get('brier_score')}")
    L.append(f"Calibration-in-the-large: {cil}")
    L.append(f"Resumen semáforo: {res.get('resumen_semaforo')}")
    return "\n".join(L)


def _fmt_metodologia(res: dict) -> str:
    L = ["INFORME DE VALIDACION METODOLOGICA", ""]
    var = res.get("variables", {})
    pp = var.get("poder_predictivo", {})
    if isinstance(pp, dict) and pp and "nota" not in pp:
        L.append("Poder predictivo (IV) y monotonía WoE:")
        for v, d in pp.items():
            if isinstance(d, dict):
                L.append(f"  - {v}: IV={d.get('IV')} ({d.get('fuerza')}), monotónica={d.get('monotonia_WoE')}")
    mc = var.get("multicolinealidad", {})
    if isinstance(mc, dict) and mc and "nota" not in mc:
        L.append("Multicolinealidad (VIF):")
        for v, d in mc.items():
            if isinstance(d, dict):
                L.append(f"  - {v}: VIF={d.get('VIF')} [{d.get('estado')}]")
    sg = var.get("coherencia_signos", {})
    if isinstance(sg, dict) and sg and "nota" not in sg:
        L.append("Coherencia de signos de coeficientes:")
        for v, d in sg.items():
            if isinstance(d, dict):
                L.append(f"  - {v}: coef={d.get('coef')}, corr_default={d.get('corr_con_default')}, coherente={d.get('coherente')}")
    est = res.get("estabilidad", {})
    if "PSI_score" in est:
        L.append(f"Estabilidad: PSI score={est.get('PSI_score')} [{est.get('estado_PSI')}] "
                 f"(base {est.get('periodo_base')} vs actual {est.get('periodo_actual')})")
    bm = res.get("benchmark", {})
    if bm.get("modelo"):
        m = bm["modelo"]
        L.append(f"Discriminación modelo: Gini={m.get('Gini')}, KS={m.get('KS')}, cumple_gini_min={m.get('cumple_gini_min')}")
    if bm.get("challenger") and "Gini" in bm.get("challenger", {}):
        L.append(f"Benchmark challenger (logístico): Gini={bm['challenger']['Gini']}, "
                 f"delta(chall-modelo)={bm.get('delta_gini_challenger_vs_modelo')} [{bm.get('estado_benchmark')}]")
    if bm.get("replicacion"):
        rp = bm["replicacion"]
        L.append(f"Replicación PD (Modo B): corr={rp.get('correlacion')}, max_dif={rp.get('max_abs_dif')} [{rp.get('estado')}]")
    L.append("")
    L.append("Hallazgos (semáforo):")
    for h in res.get("resumen", []) or []:
        L.append(f"  - [{h.get('estado')}] {h.get('test')}: {h.get('detalle')}")
    L.append("")
    L.append("Reporte preliminar; requiere aprobación del validador humano.")
    return "\n".join(L)


def _prompt_validacion(modelo, periodo, texto_metodologia, codigo_modelo,
                       resumen_metricas, memoria_historica, observacion, evidencia_rag):
    return f"""
Analiza el siguiente caso de validación de modelo de riesgo.

MODELO:
{modelo}

PERIODO:
{periodo}

OBSERVACIÓN O FOCO DEL VALIDADOR:
{observacion or "No se indicó una observación específica. Realiza revisión general."}

DOCUMENTO METODOLÓGICO:
{texto_metodologia or "No se adjuntó documento metodológico."}

CÓDIGO / NOTEBOOK:
{codigo_modelo or "No se adjuntó código."}

RESULTADOS / MÉTRICAS DISPONIBLES (resultados de las pruebas ya ejecutadas):
{resumen_metricas or "No se adjuntaron métricas tabulares. Este insumo es complementario."}

MEMORIA HISTÓRICA CONSULTADA:
{memoria_historica or "No hay memoria histórica disponible."}

EVIDENCIA DOCUMENTAL (RAG - normativa SBS):
{evidencia_rag or "No se encontró evidencia documental adicional."}

TAREA:
Genera un reporte preliminar de validación con enfoque bancario siguiendo el formato del sistema.
Cruza SIEMPRE la metodología del modelo con la normativa SBS y cita ambas.
Incluye los resultados de las pruebas ya ejecutadas (arriba) e interprétalos.
Deja claro que la decisión final es del validador humano.
"""


def revisar(*, modelo="", periodo="", observacion="",
            doc_metodologia=None, archivos_codigo=None, archivo_datos=None,
            archivo_scores=None, especificacion=None, session_id=None):
    """Cada archivo es (nombre, bytes) o None; archivos_codigo es lista de tuplas.
    Devuelve un dict listo para el frontend (JSON-serializable)."""
    estado = {k: "Pendiente" for k in
              ["insumos", "preparacion", "metricas", "memoria", "rag", "agente", "reporte"]}
    grafico_cal = grafico_psi = None
    calibracion_txt = metodologia_txt = ""
    resumen_metricas = ""

    # 1) Insumos
    texto_metodologia = D.preparar_contexto(D.leer_documento(*doc_metodologia)) if doc_metodologia else ""
    codigo_modelo = D.preparar_contexto(D.leer_codigo(archivos_codigo)) if archivos_codigo else ""
    if not (texto_metodologia or codigo_modelo or observacion or archivo_scores):
        estado["insumos"] = "Error"
        return {"error": "Debes cargar al menos un documento/código, unos scores o una observación.",
                "estado_flujo": estado}
    estado["insumos"] = "OK"

    # 2) Scores -> replicación (Modo B) + calibración + metodología + gráficos
    if archivo_scores:
        df_sc = D.leer_tabular(*archivo_scores)
        if df_sc is not None and not df_sc.empty:
            espec = {}
            if especificacion:
                try:
                    espec = json.loads(especificacion[1].decode("utf-8"))
                except Exception as e:
                    logger.warning(f"Especificación inválida: {e}")

            # df_full conserva las variables originales + pd + default (+ periodo)
            # para que la validación metodológica pueda correr IV/WoE, VIF y coherencia
            # de signos, no solo el benchmark. (replicar_pd_logistica sólo devuelve
            # pd/default/rating, por eso partimos de la data cruda.)
            df_full = df_sc.copy()
            col_default = espec.get("col_default", "default")
            # Modo B: si no hay 'pd' pero hay especificación, se replica y se añade
            if "pd" not in df_full.columns and espec:
                try:
                    df_rep = replicar_pd_logistica(df_sc, espec)
                    df_full["pd"] = df_rep["pd"].to_numpy()
                    df_full["default"] = df_rep["default"].to_numpy()
                except Exception as e:
                    logger.warning(f"No se pudo replicar (Modo B): {e}")
            # Normaliza el nombre del target a 'default'
            if "default" not in df_full.columns and col_default in df_full.columns:
                df_full["default"] = df_full[col_default]

            # Calibración + gráficos
            if {"pd", "default"}.issubset(df_full.columns):
                try:
                    calibracion_txt = _fmt_calibracion(C.evaluar_calibracion(df_full))
                    estado["metricas"] = "OK"
                except Exception as e:
                    logger.warning(f"Calibración falló: {e}")
                grafico_cal = G.grafico_calibracion(df_full)
                grafico_psi = G.grafico_psi(df_full)
            # Metodología (usa las variables de la especificación presentes en df_full)
            try:
                metodologia_txt = _fmt_metodologia(
                    VM.validar_metodologia(df_full, especificacion=espec or None))
                estado["metricas"] = "OK"
            except Exception as e:
                logger.warning(f"Metodología falló: {e}")

    if archivo_datos and not calibracion_txt:
        df_m = D.leer_tabular(*archivo_datos)
        if df_m is not None:
            resumen_metricas = f"Métricas complementarias: {list(df_m.columns)} ({len(df_m)} filas)."
            estado["metricas"] = "OK"

    # Manejo de error controlado: scores subidos pero sin insumos suficientes para
    # ejecutar pruebas (faltan 'pd'/'default' y no hay especificación para Modo B).
    advertencia = ""
    if archivo_scores and not calibracion_txt and not metodologia_txt:
        advertencia = (
            "El archivo de scores no contiene las columnas necesarias ('pd' y 'default') "
            "ni se adjuntó la especificación del modelo (.json) para calcular la PD. "
            "No se ejecutaron las pruebas de calibración ni de validación metodológica. "
            "Sube los scores con la PD ya calculada, o adjunta la especificación del modelo."
        )
        estado["metricas"] = "Error"

    # Error controlado: si el insumo es inválido y no hay nada más que validar
    # (ni documento, ni código, ni observación), se detiene con el mensaje claro
    # y NO se genera un reporte preliminar.
    if advertencia and not (texto_metodologia or codigo_modelo or (observacion or "").strip()):
        estado["preparacion"] = estado["reporte"] = "Error"
        return {"reporte": "", "trazas": [], "advertencia": advertencia,
                "resumen_metricas": "", "calibracion_texto": "", "metodologia_texto": "",
                "grafico_calibracion": None, "grafico_psi": None,
                "evidencia_rag": "", "estado_flujo": estado}

    partes = [p for p in [("ADVERTENCIA: " + advertencia) if advertencia else "",
                          resumen_metricas,
                          ("=== CALIBRACION (PD) ===\n" + calibracion_txt) if calibracion_txt else "",
                          ("=== METODOLOGIA ===\n" + metodologia_txt) if metodologia_txt else ""] if p]
    resumen_metricas = "\n\n".join(partes)
    estado["preparacion"] = "OK"

    # 3) Memoria histórica + evidencia RAG
    keyword = observacion or modelo
    memoria_historica = memoria.consultar_hallazgos_previos(keyword, 5)
    estado["memoria"] = "OK"
    try:
        evidencia_rag, _fuentes = rag.buscar_evidencia(observacion or modelo)
        estado["rag"] = "OK"
    except Exception as e:
        evidencia_rag = ""
        logger.warning(f"RAG falló: {e}")

    # 4) Agente -> reporte
    prompt = _prompt_validacion(modelo, periodo, texto_metodologia, codigo_modelo,
                                resumen_metricas, memoria_historica, observacion, evidencia_rag)
    try:
        reporte, trazas = A.responder(prompt, session_id or "revision", modo="validacion")
        estado["agente"] = estado["reporte"] = "OK"
    except Exception as e:
        logger.exception("Agente falló")
        reporte, trazas = f"No se pudo generar el reporte: {e}", []
        estado["agente"] = estado["reporte"] = "Error"

    memoria.guardar_ejecucion(modelo, periodo, "OK" if estado["reporte"] == "OK" else "ERROR",
                              "Revisión ejecutada desde el frontend.")

    return {
        "reporte": reporte,
        "trazas": trazas,
        "advertencia": advertencia,
        "resumen_metricas": resumen_metricas,
        "calibracion_texto": calibracion_txt,
        "metodologia_texto": metodologia_txt,
        "grafico_calibracion": grafico_cal,   # base64 PNG o None
        "grafico_psi": grafico_psi,           # base64 PNG o None
        "evidencia_rag": evidencia_rag,
        "estado_flujo": estado,
    }
