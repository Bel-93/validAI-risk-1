# ============================================================
# validairisk-mcp — Servidor MCP con 2 tools:
#   buscar_evidencia_rag  +  calcular_calibracion  +  validar_metodologia
# Version AUTOCONTENIDA para Cloud Run — incluye HybridRetriever
# y HyDERetriever completos (no depende de otras celdas del notebook).
# ============================================================

import os
import sys
import time
import logging
from datetime import datetime

from fastmcp import FastMCP  # paquete standalone, compatible con Cloud Run
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger("validairisk-mcp")


def cargar_credenciales():
    creds = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ELASTIC_URL": os.environ.get("ELASTIC_URL", ""),
        "ELASTIC_API_KEY": os.environ.get("ELASTIC_API_KEY", ""),
    }
    if all(creds.values()):
        return creds
    try:
        with open("api.txt", "r", encoding="utf-8") as f:
            for linea in f:
                if "=" in linea:
                    k, v = linea.strip().split("=", 1)
                    if not creds.get(k.strip()):
                        creds[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return creds


credenciales = cargar_credenciales()
OPENAI_API_KEY = credenciales.get("OPENAI_API_KEY", "")
ELASTIC_URL = credenciales.get("ELASTIC_URL", "")
ELASTIC_API_KEY = credenciales.get("ELASTIC_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

INDEX_NAME = "validairisk_hybrid"

HYDE_SYSTEM_PROMPT = (
    "Eres un experto en validacion de modelos de riesgo crediticio "
    "y normativa bancaria peruana (SBS). Genera un fragmento tecnico de 2-4 "
    "oraciones que responderia directamente la pregunta, como si fuera "
    "extraido de un documento oficial SBS o informe de validacion. Solo el "
    "fragmento, sin introduccion."
)

es = None
if ELASTIC_URL and ELASTIC_API_KEY:
    try:
        es = Elasticsearch(ELASTIC_URL, api_key=ELASTIC_API_KEY)
        es.info()
        logger.info("Elasticsearch conectado")
    except Exception as e:
        logger.warning(f"No se pudo conectar a Elasticsearch: {e}")
        es = None
else:
    logger.warning("ELASTIC_URL / ELASTIC_API_KEY no configurados")

embeddings_model = None
if OPENAI_API_KEY:
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("Embeddings inicializados (text-embedding-3-small)")
    except Exception as e:
        logger.warning(f"No se pudo inicializar embeddings: {e}")


def check_rrf():
    if es is None:
        return False
    try:
        v = tuple(int(x) for x in es.info()["version"]["number"].split(".")[:2])
        return v >= (8, 9)
    except Exception:
        return False


RRF_OK = check_rrf()
logger.info(f"RRF disponible: {RRF_OK}")


class HybridRetriever:
    """Busqueda hibrida: BM25 + kNN coseno + RRF, con fallback automatico."""
    def __init__(self, k=5):
        self.k = k
        self._rrf = RRF_OK

    def search(self, pregunta, filtro_normativa=None, filtro_tipo=None, k=None):
        k = k or self.k
        if es is None or not es.indices.exists(index=INDEX_NAME):
            return []
        try:
            qvec = embeddings_model.embed_query(pregunta) if embeddings_model else None
        except Exception:
            qvec = None

        filtros = []
        if filtro_normativa:
            filtros.append({"term": {"metadata.normativa_sbs": filtro_normativa}})
        if filtro_tipo:
            filtros.append({"term": {"metadata.tipo_documento": filtro_tipo}})

        if self._rrf and qvec:
            return self._rrf_search(pregunta, qvec, k, filtros)
        elif qvec:
            return self._hybrid_manual(pregunta, qvec, k, filtros)
        return self._bm25(pregunta, k, filtros)

    def _rrf_search(self, p, qvec, k, filtros):
        body = {"retriever": {"rrf": {
            "retrievers": [
                {"standard": {"query": {"match": {"page_content": p}}}},
                {"knn": {"field": "embedding", "query_vector": qvec, "num_candidates": k * 4}}
            ],
            "rank_window_size": k * 3, "rank_constant": 60
        }}, "size": k}
        if filtros:
            body["retriever"]["rrf"]["filter"] = {"bool": {"must": filtros}}
        try:
            return self._parse(es.search(index=INDEX_NAME, body=body))
        except Exception:
            return self._hybrid_manual(p, qvec, k, filtros)

    def _hybrid_manual(self, p, qvec, k, filtros):
        fc = {"bool": {"must": filtros}} if filtros else {"match_all": {}}
        body = {"query": {"bool": {"must": {"match": {"page_content": p}}, "filter": fc}},
                "knn": {"field": "embedding", "query_vector": qvec,
                        "num_candidates": k * 4, "k": k, "filter": fc}, "size": k}
        try:
            return self._parse(es.search(index=INDEX_NAME, body=body))
        except Exception:
            return self._bm25(p, k, filtros)

    def _bm25(self, p, k, filtros):
        q = ({"bool": {"must": {"match": {"page_content": p}}, "filter": filtros}}
             if filtros else {"match": {"page_content": p}})
        try:
            return self._parse(es.search(index=INDEX_NAME, body={"query": q, "size": k}))
        except Exception:
            return []

    def _parse(self, resp):
        return [{"page_content": h["_source"].get("page_content", ""),
                 "metadata": h["_source"].get("metadata", {}),
                 "score": round(h.get("_score") or 0, 4)}
                for h in resp["hits"]["hits"]]


class HyDERetriever:
    """
    Hypothetical Document Embeddings.
    Flujo: pregunta -> LLM genera doc hipotetico -> embed(doc) -> kNN+BM25 -> chunks
    """
    def __init__(self, base_retriever, use_hyde=True):
        self.base = base_retriever
        self._cache = {}
        self.llm = (ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
                    if (OPENAI_API_KEY and use_hyde) else None)

    def _doc_hipotetico(self, pregunta):
        if pregunta in self._cache:
            return self._cache[pregunta]
        if not self.llm:
            return pregunta
        try:
            doc = self.llm.invoke([
                SystemMessage(content=HYDE_SYSTEM_PROMPT),
                HumanMessage(content=f"Pregunta: {pregunta}\n\nFragmento hipotetico:")
            ]).content.strip()
            self._cache[pregunta] = doc
            return doc
        except Exception:
            return pregunta

    def search(self, pregunta, filtro_normativa=None, filtro_tipo=None, k=5):
        if not self.llm:
            return self.base.search(pregunta, filtro_normativa, filtro_tipo, k)

        doc_h = self._doc_hipotetico(pregunta)
        try:
            hvec = embeddings_model.embed_query(doc_h) if embeddings_model else None
        except Exception:
            hvec = None
        if hvec is None:
            return self.base.search(pregunta, filtro_normativa, filtro_tipo, k)

        filtros = []
        if filtro_normativa:
            filtros.append({"term": {"metadata.normativa_sbs": filtro_normativa}})
        if filtro_tipo:
            filtros.append({"term": {"metadata.tipo_documento": filtro_tipo}})

        if RRF_OK:
            body = {"retriever": {"rrf": {
                "retrievers": [
                    {"standard": {"query": {"match": {"page_content": pregunta}}}},
                    {"knn": {"field": "embedding", "query_vector": hvec, "num_candidates": k * 4}}
                ],
                "rank_window_size": k * 3, "rank_constant": 60
            }}, "size": k}
            if filtros:
                body["retriever"]["rrf"]["filter"] = {"bool": {"must": filtros}}
        else:
            fc = {"bool": {"must": filtros}} if filtros else {"match_all": {}}
            body = {"query": {"bool": {"must": {"match": {"page_content": pregunta}}, "filter": fc}},
                    "knn": {"field": "embedding", "query_vector": hvec,
                            "num_candidates": k * 4, "k": k, "filter": fc}, "size": k}
        try:
            resp = es.search(index=INDEX_NAME, body=body)
            return [{"page_content": h["_source"].get("page_content", ""),
                     "metadata": h["_source"].get("metadata", {}),
                     "score": round(h.get("_score") or 0, 4),
                     "hyde_doc": doc_h}
                    for h in resp["hits"]["hits"]]
        except Exception:
            return self.base.search(pregunta, filtro_normativa, filtro_tipo, k)


RAG_K = 10  # coherente con la comparacion RAG (HyDE + hibrido, k=10)
base_retriever = HybridRetriever(k=RAG_K)
hyde_retriever = HyDERetriever(base_retriever, use_hyde=True)


def log_rag_trace(pregunta, hyde_doc, modo, resultados, latencia_ms, error=None):
    """
    En Cloud Run el filesystem es efimero, asi que se registra a stdout
    (capturado por Cloud Logging) en vez de SQLite.
    """
    fuentes = ", ".join({r["metadata"].get("source", "?") for r in resultados})
    logger.info(
        f"RAG trace | modo={modo} | chunks={len(resultados)} | "
        f"latencia_ms={latencia_ms} | fuentes=[{fuentes}] | error={error}"
    )


# ============================================================
# CALIBRACION — tests de validacion de modelos de PD
# Umbrales del semaforo (cambiar por los del manual interno)
# ============================================================
ALPHA_AMBAR = 0.05     # 95%
ALPHA_ROJO  = 0.001    # 99.9%


def _test_binomial(n, defaults, pd_est):
    """Una cola (superior): p-value P(X >= defaults | Binom(n, pd))."""
    if n == 0:
        return float("nan")
    return stats.binomtest(defaults, n, pd_est, alternative="greater").pvalue


def _test_jeffreys(n, defaults, pd_est):
    """Posterior Beta(def+0.5, n-def+0.5); P(theta <= pd)."""
    if n == 0:
        return float("nan")
    return float(stats.beta.cdf(pd_est, defaults + 0.5, n - defaults + 0.5))


def _semaforo(p):
    if p != p:
        return "SIN DATOS"
    if p >= ALPHA_AMBAR:
        return "VERDE"
    if p >= ALPHA_ROJO:
        return "AMBAR"
    return "ROJO"


def _hosmer_lemeshow(tabla):
    n = tabla["n"].to_numpy(float)
    obs = tabla["defaults"].to_numpy(float)
    p = tabla["pd_media"].to_numpy(float)
    esp = n * p
    den = np.where(esp * (1 - p) == 0, np.nan, esp * (1 - p))
    hl = np.nansum((obs - esp) ** 2 / den)
    dof = max(len(tabla) - 2, 1)
    return hl, dof, float(stats.chi2.sf(hl, dof))


def _spiegelhalter_z(pd_ind, y_ind):
    num = np.sum((y_ind - pd_ind) * (1 - 2 * pd_ind))
    den = np.sqrt(np.sum((1 - 2 * pd_ind) ** 2 * pd_ind * (1 - pd_ind)))
    if den == 0:
        return float("nan"), float("nan")
    z = num / den
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def _cargar_scores(ruta_datos):
    """Scores a nivel obligor: columnas pd, default (0/1), rating (opcional).
    Para produccion, reemplazar por Athena:
        import awswrangler as wr
        return wr.athena.read_sql_query(query, database="validairisk")
    """
    if ruta_datos.endswith((".xlsx", ".xls")):
        return pd.read_excel(ruta_datos)
    return pd.read_csv(ruta_datos)


mcp = FastMCP("validairisk")


@mcp.tool(
    description=(
        "Busca evidencia documental normativa SBS usando RAG hibrido con HyDE. "
        "Retorna fragmentos con fuente, pagina, normativa SBS, articulo y score."
    )
)
async def buscar_evidencia_rag(
    pregunta: str,
    filtro_normativa: str = None,
    filtro_tipo: str = None,
    usar_hyde: bool = True,
) -> str:
    """Busca evidencia normativa SBS y devuelve fragmentos citables."""
    r_activo, modo = None, "sin_retriever"
    if usar_hyde and hyde_retriever.llm is not None:
        r_activo, modo = hyde_retriever, "HyDE"
    elif base_retriever is not None:
        r_activo, modo = base_retriever, "Hibrido"

    if r_activo is None or es is None:
        return "Retriever no disponible. Verifica ELASTIC_URL, ELASTIC_API_KEY y OPENAI_API_KEY."

    t0 = time.time()
    hyde_doc_log = None
    try:
        resultados = r_activo.search(pregunta, filtro_normativa, filtro_tipo, k=RAG_K)
        if modo == "HyDE" and hasattr(r_activo, "_doc_hipotetico"):
            hyde_doc_log = r_activo._doc_hipotetico(pregunta)
    except Exception as e:
        log_rag_trace(pregunta, None, "error", [], int((time.time() - t0) * 1000), str(e))
        return f"Error RAG: {e}"

    log_rag_trace(pregunta, hyde_doc_log, modo, resultados, int((time.time() - t0) * 1000))

    if not resultados:
        return f"No se encontro evidencia [modo: {modo}]. Considera reformular la pregunta."

    lineas = [f"Evidencia — {len(resultados)} fragmentos [modo: {modo}]:\n"]
    for i, doc in enumerate(resultados, 1):
        meta = doc.get("metadata", {})
        header = f"[{i}] Fuente: {meta.get('source','?')} | Pag: {meta.get('page','—')} | Score: {doc.get('score','—')}"
        if meta.get("normativa_sbs"):
            header += f" | {meta['normativa_sbs']}"
        if meta.get("articulo_sbs"):
            header += f" | {meta['articulo_sbs']}"
        lineas.append(header)
        lineas.append(doc.get("page_content", "")[:900])
        lineas.append("")
    return "\n".join(lineas)


@mcp.tool(
    description=(
        "Evalua la CALIBRACION de un modelo de PD de riesgo de credito. "
        "Lee scores a nivel obligor (pd, default, rating) y devuelve, por "
        "bucket, la tasa de default observada vs PD, test binomial, Jeffreys "
        "y semaforo; y a nivel global Hosmer-Lemeshow y Spiegelhalter Z."
    )
)
async def calcular_calibracion(ruta_datos: str = "scores_modelo.csv") -> str:
    """Informe de calibracion citable para el agente validador."""
    try:
        df = _cargar_scores(ruta_datos)
    except Exception as e:
        return f"No se pudo cargar la data de scores ({ruta_datos}): {e}"

    if "rating" not in df.columns:
        df["rating"] = pd.qcut(df["pd"], 10, labels=False, duplicates="drop")

    filas = []
    for r, g in df.groupby("rating"):
        n = len(g)
        d = int(g["default"].sum())
        pm = float(g["pd"].mean())
        p_bin = _test_binomial(n, d, pm)
        filas.append({
            "rating": r, "n": n, "defaults": d,
            "pd_media": round(pm, 5), "dr_obs": round(d / n, 5) if n else None,
            "p_binomial": round(p_bin, 4) if p_bin == p_bin else None,
            "p_jeffreys": round(_test_jeffreys(n, d, pm), 4),
            "semaforo": _semaforo(p_bin),
        })
    tabla = pd.DataFrame(filas).sort_values("rating")

    hl, dof, hl_p = _hosmer_lemeshow(tabla)
    z, z_p = _spiegelhalter_z(df["pd"].to_numpy(float), df["default"].to_numpy(float))
    pd_media = float(df["pd"].mean())
    dr_glob = float(df["default"].mean())
    conteo = tabla["semaforo"].value_counts().to_dict()

    out = ["INFORME DE CALIBRACION — modelo de PD\n", "Por bucket / grado de rating:"]
    out.append(f"{'rating':>6} {'n':>7} {'def':>6} {'pd_media':>9} {'dr_obs':>9} {'p_binom':>8} {'semaforo':>9}")
    for _, row in tabla.iterrows():
        out.append(f"{row['rating']:>6} {row['n']:>7} {row['defaults']:>6} "
                   f"{row['pd_media']:>9} {row['dr_obs']:>9} {row['p_binomial']:>8} {row['semaforo']:>9}")
    out.append("")
    out.append(f"Hosmer-Lemeshow: HL={hl:.3f} (gl={dof}), p={hl_p:.4f} -> "
               f"{'OK' if hl_p >= ALPHA_AMBAR else 'REVISAR'}")
    out.append(f"Spiegelhalter Z: Z={z:.3f}, p={z_p:.4f} -> "
               f"{'OK' if z_p >= ALPHA_AMBAR else 'REVISAR'}")
    out.append(f"Calibration-in-the-large: PD media={pd_media:.5f} vs DR observada={dr_glob:.5f} "
               f"(dif={dr_glob - pd_media:+.5f})")
    out.append(f"Resumen semaforo: {conteo}")
    return "\n".join(out)



# ============================================================
# TOOL 3 — Validacion metodologica (importa el modulo compartido)
# Se despliega validacion_metodologica.py junto a este servidor (ver celda de deploy).
# ============================================================
try:
    import validacion_metodologica as _vm
except Exception as _e:
    _vm = None
    logger.warning(f"validacion_metodologica no disponible: {_e}")


def _fmt_metodologia_srv(res: dict) -> str:
    L = ["INFORME DE VALIDACION METODOLOGICA", ""]
    var = res.get("variables", {})
    pp = var.get("poder_predictivo", {})
    if isinstance(pp, dict) and pp and "nota" not in pp:
        L.append("Poder predictivo (IV) y monotonia WoE:")
        for v, d in pp.items():
            if isinstance(d, dict):
                L.append(f"  - {v}: IV={d.get('IV')} ({d.get('fuerza')}), monotonica={d.get('monotonia_WoE')}")
    mc = var.get("multicolinealidad", {})
    if isinstance(mc, dict) and mc and "nota" not in mc:
        L.append("Multicolinealidad (VIF):")
        for v, d in mc.items():
            if isinstance(d, dict):
                L.append(f"  - {v}: VIF={d.get('VIF')} [{d.get('estado')}]")
    sg = var.get("coherencia_signos", {})
    if isinstance(sg, dict) and sg and "nota" not in sg:
        L.append("Coherencia de signos:")
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
        L.append(f"Discriminacion modelo: Gini={m.get('Gini')}, KS={m.get('KS')}, cumple_gini_min={m.get('cumple_gini_min')}")
    if bm.get("challenger") and "Gini" in bm.get("challenger", {}):
        L.append(f"Benchmark challenger: Gini={bm['challenger']['Gini']}, "
                 f"delta(chall-modelo)={bm.get('delta_gini_challenger_vs_modelo')} [{bm.get('estado_benchmark')}]")
    if bm.get("replicacion"):
        rp = bm["replicacion"]
        L.append(f"Replicacion PD: corr={rp.get('correlacion')}, max_dif={rp.get('max_abs_dif')} [{rp.get('estado')}]")
    L.append("")
    L.append("Hallazgos (semaforo):")
    for h in res.get("resumen", []):
        L.append(f"  - [{h.get('estado')}] {h.get('test')}: {h.get('detalle')}")
    if not res.get("resumen"):
        L.append("  - Sin hallazgos metodologicos relevantes.")
    L.append("")
    L.append("Reporte preliminar; requiere aprobacion del validador humano.")
    return "\n".join(L)


@mcp.tool(
    description=(
        "Valida la METODOLOGIA de un modelo de PD (rol validador): IV/WoE + monotonia, "
        "VIF, coherencia de signos, estabilidad PSI/CSI y benchmark contra un challenger "
        "logistico; compara PD replicada vs entregada. Devuelve hallazgos con semaforo. "
        "Recibe la ruta/consulta de datos y, opcional, la especificacion del modelo en JSON."
    )
)
async def validar_metodologia(ruta_datos: str = "scores_modelo.csv",
                              especificacion_json: str = "{}") -> str:
    """Informe de validacion metodologica citable para el agente validador."""
    if _vm is None:
        return "Modulo de validacion metodologica no disponible en el servidor."
    try:
        df = _cargar_scores(ruta_datos)
    except Exception as e:
        return f"No se pudo cargar la data ({ruta_datos}): {e}"
    import json as _json
    try:
        espec = _json.loads(especificacion_json) if especificacion_json else {}
    except Exception:
        espec = {}
    try:
        res = _vm.validar_metodologia(df, especificacion=espec or None)
    except Exception as e:
        return f"Error en validacion metodologica: {e}"
    return _fmt_metodologia_srv(res)


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        import asyncio
        port = int(os.getenv("PORT", 8080))
        logger.info(f"Iniciando servidor MCP en 0.0.0.0:{port} (streamable-http)")
        asyncio.run(
            mcp.run_async(transport="streamable-http", host="0.0.0.0", port=port)
        )
