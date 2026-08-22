# ============================================================
# ValidAI Risk — Backend: módulo RAG (Elasticsearch híbrido + HyDE)
# HybridRetriever (BM25 + kNN + RRF) y HyDERetriever completos.
# buscar_evidencia(pregunta) -> (texto_con_fuentes, fuentes[]). RAG_K=10.
# ============================================================

import os
import sys
import time
import logging
from datetime import datetime

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


# RAG_K=10: configuración recomendada por la comparación RAG del notebook
# (HyDE + híbrido, k=10 gana en hit@k / MRR sobre corpus real).
RAG_K = 10
base_retriever = HybridRetriever(k=RAG_K)
hyde_retriever = HyDERetriever(base_retriever, use_hyde=True)
# Alias que usa buscar_evidencia(): antes referenciaba un global 'retriever'
# inexistente y el RAG siempre degradaba a "no disponible".
retriever = hyde_retriever


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




# ============================================================
# API de búsqueda para el backend (reutiliza el retriever de arriba)
# ============================================================
def buscar_evidencia(pregunta: str, k: int = RAG_K,
                     filtro_normativa=None, filtro_tipo=None):
    """Devuelve (texto_formateado, fuentes[]). Degrada si no hay retriever."""
    if 'retriever' not in globals() or retriever is None:
        return "Retriever RAG no disponible (revisa ELASTIC_URL / OPENAI_API_KEY).", []
    try:
        res = retriever.search(pregunta, filtro_normativa, filtro_tipo, k)
    except Exception as e:
        return f"Error RAG: {e}", []
    if not res:
        return "No se encontró evidencia suficiente en la base de conocimiento.", []
    fuentes, lineas = [], []
    for i, r in enumerate(res, 1):
        meta = r.get("metadata", {}) if isinstance(r, dict) else {}
        cont = (r.get("page_content", "") if isinstance(r, dict) else str(r))
        fuentes.append({"n": i, "source": meta.get("source"),
                        "normativa": meta.get("normativa"), "snippet": cont[:300]})
        lineas.append(f"[{i}] {cont[:600]}")
    return "\n\n".join(lineas), fuentes
