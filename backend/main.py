# ============================================================
# ValidAI Risk — Backend API (FastAPI, microservicio)
# El frontend NO ejecuta agente/RAG/MCP: todo pasa por estos endpoints.
#   GET  /health           estado del servicio
#   GET  /estado_mcp       conexión a servidores MCP (propio + Tavily)
#   POST /consulta         chat con el agente (memoria por session_id)
#   POST /revisar          revisión completa (multipart: docs, código, scores)
#   POST /reporte_pdf      texto del reporte -> PDF
#   POST /transcribir      audio -> texto (Whisper)
#   POST /hallazgo         guarda un hallazgo validado (HITL) en memoria
#   GET  /memoria          lee la tabla de memoria (hallazgos/ejecuciones)
# ============================================================
import io
import os
import uuid
import base64
import logging
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import agent as A
import servicio as S
import reporte as R
import memoria as M

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
app = FastAPI(title="ValidAI Risk — Backend", version="2.0")


class ConsultaIn(BaseModel):
    pregunta: str
    session_id: Optional[str] = None


class ReportePDFIn(BaseModel):
    texto: str
    modelo: Optional[str] = ""
    periodo: Optional[str] = ""


class HallazgoIn(BaseModel):
    modelo: str = ""
    periodo: str = ""
    categoria: str = ""
    hallazgo: str = ""
    severidad: str = ""
    impacto: str = ""
    recomendacion: str = ""
    decision_humana: str = ""
    comentario: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "servicio": "validai-backend"}


@app.get("/estado_mcp")
def estado_mcp():
    try:
        return A.estado_mcp()
    except Exception as e:
        logging.exception("estado_mcp")
        return {"error": str(e)}


@app.post("/consulta")
def consulta(inp: ConsultaIn):
    sid = inp.session_id or str(uuid.uuid4())
    if not inp.pregunta or not inp.pregunta.strip():
        return {"session_id": sid, "error": "La consulta está vacía."}
    try:
        respuesta, trazas = A.responder(inp.pregunta, sid)
        return {"session_id": sid, "respuesta": respuesta, "trazas": trazas}
    except Exception as e:
        logging.exception("Error procesando la consulta")
        return {"session_id": sid, "error": f"Ocurrió un error: {e}"}


async def _tupla(f: Optional[UploadFile]):
    if f is None:
        return None
    return (f.filename, await f.read())


@app.post("/revisar")
async def revisar(
    modelo: str = Form(""),
    periodo: str = Form(""),
    observacion: str = Form(""),
    session_id: str = Form(""),
    doc_metodologia: Optional[UploadFile] = File(None),
    archivo_datos: Optional[UploadFile] = File(None),
    archivo_scores: Optional[UploadFile] = File(None),
    especificacion: Optional[UploadFile] = File(None),
    archivos_codigo: Optional[List[UploadFile]] = File(None),
):
    try:
        codigo = [(c.filename, await c.read()) for c in (archivos_codigo or [])]
        res = S.revisar(
            modelo=modelo, periodo=periodo, observacion=observacion,
            doc_metodologia=await _tupla(doc_metodologia),
            archivos_codigo=codigo or None,
            archivo_datos=await _tupla(archivo_datos),
            archivo_scores=await _tupla(archivo_scores),
            especificacion=await _tupla(especificacion),
            session_id=session_id or None,
        )
        return res
    except Exception as e:
        logging.exception("Error en /revisar")
        return {"error": f"Ocurrió un error en la revisión: {e}"}


@app.post("/reporte_pdf")
def reporte_pdf(inp: ReportePDFIn):
    data = R.reporte_a_pdf(inp.texto, inp.modelo or "", inp.periodo or "")
    return StreamingResponse(
        io.BytesIO(data), media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="reporte_validacion.pdf"'},
    )


@app.post("/transcribir")
async def transcribir(audio: UploadFile = File(...)):
    """Transcribe audio a texto con Whisper (OpenAI)."""
    try:
        from openai import OpenAI
        oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        data = await audio.read()
        tr = oa.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "dictado.wav", data, "audio/wav"),
        )
        return {"texto": tr.text}
    except Exception as e:
        logging.exception("transcribir")
        return {"error": f"No se pudo transcribir: {e}"}


@app.post("/hallazgo")
def hallazgo(inp: HallazgoIn):
    if not inp.hallazgo.strip():
        return {"error": "El hallazgo está vacío."}
    try:
        M.guardar_hallazgo_validado(
            inp.modelo, inp.periodo, inp.categoria, inp.hallazgo, inp.severidad,
            inp.impacto, inp.recomendacion, inp.decision_humana, inp.comentario)
        return {"status": "ok", "guardado": True}
    except Exception as e:
        logging.exception("hallazgo")
        return {"error": str(e)}


@app.get("/memoria")
def leer_memoria(tabla: str = "hallazgos"):
    df = M.obtener_tabla(tabla)
    return {"tabla": tabla, "filas": df.to_dict(orient="records"),
            "columnas": list(df.columns)}
