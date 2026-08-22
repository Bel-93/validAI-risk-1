# ============================================================
# ValidAI Risk — Backend: lectura y preparación de insumos
# Adaptado para trabajar con (nombre, bytes) que envía el frontend
# vía multipart, en lugar de los uploaders de Streamlit.
# ============================================================
import io
import re
import json
import zipfile
import tempfile
import os

import pandas as pd

MAX_CONTEXT_CHARS = 10000  # simplificación MVP: el documento se inyecta al prompt


# ---------- limpieza / anonimización / recorte ----------
def limpiar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.replace("\x00", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def anonimizar_texto_basico(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", texto)
    texto = re.sub(r"\b\d{8,12}\b", "[ID_NUMERICO]", texto)
    texto = re.sub(r"\b9\d{8}\b", "[TELEFONO]", texto)
    texto = re.sub(r"\b\d{13,19}\b", "[NUMERO_LARGO]", texto)
    return texto


def limitar_contexto(texto: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not texto:
        return ""
    texto = limpiar_texto(texto)
    if len(texto) <= max_chars:
        return texto
    return (
        texto[:max_chars]
        + "\n\n[CONTEXTO LIMITADO: el archivo supera el tamaño máximo configurado para el "
        + "MVP. En producción se recomienda usar chunking/RAG para analizar el documento completo.]"
    )


def preparar_contexto(texto: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    texto = limpiar_texto(texto)
    texto = anonimizar_texto_basico(texto)
    return limitar_contexto(texto, max_chars=max_chars)


# ---------- lectores ----------
def leer_documento(nombre: str, contenido: bytes) -> str:
    """Documento metodológico: PDF, DOCX, TXT, MD."""
    if not contenido:
        return ""
    nombre = (nombre or "").lower()
    sufijo = "." + nombre.split(".")[-1] if "." in nombre else ""
    try:
        if sufijo == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(contenido)
                tmp_path = tmp.name
            paginas = PyPDFLoader(tmp_path).load()
            os.unlink(tmp_path)
            return "\n\n".join(p.page_content for p in paginas)
        if sufijo == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(contenido)
                tmp_path = tmp.name
            texto = Docx2txtLoader(tmp_path).load()[0].page_content
            os.unlink(tmp_path)
            return texto
        return contenido.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[ERROR_LECTURA_ARCHIVO] {nombre}: {e}"


def _leer_zip_codigo(contenido: bytes) -> str:
    bloques = []
    try:
        with zipfile.ZipFile(io.BytesIO(contenido), "r") as z:
            for nombre_interno in z.namelist():
                if nombre_interno.endswith((".py", ".sql", ".ipynb")):
                    try:
                        bloques.append(
                            f"--- {nombre_interno} ---\n"
                            + z.read(nombre_interno).decode("utf-8", errors="ignore")
                        )
                    except Exception:
                        continue
    except Exception as e:
        return f"[ERROR_LECTURA_ZIP] {e}"
    return "\n\n".join(bloques)


def leer_codigo(archivos: list) -> str:
    """archivos: lista de (nombre, bytes) de .py/.ipynb/.sql/.txt/.md/.json/.zip."""
    if not archivos:
        return ""
    bloques = []
    for nombre, contenido in archivos:
        n = (nombre or "").lower()
        try:
            if n.endswith(".ipynb"):
                nb = json.loads(contenido.decode("utf-8", errors="ignore"))
                celdas = []
                for celda in nb.get("cells", []):
                    fuente = "".join(celda.get("source", []))
                    etiqueta = "CODE" if celda.get("cell_type") == "code" else "MARKDOWN"
                    celdas.append(f"# [{etiqueta}]\n{fuente}")
                cont = "\n\n".join(celdas)
            elif n.endswith(".zip"):
                cont = _leer_zip_codigo(contenido)
            else:
                cont = contenido.decode("utf-8", errors="ignore")
            bloques.append(f"===== ARCHIVO: {nombre} =====\n{cont}")
        except Exception as e:
            bloques.append(f"===== ARCHIVO: {nombre} =====\n[ERROR_LECTURA] {e}")
    return "\n\n".join(bloques)


def leer_tabular(nombre: str, contenido: bytes):
    """CSV/XLSX/XLS -> DataFrame (o None)."""
    if not contenido:
        return None
    n = (nombre or "").lower()
    try:
        if n.endswith(".csv"):
            return pd.read_csv(io.BytesIO(contenido))
        if n.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(contenido))
    except Exception:
        return None
    return None
