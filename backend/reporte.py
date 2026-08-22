# ============================================================
# ValidAI Risk — Backend: reporte preliminar a PDF (reportlab)
# ============================================================
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def reporte_a_pdf(texto: str, modelo: str = "", periodo: str = "") -> bytes:
    """Convierte el reporte (texto/markdown) a PDF y devuelve los bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    ss = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=ss["Heading2"], fontSize=13,
                       textColor=colors.HexColor("#1B3A5B"))
    P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9.5, leading=13)
    E = [
        Paragraph("ValidAI Risk — Reporte preliminar de validación", H),
        Paragraph(f"Modelo: {modelo or '-'} &nbsp;·&nbsp; Periodo: {periodo or '-'}", P),
        Spacer(1, 8),
    ]
    for linea in (texto or "").split("\n"):
        s = linea.rstrip()
        if not s.strip():
            E.append(Spacer(1, 4)); continue
        es_titulo = s.lstrip().startswith("#")
        s = s.replace("**", "").replace("###", "").replace("##", "").replace("#", "")
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        E.append(Paragraph(s, H if es_titulo else P))
    E.append(Spacer(1, 10))
    E.append(Paragraph("Reporte preliminar — requiere aprobación del validador humano.", P))
    doc.build(E)
    return buf.getvalue()
