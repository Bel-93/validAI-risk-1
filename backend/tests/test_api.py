# Smoke test del contrato de la API (sin llamar al LLM real).
from fastapi.testclient import TestClient
import main


def test_health():
    c = TestClient(main.app)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_consulta_vacia():
    c = TestClient(main.app)
    r = c.post("/consulta", json={"pregunta": "   "})
    body = r.json()
    assert "session_id" in body and "error" in body


def test_reporte_pdf():
    c = TestClient(main.app)
    r = c.post("/reporte_pdf", json={"texto": "1. Resumen\nHallazgo de prueba.",
                                     "modelo": "Test", "periodo": "2026-08"})
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"  # es un PDF válido


def test_memoria_contrato():
    c = TestClient(main.app)
    r = c.get("/memoria", params={"tabla": "hallazgos"})
    body = r.json()
    assert body["tabla"] == "hallazgos" and "filas" in body
