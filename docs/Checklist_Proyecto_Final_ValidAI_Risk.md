# ValidAI Risk — Checklist del Proyecto Final Integrador

**Fecha de entrega:** miércoles 26 de agosto · **Estado general:** núcleo de IA (LangChain + RAG + MCP) muy avanzado; el mayor riesgo es la **separación frontend/backend** y el **despliegue**.

Leyenda: ✅ hecho · 🟡 parcial · ❌ falta

---

## 1. Mapeo de la rúbrica (100 pts) contra lo que ya tienes

| Criterio (puntaje) | Estado | Qué tienes | Qué falta |
|---|---|---|---|
| **Problema, usuario y valor (8)** | ✅ | Documento de Visión del copiloto + caso PD/SBS bien definido | Resumirlo en README y presentación |
| **Arquitectura end-to-end y separación (12)** | 🟡 | MCP ya está separado; agente, RAG y MCP-client funcionan | **Separar backend del frontend**: hoy todo vive en un solo Streamlit (`app_check.py`) |
| **LangChain: agente, tools, memoria (15)** | ✅ | ReAct con `create_react_agent`, tools, memoria por `thread_id` | Moverlo detrás del backend, no en el frontend |
| **RAG: pipeline, recuperación, fuentes (15)** | ✅ | Elasticsearch híbrido (BM25+kNN+RRF) + HyDE, fuentes y metadata | Documentar ingesta reproducible + **comparar 2 configuraciones** + caso "falta de evidencia" |
| **MCP: servidor, cliente, integración (15)** | ✅ | Servidor FastMCP separado (Cloud Run) con 2 tools; cliente `streamable_http` | Documentar contrato; no duplicar como tool local lo que evalúas por MCP |
| **Backend como microservicio (10)** | ❌ | Lógica existe dentro de Streamlit | **API HTTP (FastAPI): endpoint principal + `/health`, contrato, Dockerfile, logs** |
| **Frontend funcional (8)** | 🟡 | UI Streamlit con estilo | Que **consuma el backend por HTTP**, estados carga/error, fuentes, sesión, URL por env |
| **Despliegue, seguridad, repo Git (10)** | ❌ | Docs y diagramas AWS | **Repo Git público + estructura monorepo + Docker + URLs desplegadas + `.env.example`** |
| **Pruebas, demo, comunicación (7)** | 🟡 | `test_calib.py` (1 prueba) | Prueba unitaria de tool + smoke test del endpoint + evidencia de 7 escenarios + demo |
| **Bonus LangGraph (+10)** | 🟡 | Usa `create_react_agent` (LangGraph) con checkpointer | Demostrar HITL / replay / fork + estado explícito documentado |

---

## 2. Condiciones de completitud (sección 15 — obligatorio para no ser "parcial")

- ❌ Frontend desplegado que consume un backend separado
- ❌ Backend como microservicio con API documentada
- 🟡 LangChain + RAG + MCP integrados en el flujo real (sí, pero acoplados en Streamlit)
- ❌ Repositorio Git accesible y reproducible
- ❌ Demo sobre entorno desplegado

> ⚠️ Una entrega sin repo, sin frontend/backend desplegados por separado, o sin RAG/MCP funcional se evalúa como **implementación parcial**. Este es el foco de los 3 días.

---

## 3. Checklist final de entrega (sección 17)

- ❌ Repositorio Git público con acceso al docente
- ❌ README completo + `.env.example` sin secretos
- ❌ Frontend y backend en carpetas separadas
- 🟡 Servidor MCP separado y documentado (existe; falta doc de contrato)
- ❌ Dockerfile del backend + mecanismo de ejecución del MCP
- ❌ URLs desplegadas y verificadas
- ✅ Agente/workflow LangChain funcional
- ✅ Memoria/contexto conversacional
- 🟡 Pipeline RAG reproducible y fuentes visibles (falta doc de ingesta)
- ✅ MCP con al menos una capacidad (tiene 2)
- 🟡 Pruebas de RAG, MCP, fallback, errores y contexto (falta evidenciarlas)
- 🟡 Pruebas automatizadas mínimas (1 existe; falta smoke test del endpoint)
- ✅ Diagrama de arquitectura (tienes AWS; ajustar al flujo del proyecto)
- ❌ Presentación y demo preparadas

---

## 4. Plan sugerido de 3 días (mínima complejidad)

**Día 1 — Separar backend (lo más crítico).**
Extraer de `app_check.py` la lógica de agente + RAG + cliente MCP a un **backend FastAPI** con dos endpoints: `POST /consulta` (recibe pregunta + session_id, devuelve respuesta + fuentes + trazas) y `GET /health`. Dejar `.env` para claves/URLs. Probar local.

**Día 2 — Frontend + despliegue.**
Adelgazar el Streamlit para que solo llame al backend por HTTP (URL por variable de entorno), con estados de carga/error y panel de fuentes. Escribir Dockerfile de backend, Docker Compose (backend + MCP), y desplegar los tres servicios. Armar el repo monorepo (`frontend/ backend/ mcp-server/ data/ docs/ deploy/ README.md`).

**Día 3 — Pruebas, docs y demo.**
Prueba unitaria de una tool (p. ej. `evaluar_performance` o `calcular_calibracion`) + smoke test de `/health` y `/consulta`. Evidenciar los 7 escenarios (consulta con fuentes, síntesis, falta de evidencia, MCP, error controlado, contexto, despliegue). README + `.env.example` + diagrama. Preparar demo de 10–12 min.

---

## 5. Alcance de validación (rol VALIDADOR, no monitoring)

Se descartó el análisis de error relativo. El copiloto reproduce el proceso de validación con pruebas consistentes alineadas a SBS, en pilares:

- **Pilar 0 — Replicación (Modo B)** (`copiloto_auto.py`): recalcula la PD desde data cruda + especificación; diferencia vs la entregada = hallazgo.
- **Pilar 1 — Metodológica** (`validacion_metodologica.py`, nuevo y verificado): IV/WoE + monotonía, VIF, coherencia de signos de coeficientes.
- **Pilar 2 — Estabilidad** (mismo módulo): PSI (población) y CSI (características).
- **Pilar 3 — Benchmark** (mismo módulo): challenger logístico vs modelo (Gini/KS comparados) + replicada vs entregada.
- **Pilar 4 — Calibración** (`calibracion.py`): binomial, Jeffreys, semáforo, Hosmer-Lemeshow, Spiegelhalter Z, calibration-in-the-large.
- **Transversal — Normativa SBS (RAG)**: cada prueba cita la resolución/artículo que la respalda; cierre con aprobación humana.

`metricas_performance.py` quedó absorbido (la discriminación vive dentro del benchmark). La discriminación NO se usa como métrica suelta de monitoring, sino como prueba de validación comparada.

**Pendiente de cableado (validación):** registrar `validar_metodologia` como tool del agente/MCP, agregar su bloque al reporte de hallazgos, y una prueba unitaria del módulo.
