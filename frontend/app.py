# ============================================================
# ValidAI Risk — Frontend (Streamlit) — consume el backend por HTTP.
# NO ejecuta agente/RAG/MCP: solo llama a la API del backend.
# ============================================================
import os
import io
import uuid
import base64

import requests
import pandas as pd
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

st.set_page_config(page_title="ValidAI Risk — Copiloto de Validación",
                   page_icon="🛡️", layout="wide")

# ---------- estilo ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.vr-badge {display:inline-block;background:#04120c;color:#00ff9c;border:1px solid #00ff9c;
  padding:3px 12px;border-radius:6px;font-size:.68rem;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;box-shadow:0 0 8px rgba(0,255,156,.35);}
.vr-hero {border:1px solid #00ff9c;border-radius:14px;padding:20px 26px;background:#05090a;
  box-shadow:0 0 18px rgba(0,255,156,.18), inset 0 0 22px rgba(0,255,156,.05);
  display:flex;align-items:center;gap:24px;flex-wrap:wrap;}
.vr-hero-txt {flex:1;min-width:280px;}
.vr-brain {flex:0 0 auto;}
.vr-hero-txt h1 {margin:.15rem 0 0;color:#eafff6;font-size:2.5rem;font-weight:800;letter-spacing:.05em;
  text-shadow:0 0 12px rgba(0,255,156,.55);line-height:1.05;}
.vr-hero-txt .sub {color:#00ff9c;font-size:1.05rem;font-weight:600;margin:.2rem 0 .55rem;letter-spacing:.03em;
  text-shadow:0 0 8px rgba(0,255,156,.4);}
.vr-hero-txt p {color:#9fb4ad;font-size:.9rem;margin:0;max-width:760px;}
.vr-card {border:1px solid #00ff9c;border-radius:12px;padding:16px;background:#05090a;height:100%;min-height:200px;
  box-shadow:0 0 12px rgba(0,255,156,.12), inset 0 0 16px rgba(0,255,156,.04);transition:box-shadow .25s;}
.vr-card:hover {box-shadow:0 0 20px rgba(0,255,156,.4);}
.vr-card h4 {margin:.3rem 0;color:#00ff9c;font-size:1rem;letter-spacing:.03em;text-shadow:0 0 6px rgba(0,255,156,.4);}
.vr-card p {color:#9fb4ad;font-size:.85rem;margin:0;}
.vr-brain svg {filter:drop-shadow(0 0 6px rgba(0,255,156,.6));}
.vr-brain .ln {animation:vrpulse 2.4s ease-in-out infinite;}
.vr-brain .ln.d1{animation-delay:.4s;} .vr-brain .ln.d2{animation-delay:.9s;}
.vr-brain .ln.d3{animation-delay:1.3s;} .vr-brain .ln.d4{animation-delay:1.7s;}
.vr-brain .nd {animation:vrnode 1.9s ease-in-out infinite;}
.vr-brain .nd.n2{animation-delay:.6s;} .vr-brain .nd.n3{animation-delay:1.1s;} .vr-brain .nd.n4{animation-delay:1.5s;}
@keyframes vrpulse {0%,100%{opacity:.2;} 50%{opacity:1;}}
@keyframes vrnode {0%,100%{opacity:.4;} 50%{opacity:1;}}
</style>
""", unsafe_allow_html=True)


# ---------- helpers HTTP ----------
def api_get(path, **kw):
    return requests.get(f"{BACKEND_URL}{path}", timeout=TIMEOUT, **kw)


def api_post(path, **kw):
    return requests.post(f"{BACKEND_URL}{path}", timeout=TIMEOUT, **kw)


def mostrar_grafico_b64(b64, caption):
    if b64:
        st.markdown(f"**{caption}**")
        st.image(base64.b64decode(b64))


def descargar_pdf(texto, modelo, periodo, key):
    try:
        r = api_post("/reporte_pdf", json={"texto": texto, "modelo": modelo, "periodo": periodo})
        if r.ok:
            st.download_button("⬇️ Descargar reporte (PDF)", data=r.content,
                               file_name="reporte_validacion.pdf",
                               mime="application/pdf", key=key)
    except Exception as e:
        st.caption(f"No se pudo generar el PDF: {e}")


# ---------- estado de sesión ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "historial" not in st.session_state:
    st.session_state.historial = []
if "revision" not in st.session_state:
    st.session_state.revision = None

st.markdown(
    '<div class="vr-hero">'
    '<div class="vr-brain"><svg viewBox="0 0 120 108" width="104" height="94" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="#00ff9c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M60 18 C46 8 26 12 22 28 C10 30 8 46 18 53 C9 61 15 78 30 77 C33 92 52 96 60 86 C68 96 87 92 90 77 C105 78 111 61 102 53 C112 46 110 30 98 28 C94 12 74 8 60 18 Z"/><line x1="60" y1="18" x2="60" y2="86" stroke-opacity=".55"/><path class="ln" d="M60 34 C50 34 46 42 52 48 C46 54 50 62 60 60" stroke-width="1.6"/><path class="ln d1" d="M60 40 C70 40 74 48 68 54 C74 60 70 68 60 66" stroke-width="1.6"/><path class="ln d2" d="M30 44 C38 42 40 50 34 54" stroke-width="1.4"/><path class="ln d3" d="M90 44 C82 42 80 50 86 54" stroke-width="1.4"/><path class="ln d4" d="M42 70 C48 66 54 70 52 76" stroke-width="1.4"/><circle class="nd" cx="52" cy="48" r="2.6" fill="#00ff9c" stroke="none"/><circle class="nd n2" cx="68" cy="54" r="2.6" fill="#00ff9c" stroke="none"/><circle class="nd n3" cx="34" cy="54" r="2.2" fill="#00ff9c" stroke="none"/><circle class="nd n4" cx="86" cy="54" r="2.2" fill="#00ff9c" stroke="none"/><circle class="nd n2" cx="60" cy="60" r="2.4" fill="#00ff9c" stroke="none"/></svg></div>'
    '<div class="vr-hero-txt">'
    '<h1>ValidAI Risk</h1>'
    '<div class="sub">Copiloto de Validación de Modelos de Riesgo</div>'
    '<p>Validación de modelos de probabilidad de incumplimiento, alineada a normativa SBS. '
    'La decisión final es del validador humano.</p>'
    '</div></div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Panel de control")
    try:
        h = api_get("/health").json()
        st.success(f"Backend: {h.get('status', '?')}")
    except Exception:
        st.error("Backend no disponible")
        st.caption(f"Esperado en: {BACKEND_URL}")

    st.divider()
    st.caption("Estado de conexión MCP")
    try:
        e = api_get("/estado_mcp").json()
        def _b(ok): return "🟢" if ok else "⚪"
        st.markdown(f"{_b(e.get('rag'))} RAG normativo (SBS)")
        st.markdown(f"{_b(e.get('calibracion'))} Calibración (MCP)")
        st.markdown(f"{_b(e.get('metodologia'))} Metodología (MCP)")
        st.markdown(f"{_b(e.get('web_search'))} Búsqueda web (Tavily)")
    except Exception:
        st.caption("No se pudo leer el estado MCP.")

    st.divider()
    modelo = st.text_input("Nombre del modelo", value="Modelo Riesgo Crediticio")
    periodo = st.text_input("Periodo", value=pd.Timestamp.now().strftime("%Y-%m"))

    st.divider()
    st.subheader("Estado del flujo")
    estado = (st.session_state.revision or {}).get("estado_flujo", {})
    _ic = {"OK": "✓", "Error": "!", "En proceso": "…"}
    for k in ["insumos", "preparacion", "metricas", "memoria", "rag", "agente", "reporte"]:
        v = estado.get(k, "Pendiente")
        st.write(f"{_ic.get(v, '○')} {k.capitalize()}: `{v}`")

    st.divider()
    st.caption(f"Sesión: `{st.session_state.session_id[:8]}…`")
    if st.button("Nueva sesión"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.historial = []
        st.session_state.revision = None
        st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧭 Cómo funciona", "🧪 Revisión IA", "💬 Consulta",
    "👤 Validación humana", "🗃️ Memoria",
])

# ===================== TAB 1: cómo funciona =====================
with tab1:
    st.subheader("Cómo funciona ValidAI Risk")
    cols = st.columns(4)
    tarjetas = [
        ("Insumos", "Documento metodológico, código/notebook/SQL/ZIP, scores y especificación del modelo."),
        ("Backend (API)", "El frontend llama al backend; ahí viven el agente LangChain, el RAG y el cliente MCP."),
        ("Pruebas de validación", "Replicación (Modo B), metodológica (IV/WoE, VIF, signos, PSI), calibración y benchmark."),
        ("Reporte + HITL", "Reporte preliminar con evidencia y cita SBS; el validador humano aprueba/ajusta/descarta."),
    ]
    for c, (t, p) in zip(cols, tarjetas):
        c.markdown(f'<div class="vr-card"><h4>{t}</h4><p>{p}</p></div>', unsafe_allow_html=True)


# ===================== TAB 2: revisión IA =====================
with tab2:
    st.subheader("Carga de insumos y revisión")
    c1, c2, c3 = st.columns(3)
    with c1:
        f_metodo = st.file_uploader("Documento metodológico", type=["pdf", "docx", "txt", "md"])
    with c2:
        f_codigo = st.file_uploader("Código / Notebook / SQL / ZIP",
                                    type=["py", "ipynb", "sql", "txt", "md", "json", "zip"],
                                    accept_multiple_files=True)
    with c3:
        f_datos = st.file_uploader("Métricas complementarias", type=["csv", "xlsx", "xls"])

    f_scores = st.file_uploader(
        "Data para calibración: scores (pd, default) o data cruda (variables + flag_default)",
        type=["csv", "xlsx", "xls"])
    f_espec = st.file_uploader(
        "Especificación del modelo (JSON) — para replicar la PD (Modo B)", type=["json"])

    # Dictado por voz (Whisper en el backend)
    audio = st.audio_input("🎤 Dictar la observación (opcional)")
    if audio is not None and st.button("Transcribir dictado"):
        try:
            r = api_post("/transcribir",
                         files={"audio": ("dictado.wav", audio.getvalue(), "audio/wav")})
            data = r.json()
            if data.get("texto"):
                st.session_state["observacion_texto"] = data["texto"]
                st.success("Dictado transcrito. Puedes editarlo antes de ejecutar.")
                st.rerun()
            else:
                st.error(data.get("error", "No se pudo transcribir."))
        except Exception as e:
            st.error(f"No se pudo transcribir: {e}")

    observacion = st.text_area(
        "Observación metodológica o foco de revisión",
        key="observacion_texto", height=120,
        placeholder="Ej.: revisar si la población del código coincide con la metodología "
                    "y si la caída del Gini implica degradación.")

    if st.button("Ejecutar revisión IA", type="primary"):
        files = []
        if f_metodo:
            files.append(("doc_metodologia", (f_metodo.name, f_metodo.getvalue())))
        if f_datos:
            files.append(("archivo_datos", (f_datos.name, f_datos.getvalue())))
        if f_scores:
            files.append(("archivo_scores", (f_scores.name, f_scores.getvalue())))
        if f_espec:
            files.append(("especificacion", (f_espec.name, f_espec.getvalue())))
        for c in (f_codigo or []):
            files.append(("archivos_codigo", (c.name, c.getvalue())))
        data = {"modelo": modelo, "periodo": periodo, "observacion": observacion or "",
                "session_id": st.session_state.session_id}
        with st.spinner("Ejecutando pruebas y generando el reporte preliminar…"):
            try:
                r = api_post("/revisar", data=data, files=files or None)
                res = r.json()
            except Exception as e:
                res = {"error": f"Error de conexión con el backend: {e}"}
        if res.get("error"):
            st.error(res["error"])
        st.session_state.revision = res
        st.rerun()

    res = st.session_state.revision
    if res and not res.get("error"):
        if res.get("advertencia"):
            st.warning(res["advertencia"])
        if res.get("grafico_calibracion") or res.get("grafico_psi"):
            gc1, gc2 = st.columns(2)
            with gc1:
                mostrar_grafico_b64(res.get("grafico_calibracion"), "Calibración por grado")
            with gc2:
                mostrar_grafico_b64(res.get("grafico_psi"), "Estabilidad PSI del score")
        rep = res.get("reporte", "")
        if rep:
            st.success("Reporte preliminar generado.")
            st.markdown("## Reporte preliminar")
            st.markdown(rep)
            descargar_pdf(rep, modelo, periodo, key="pdf_tab2")
        with st.expander("Resultados de las pruebas (texto)"):
            st.text(res.get("resumen_metricas") or "Sin métricas.")
        with st.expander("Evidencia recuperada por RAG"):
            st.text(res.get("evidencia_rag") or "Sin evidencia.")
        trazas = res.get("trazas") or []
        if trazas:
            with st.expander("Trazabilidad (tools usadas)"):
                for t in trazas:
                    st.markdown(f"**{t.get('tool', 'tool')}**")
                    st.code(t.get("contenido", ""))

# ===================== TAB 3: consulta (chat) =====================
with tab3:
    st.subheader("Consulta al copiloto")
    # Todos los mensajes se renderizan arriba; el cuadro de escritura queda abajo.
    for m in st.session_state.historial:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("trazas"):
                with st.expander("Evidencia y trazabilidad"):
                    for t in m["trazas"]:
                        st.markdown(f"**{t.get('tool', 'tool')}**")
                        st.code(t.get("contenido", ""))
    pregunta = st.chat_input("Consulta metodológica, de calibración o normativa SBS…")
    if pregunta:
        st.session_state.historial.append({"role": "user", "content": pregunta})
        with st.spinner("Analizando…"):
            try:
                r = api_post("/consulta", json={"pregunta": pregunta,
                                                "session_id": st.session_state.session_id})
                data = r.json()
            except Exception as e:
                data = {"error": f"Error de conexión: {e}"}
        if data.get("error"):
            st.session_state.historial.append(
                {"role": "assistant", "content": f"⚠️ {data['error']}"})
        else:
            st.session_state.session_id = data.get("session_id", st.session_state.session_id)
            st.session_state.historial.append(
                {"role": "assistant", "content": data.get("respuesta", ""),
                 "trazas": data.get("trazas")})
        st.rerun()

# ===================== TAB 4: validación humana =====================
with tab4:
    st.subheader("Validación humana y guardado en memoria")
    rep = (st.session_state.revision or {}).get("reporte", "")
    if not rep:
        st.warning("Primero ejecuta una revisión en la pestaña Revisión IA.")
    else:
        st.markdown("### Reporte preliminar")
        st.markdown(rep)
        descargar_pdf(rep, modelo, periodo, key="pdf_tab4")
        st.divider()
        st.markdown("### Registrar feedback validado")
        categoria = st.selectbox("Categoría del hallazgo",
            ["Metodología", "Código / implementación", "Población objetivo",
             "Métricas / resultados", "Trazabilidad", "Benchmark", "Documentación", "Otro"])
        hallazgo = st.text_area("Hallazgo validado por el humano", height=120)
        ca, cb = st.columns(2)
        severidad = ca.selectbox("Severidad", ["Baja", "Media", "Alta"])
        impacto = cb.selectbox("Impacto", ["Sin impacto material", "Impacto documental",
            "Impacto metodológico", "Impacto en resultados", "Impacto por confirmar"])
        recomendacion = st.text_area("Recomendación", height=90)
        decision = st.selectbox("Decisión humana", ["Aceptar", "Ajustar", "Descartar"])
        comentario = st.text_area("Comentario del validador", height=90)
        confirmar = st.checkbox("Confirmo que este feedback fue revisado por un validador humano.")
        if st.button("Guardar feedback validado", type="primary"):
            if not confirmar:
                st.error("Debes confirmar la revisión humana antes de guardar.")
            elif not hallazgo.strip():
                st.error("Debes ingresar un hallazgo validado.")
            else:
                try:
                    r = api_post("/hallazgo", json={
                        "modelo": modelo, "periodo": periodo, "categoria": categoria,
                        "hallazgo": hallazgo, "severidad": severidad, "impacto": impacto,
                        "recomendacion": recomendacion, "decision_humana": decision,
                        "comentario": comentario})
                    if r.json().get("guardado"):
                        st.success("Feedback validado guardado en la memoria del asistente.")
                    else:
                        st.error(r.json().get("error", "No se pudo guardar."))
                except Exception as e:
                    st.error(f"Error: {e}")

# ===================== TAB 5: memoria =====================
with tab5:
    st.subheader("Memoria del asistente")
    tabla = st.selectbox("Selecciona tabla", ["hallazgos", "ejecuciones"])
    if st.button("Actualizar tabla"):
        try:
            data = api_get("/memoria", params={"tabla": tabla}).json()
            df = pd.DataFrame(data.get("filas", []))
            if df.empty:
                st.caption("Tabla vacía.")
            else:
                st.dataframe(df, use_container_width=True)
                st.download_button("Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{tabla}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"No se pudo leer la tabla: {e}")
