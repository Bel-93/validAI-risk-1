# Despliegue gratis — Google Cloud Run (los 3 servicios)

Todo desde la terminal, sin construir imágenes a mano (`--source` usa Cloud Build).
Orden importante: **MCP → Backend → Frontend** (cada uno necesita la URL del anterior).

> Requisitos: tener `gcloud` instalado y un proyecto GCP (el mismo que ya usaste para el MCP).
> Free tier: 2M requests/mes y escala a cero, así que un demo no genera costo.

## 0. Preparación (una vez)

```bash
gcloud auth login
gcloud config set project TU_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## 1. Servidor MCP

```bash
cd mcp-server
gcloud run deploy validairisk-mcp \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=sk-...,ELASTIC_URL=https://...,ELASTIC_API_KEY=...
```

Al terminar, copia la **URL** que imprime (ej. `https://validairisk-mcp-xxxx.run.app`).
La URL del endpoint MCP es esa URL + `/mcp/`.

## 2. Backend (FastAPI)

```bash
cd ../backend
gcloud run deploy validairisk-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars LLM_PROVIDER=openai,OPENAI_API_KEY=sk-...,OPENAI_MODEL=gpt-4o-mini,ELASTIC_URL=https://...,ELASTIC_API_KEY=...,VALIDAIRISK_MCP_URL=https://validairisk-mcp-xxxx.run.app/mcp/,TAVILY_API_KEY=tvly-...
```

Copia la **URL** del backend. Verifica salud:

```bash
curl https://validairisk-backend-xxxx.run.app/health
# -> {"status":"ok","servicio":"validai-backend"}
```

## 3. Frontend (Streamlit)

```bash
cd ../frontend
gcloud run deploy validairisk-frontend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars BACKEND_URL=https://validairisk-backend-xxxx.run.app
```

La URL del frontend es la que abres en el navegador para la demo.

## 4. Verificación end-to-end

1. Abre la URL del frontend.
2. En la barra lateral debe decir "Backend: ok".
3. Haz una consulta (ej. *"¿Qué Gini mínimo exige la SBS?"*) → respuesta con evidencia.
4. Los tres servicios aparecen desplegados en la consola de Cloud Run.

## Notas de seguridad

- No subas `.env` ni claves al repo (`.gitignore` ya las excluye).
- Para producción, en vez de `--set-env-vars` usa **Secret Manager**:
  `--set-secrets OPENAI_API_KEY=openai-key:latest` (tras crear el secreto).

## Alternativa para el frontend: Streamlit Community Cloud (gratis)

1. Sube el repo a GitHub.
2. En https://share.streamlit.io conecta el repo.
3. Main file path: `frontend/app.py`.
4. En "Secrets" agrega `BACKEND_URL="https://validairisk-backend-xxxx.run.app"`.
