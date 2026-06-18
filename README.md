# ValidAI Risk — Copiloto Inteligente para Validación de Modelos de Riesgo

Copiloto con IA generativa que asiste al validador independiente de modelos de riesgo crediticio en banca peruana, acelerando la revisión preliminar, estructurando hallazgos y sustentando cada observación con evidencia documental real de la normativa SBS.

> El agente no reemplaza al validador: actúa como asistente para acelerar la revisión preliminar, estructurar hallazgos y mejorar la trazabilidad institucional.

## Qué hace

1. El validador sube los insumos de un caso: documento metodológico, código del modelo, métricas de monitoreo.
2. Un agente (ReAct, LangGraph) analiza esos insumos con un set de herramientas especializadas.
3. Cuando necesita contrastar una métrica o un criterio contra la normativa SBS, el agente recupera evidencia real desde una base de conocimiento indexada — no responde con criterio general del modelo de lenguaje.
4. Genera un reporte preliminar con hallazgos, severidad, impacto y benchmark, citando documento, página, resolución y artículo.
5. El validador humano revisa, ajusta o descarta cada hallazgo antes de que se guarde como conocimiento institucional.

## Arquitectura

| Capa | Componente | Detalle |
|---|---|---|
| Interfaz | Streamlit | Carga de archivos, panel de control, revisión humana (HITL) |
| Ingesta y RAG | DocumentLoader, AdaptiveChunker, MetadataManager | PDF/DOCX/TXT/CSV/IPYNB → chunks con metadata de normativa SBS |
| Recuperación | Elasticsearch (BM25 + kNN + RRF), HyDE | Búsqueda híbrida léxica + semántica, con documento hipotético previo a la búsqueda |
| Orquestación | LangGraph (ReAct Agent), GPT-4o-mini | Agente único con 7 herramientas especializadas |
| Memoria y gobierno | SQLite, RAGTracer | Hallazgos validados, trazas de cada consulta RAG, LangSmith opcional |

## Stack técnico

- **LLM:** GPT-4o-mini (agente principal y generación de documento hipotético en HyDE)
- **Embeddings:** `text-embedding-3-small` (1536 dimensiones)
- **Vector store:** Elasticsearch (búsqueda híbrida BM25 + kNN + Reciprocal Rank Fusion)
- **Orquestación:** LangChain / LangGraph
- **Interfaz:** Streamlit
- **Memoria:** SQLite
- **Observabilidad:** LangSmith (opcional)

## Normativa SBS indexada

El sistema cita evidencia real de tres resoluciones de la Superintendencia de Banca, Seguros y AFP del Perú:

- **Resolución SBS N° 3780-2011** — Reglamento de Gestión del Riesgo de Crédito (definición de incumplimiento)
- **Resolución SBS N° 00053-2023** — Reglamento de Gestión de Riesgos de Modelo (umbrales de Gini, PSI, backtesting)
- **Resolución SBS N° 272-2017** — Reglamento de Gobierno Corporativo y Gestión Integral de Riesgos (rol e independencia del validador, periodicidad)

## Configuración

### 1. Credenciales

Crea un archivo `api.txt` en la raíz del proyecto (nunca se sube al repositorio) siguiendo la estructura de [`.env.example`](.env.example):

```
OPENAI_API_KEY=sk-...
ELASTIC_URL=https://tu-cluster.es.region.aws.found.io:443
ELASTIC_API_KEY=...
LANGSMITH_API_KEY=lsv2-...   # opcional
```

### 2. Base de conocimiento

Crea la carpeta `knowledge_base/normativa_sbs/` y sube ahí los PDFs oficiales de las resoluciones SBS. Si la carpeta está vacía, el sistema usa fragmentos de demostración como respaldo para no quedarse sin evidencia mientras se cargan los documentos reales.

### 3. Indexación

Ejecuta el pipeline de ingesta (notebook principal) para indexar la normativa en Elasticsearch:

```python
pipeline_ingesta_kb(recrear_indice=True, usar_demo_si_vacio=False)
```

Una vez indexado, la normativa persiste en Elasticsearch de forma permanente — no es necesario repetir este paso en cada sesión, solo cuando se agregue normativa nueva.

### 4. Ejecutar la aplicación

El notebook principal escribe `app.py` y lo expone vía `pyngrok` para pruebas en Google Colab. Para un entorno local:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura del repositorio

```
├── ValidAIRisk_M4_Hybrid_RAG_SBS.ipynb   # Notebook principal: ingesta, RAG, pruebas, despliegue
├── app.py                                 # Aplicación Streamlit (generada por el notebook)
├── .env.example                           # Estructura de credenciales (sin valores reales)
└── .gitignore
```

## Roadmap

**Fase 01 — MVP actual (completo):** Streamlit, agente ReAct con tools, RAG híbrido (BM25 + kNN + RRF), HyDE, normativa SBS indexada y filtrable, memoria SQLite con trazabilidad RAG, revisión humana.

**Fase 02 — Evolución técnica (en curso):** comparación metodología vs. código, exportación automática a PDF, dashboard de métricas RAG, reranking con CrossEncoder, ampliación de la base normativa con documentos oficiales completos.

**Fase 03 — Producción y escalamiento (pendiente):** base de datos cloud, integración con almacenamiento externo (S3/GCP/Azure), roles y permisos, auditoría formal, control de versiones, despliegue seguro. La migración de SQLite a PostgreSQL se mantiene en evaluación: la memoria actual cubre el alcance del MVP y se activará cuando el volumen de hallazgos o usuarios concurrentes lo justifique.

## Principio de diseño

El agente se utiliza únicamente donde existe razonamiento dinámico; los pasos críticos del flujo quedan controlados por workflow explícito y revisión humana, no delegados completamente al modelo de lenguaje.
