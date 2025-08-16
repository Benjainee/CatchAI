CatchAI – Copiloto Conversacional sobre Documentos

Copiloto que permite subir hasta 5 PDFs y hacer preguntas en lenguaje natural sobre su contenido.
Responde de forma contextual (RAG) y ofrece resumen y clasificación por temas.
UI en Streamlit y contenerización con Docker + docker-compose.

🚀 Levantar el entorno (Docker + docker-compose)

💡 Dónde ejecutar los comandos: puedes usar la terminal integrada de VS Code (PowerShell/CMD/Bash). No es necesario abrir PowerShell por separado.
🪟 En Windows, asegúrate de que Docker Desktop está iniciado y con WSL2 habilitado.

0) Prerrequisitos

    Docker Desktop (WSL2 habilitado si estás en Windows).

    OpenAI API Key con cuota activa.

1) Variables de entorno

Crea tu archivo .env a partir del ejemplo:

    # Linux/Mac
    cp .env.example .env

    # Windows PowerShell (también funciona desde la terminal de VS Code)
    Copy-Item .env.example .env


Edita .env y coloca tu clave:

    OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXX   # obligatorio (no subir al repo)
    OPENAI_MODEL=gpt-4o-mini                     # opcional (default gpt-4o-mini)


2) Build & Run
    docker compose build --no-cache
    docker compose up


Abre: http://localhost:8501

Comandos útiles

    docker compose up -d         # ejecutar en segundo plano
    docker compose logs -f web   # ver logs del servicio 'web'
    docker compose down          # bajar servicios

🧪 Uso

    1. Sube hasta 5 PDFs en la barra lateral.

    2. Cada PDF se indexa (verás “Indexando… (Xs)”); cuando pase a Listo, ya puedes preguntar.

    3. Escribe tu pregunta en español (la respuesta cita páginas cuando es posible).

Acciones opcionales:

    🧾 Resumen del PDF activo (6–8 viñetas, técnico, páginas cuando aplica).
    🧩 Clasificar por temas (JSON con { tema, palabras_clave, paginas, resumen }).
    🧹 Limpieza: botón “Vaciar base vectorial” en la UI (borra ./chroma_db) o elimina la carpeta manualmente.


🧱 Arquitectura del sistema
┌────────────────────────────────┐
│            Streamlit           │  UI (chat + botones + estado)
│  - Uploader (≤5 PDFs)          │
│  - Estado por PDF (Index/Ready)│
│  - Acciones: QA/Resumen/Clasif.│
└───────────────┬────────────────┘
                │ (tareas)
┌───────────────▼────────────────┐
│         Orquestación           │
│  LangChain:                    │
│  - PyPDFLoader (parseo)        │
│  - RecursiveTextSplitter       │
│  - HuggingFaceEmbeddings       │
│  - Chroma (vector store)       │
│  - Retriever (MMR)             │
│  - RetrievalQA + prompts       │
│  - Ejecutores (index/QA)       │
└───────────────┬────────────────┘
                │ (contexto)
┌───────────────▼────────────────┐
│              LLM               │
│  OpenAI (langchain-openai)     │
│  - Modelo: gpt-4o-mini (def.)  │
│  - Respuestas en español       │
└────────────────────────────────┘


Detalles clave

. Split: chunk_size=900, chunk_overlap=180 para buen recall.
. Retriever (MMR): k=8, fetch_k=50, lambda_mult=0.5 (cobertura diversa).
. Persistencia: carpeta por PDF dentro de ./chroma_db (limpieza granular).
. Ejecutores:
    .Indexado en serie (evita locks en Windows).
    .QA/Resumen/Clasificar en paralelo (no escribe en disco).
Auto-Refresh 1s: timers y estados se actualizan en vivo.


🧠 Flujo conversacional (RAG)

    1. Upload → lectura con PyPDFLoader.
    2. Split → RecursiveCharacterTextSplitter (metadatos incluyen nombre del archivo y página).
    3. Vectorización → HuggingFaceEmbeddings → Chroma (colección por PDF).
    4. QA → Retriever (MMR) → RetrievalQA con prompt instructivo
    (responde solo con el contexto; cita páginas cuando puede).

Extras:
    . Resumen: síntesis técnica en 6–8 viñetas.
    . Clasificar: JSON de temas, keywords y páginas.

🧰 Variables de entorno:

    Variable	Obligatoria	Default	Descripción
        . OPENAI_API_KEY	Sí	—	API key de OpenAI
        . OPENAI_MODEL	No	gpt-4o-mini	Modelo de chat (puedes cambiarlo)

Puedes usar otro modelo compatible con langchain-openai si el evaluador lo prefiere.


📦 Estructura del repo
.
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
├─ .env.example            # ejemplo (sin credenciales)
└─ chroma_db/              # (se crea en runtime; se puede limpiar)


Sugerido en .gitignore:

    .env
    chroma_db/
    __pycache__/
    .venv/

🧠 Justificación de elecciones técnicas

    1. LangChain: facilita la orquestación (loaders, retrievers, prompts) y cumplir el requisito de “flujo extensible”.
    2. Chroma: simple, local y eficiente para POC; persistencia por documento mejora trazabilidad y limpieza.
    3. HuggingFace Embeddings (MiniLM L6 v2): buena relación calidad/velocidad sin costo por token.
    4. OpenAI (ChatOpenAI): robusto para QA/Resumen en español; configuración simple vía .env.
    5. Streamlit: prototipado rápido de UI con manejo de estado; auto-refresh mejora UX.

⚠️ Limitaciones actuales

    1. Sin OCR para PDFs escaneados (solo texto extraíble).
    2. Citas de páginas desde metadatos del loader (sin resaltado exacto en el PDF).
    3. Sin reranking específico (solo MMR).
    4. Dependencia de cuota de OpenAI (error 429 si no hay crédito).
    5. Contexto limitado por el máximo de tokens del modelo.


🗺️ Roadmap (mejoras futuras)

    1. 🔎 OCR (pytesseract + layout) para PDFs escaneados.
    2. 🪄 Reranking (Cohere Rerank / LLM-as-reranker) para mayor precisión.
    3. ⚖️ Comparación multi-PDF con botón en UI (la tarea está implementada; activar acción dedicada).
    4. 🌐 Embeddings multilingües (all-mpnet-base-v2 u OpenAI embeddings).
    5. 🧰 Backend FastAPI (API, auth, logging).
    6. ☁️ Despliegue Cloud (imagen pública y compose listo).
    7. 📝 Citas enriquecidas (resalte de spans en el PDF y enlaces a páginas).


🧪 Desarrollo local sin Docker (opcional)

    python -m venv .venv

    # Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1
    # Linux/Mac
    source .venv/bin/activate

    pip install -r requirements.txt
    cp .env.example .env   # agrega tu OPENAI_API_KEY
    streamlit run app.py
    
    Abrir: http://localhost:8501


🛠️ Troubleshooting

    1. 429 / insufficient_quota: revisa plan/billing de OpenAI o usa un modelo más barato.
    2. 401 (unauthorized): verifica OPENAI_API_KEY en .env.
    3. No se ve la UI: puerto ocupado; cambia mapeo en docker-compose.yml (por ejemplo 8502:8501).
    4. Index se queda “en curso”: usa “Vaciar base vectorial” y vuelve a subir; hay timeout de seguridad.
    5. Windows/WSL2/Docker: confirma Docker Desktop corriendo y suficiente espacio en disco.