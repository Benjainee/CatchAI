import os
import time
import tempfile
import hashlib
import shutil
from typing import List
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, Future
from streamlit_autorefresh import st_autorefresh
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(
    page_title="🧠 CatchAI – Copilot PDF",
    layout="wide",
    initial_sidebar_state="expanded"
)


try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "OPENAI_MODEL" in st.secrets:
        os.environ["OPENAI_MODEL"] = st.secrets["OPENAI_MODEL"]
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =========================
# Limpieza al iniciar (opcional)
# =========================
CLEAR_CHROMA_ON_START = False
PERSIST_DIR = "./chroma_db"
if CLEAR_CHROMA_ON_START and os.path.exists(PERSIST_DIR):
    try:
        shutil.rmtree(PERSIST_DIR)
    except Exception:
        pass
os.makedirs(PERSIST_DIR, exist_ok=True)

# =========================
# Estilos
# =========================
st.markdown("""
<style>
header { visibility: hidden; height: 0 !important; }
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { display:none !important; }

section[data-testid="stSidebar"]{
  min-width:320px !important; max-width:320px !important; border-right:1px solid #2a2a2a33;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label {
    display:block !important; border:1px solid #3a3a3a55; border-radius:10px;
    padding:10px 12px; margin:6px 0; background:transparent; cursor:pointer;
    transition:background .15s, border-color .15s, box-shadow .15s;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background:rgba(144,202,249,.10); border-color:#90caf955;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label[data-checked="true"] {
    background:rgba(100,181,246,.15); border-color:#64b5f680; box-shadow:inset 0 0 0 1px #64b5f640;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display:none; }
section[data-testid="stSidebar"] [role="radiogroup"] > label > div:last-child { width:100%; }

.sidebar-note { border-top:1px solid #2a2a2a33; margin:12px 0; padding-top:12px; }
.status-chip { display:inline-block; font-size:12px; padding:4px 8px; border-radius:999px; }
.status-ok { background:#e8f5e9; color:#1b5e20; border:1px solid #c8e6c9; }
.status-warn { background:#fff8e1; color:#8d6e63; border:1px solid #ffe0b2; }
.status-err { background:#ffebee; color:#b71c1c; border:1px solid #ffcdd2; }

.chat-bubble { color:#111 !important; }
.chat-bubble.assistant { color:#0b2a4a !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Recursos cacheados
# =========================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def get_llm():
    api_key = None
    model = "gpt-4o-mini"
    
    try:
        if 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
            api_key = st.secrets['openai']['api_key']
            if 'general' in st.secrets and 'model' in st.secrets['general']:
                model = st.secrets['general']['model']
    except:
        pass
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", model)
    
    if not api_key:
        st.error("🚫 API KEY REQUERIDA")
        st.stop()

    try:
        return ChatOpenAI(
            model=model,
            temperature=0.2,
            openai_api_key=api_key
        )
    except Exception as e:
        st.warning(f"Configuración estándar falló: {e}. Intentando configuración mínima...")
        try:
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key
            )
        except Exception as e2:
            st.error(f"Error crítico inicializando ChatOpenAI: {e2}")
            st.info("Probable conflicto de versiones. Ejecuta: pip install langchain-openai==0.1.0 openai==1.12.0")
            st.stop()

@st.cache_resource(show_spinner=False)
def get_index_executor():
    return ThreadPoolExecutor(max_workers=1)

@st.cache_resource(show_spinner=False)
def get_qa_executor():
    return ThreadPoolExecutor(max_workers=3)

# =========================
# Prompt para RetrievalQA
# =========================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "document_name"],
    template=(
        "Documento: {document_name}\n\n"
        "Contexto relevante del PDF:\n{context}\n\n"
        "Pregunta: {question}\n\n"
        "Instrucciones:\n"
        "- Responde solo usando el contexto.\n"
        "- Resume con precisión; usa bullets si ayuda.\n"
        "- Cita páginas como (p. X) cuando sea posible.\n"
        "- Si no hay información suficiente, di: "
        "\"No hay información sobre esto en el documento\".\n\n"
        "Respuesta:"
    )
)

QA_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un analista de documentos, técnico y preciso. Respondes en español."),
    ("human",
     "Documento: {document_name}\n\n"
     "Contexto relevante del PDF:\n{context}\n\n"
     "Pregunta: {question}\n\n"
     "Instrucciones:\n"
     "- Responde solo usando el contexto.\n"
     "- Resume con precisión; usa bullets si ayuda.\n"
     "- Cita páginas como (p. X) cuando sea posible.\n"
     "- Si no hay información suficiente, di: "
     "\"No hay información sobre esto en el documento\".\n\n"
     "Respuesta:")
])

# =========================
# Utilidades
# =========================
def get_file_hash(name: str, b: bytes) -> str:
    m = hashlib.md5(); m.update(b); m.update(name.encode()); return m.hexdigest()

def get_file_size_bytes(b: bytes) -> str:
    size = len(b)
    if size < 1024: return f"{size} B"
    if size < 1024*1024: return f"{size/1024:.1f} KB"
    return f"{size/(1024*1024):.1f} MB"

def process_pdf_bytes(name: str, b: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(b); tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=180,
            separators=["\n\n", "\n", r"\. ", " ", ""]
        )
        docs = splitter.split_documents(pages)
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source_name"] = name
        return docs
    finally:
        try: os.unlink(tmp_path)
        except: pass

def make_retriever(vector_db):
    return vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 50, "lambda_mult": 0.5}
    )

def display_chat_message(role: str, message: str):
    if role == "user":
        st.markdown(
            f"""
            <div class="chat-bubble" style="
                background-color: #f0f2f6; border-radius: 15px 15px 0 15px;
                padding: 12px 16px; margin: 8px 0; margin-left: 25%;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 15px;">
                <div style="font-weight:700; margin-bottom:4px; color:#111;">Tú</div>
                <div style="color:#111;">{message}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class="chat-bubble assistant" style="
                background-color: #e3f2fd; border-radius: 15px 15px 15px 0;
                padding: 12px 16px; margin: 8px 0; margin-right: 25%;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 15px;">
                <div style="font-weight:700; margin-bottom:4px; color:#0b2a2a;">Asistente</div>
                <div style="color:#0b2a4a;">{message}</div>
            </div>
            """, unsafe_allow_html=True)

def detect_intent(q: str) -> str:
    ql = (q or "").strip().lower()
    if any(k in ql for k in ["compara", "comparación", "vs", "versus", "diferencia"]):
        return "compare"
    if any(k in ql for k in ["resumen", "resume", "sumariza"]):
        return "summary"
    if any(k in ql for k in ["clasifica", "temas", "tópicos", "topics", "categoriza"]):
        return "classify"
    return "qa"

def _msg_content(res):
    return res if isinstance(res, str) else getattr(res, "content", str(res))

def _doc_dir(file_hash: str) -> str:
    return os.path.join(PERSIST_DIR, f"col_{file_hash[:8]}")

def _safe_delete_dir(path: str):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception:
        pass

def _friendly_openai_error(e: Exception) -> str:
    txt = str(e)
    low = txt.lower()
    if "insufficient_quota" in low or "exceeded your current quota" in low:
        return ("No se pudo consultar OpenAI: la cuenta no tiene crédito/cuota disponible.\n"
                "Opciones: agrega método de pago o aumenta límites; o prueba más tarde.")
    if "rate limit" in low or "429" in low:
        return "OpenAI respondió 429 (rate limit). Intenta de nuevo en unos segundos."
    if "401" in low or "invalid_api_key" in low:
        return "API key inválida o no autorizada. Revisa OPENAI_API_KEY en tu .env/secrets."
    return f"Error del proveedor LLM: {txt[:300]}"

# =========================
# Tareas en background
# =========================
def index_task(file_hash: str, name: str, b: bytes, embeddings):
    try:
        docs = process_pdf_bytes(name, b)
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["file_id"] = file_hash

        doc_dir = _doc_dir(file_hash)
        os.makedirs(doc_dir, exist_ok=True)

        vectordb = Chroma.from_documents(
            docs, embeddings,
            collection_name="doc",
            persist_directory=doc_dir
        )
        return True, vectordb
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:400]}"

def _format_sources_from_resp(resp, fallback_doc_name: str) -> str:
    src_docs = resp.get("source_documents", []) if isinstance(resp, dict) else []
    if not src_docs:
        return ""
    seen = {}
    for d in src_docs:
        page = d.metadata.get("page")
        name = d.metadata.get("source_name", fallback_doc_name)
        seen.setdefault(name, set()).add(page)
    lines = ["\n\n**Fuentes:**"]
    for name, pages in seen.items():
        pages = [p for p in pages if p is not None]
        if pages:
            pages_sorted = ", ".join(str(p) for p in sorted(pages))
            lines.append(f"- {name} (págs: {pages_sorted})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)

def qa_task(vector_db, question: str, doc_name: str, llm) -> str:
    retriever = make_retriever(vector_db)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAT_PROMPT.partial(document_name=doc_name)}
    )
    try:
        resp = qa.invoke({"query": question})
        answer = resp["result"] if isinstance(resp, dict) and "result" in resp else _msg_content(resp)
        sources = _format_sources_from_resp(resp, doc_name)
        return answer + (sources if sources else "")
    except Exception as e:
        return _friendly_openai_error(e)

def summary_task(vector_db, doc_name: str, llm) -> str:
    retr = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 50, "lambda_mult": 0.6}
    )
    ctx_docs = retr.invoke("Resumen general del documento")
    joined = "\n\n".join(d.page_content for d in ctx_docs[:18])
    prompt = (
        f"Documento: {doc_name}\n\n"
        "Resume en 6–8 viñetas claras y técnicas, en español. "
        "Incluye páginas cuando puedas (p. X). Si falta contexto, dilo.\n\n"
        f"Texto:\n{joined}"
    )
    try:
        res = llm.invoke(prompt)
        return _msg_content(res)
    except Exception as e:
        return _friendly_openai_error(e)

def compare_task(vector_dbs: List, names: List[str], llm) -> str:
    partials = []
    for vdb, nm in zip(vector_dbs, names):
        retr = vdb.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 40})
        ctx = retr.invoke("Puntos clave y conclusiones principales")
        txt = "\n\n".join(d.page_content for d in ctx[:16])
        try:
            resp = llm.invoke(f"Resume técnico en 4–6 bullets (usa páginas si puedes) del documento '{nm}':\n\n{txt}")
            part = _msg_content(resp)
        except Exception as e:
            part = f"[{nm}] {_friendly_openai_error(e)}"
        partials.append(f"### {nm}\n{part}")
    synth = "\n\n".join(partials)
    prompt = (
        "Compara los documentos anteriores. Entrega:\n"
        "- Similitudes\n- Diferencias\n- Riesgos/lagunas\n- Conclusiones accionables\n"
        "Sé concreto y en español."
    )
    try:
        final = llm.invoke(synth + "\n\n" + prompt)
        return _msg_content(final)
    except Exception as e:
        return _friendly_openai_error(e)

def classify_task(vector_db, doc_name: str, llm) -> str:
    retr = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 24, "fetch_k": 90})
    ctx = retr.invoke("Temas principales, conceptos y secciones del documento")
    sample = "\n".join(d.page_content for d in ctx[:32])
    prompt = (
        f"Documento: {doc_name}\n"
        "Agrupa el contenido en temas. Devuelve JSON con esta forma:\n"
        "[{ \"tema\": str, \"palabras_clave\": [..], \"paginas\": [..], \"resumen\": str }]\n"
        "Sé breve y útil."
    )
    try:
        res = llm.invoke(prompt + "\n\n" + sample)
        return _msg_content(res)
    except Exception as e:
        return _friendly_openai_error(e)

# =========================
# App
# =========================

def main():
    st.title("🧠 CatchAI – Copiloto Conversacional sobre Documentos")

    ss = st.session_state
    ss.setdefault('embeddings', get_embeddings())
    ss.setdefault('llm', get_llm())

    index_executor = get_index_executor()
    qa_executor = get_qa_executor()

    ss.setdefault('processed_files', {})
    ss.setdefault('selected_file', None)
    ss.setdefault('qa_jobs', {})
    ss.setdefault('index_jobs', {})     
    ss.setdefault('deleted_hashes', set())

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.subheader("📂 Documentos Subidos")

        uploaded = st.file_uploader(
            "Arrastra o haz clic para subir PDFs (máx 5)",
            type=["pdf"], accept_multiple_files=True,
            key="file_uploader", label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🗑 Limpiar todo"):
                ss.clear()
                try:
                    if os.path.exists(PERSIST_DIR):
                        shutil.rmtree(PERSIST_DIR)
                    os.makedirs(PERSIST_DIR, exist_ok=True)
                except Exception:
                    pass
                st.rerun()
        with col_btn2:
            if st.button("🧹 Vaciar base vectorial"):
                try:
                    if os.path.exists(PERSIST_DIR):
                        shutil.rmtree(PERSIST_DIR)
                    os.makedirs(PERSIST_DIR, exist_ok=True)
                    for h, info in list(ss.processed_files.items()):
                        info.pop("vector_db", None)
                        info["status"] = "error"
                        info["error"] = "Base vectorial vaciada. Re-subir para reindexar."
                except Exception:
                    pass
                st.rerun()

        current_hashes, uploaded_buffers = set(), {}
        if uploaded:
            for uf in uploaded:
                b = uf.getvalue()
                fhash = get_file_hash(uf.name, b)
                current_hashes.add(fhash)
                uploaded_buffers[fhash] = (uf.name, b)

        to_remove = [h for h in ss.processed_files.keys() if h not in current_hashes]
        for h in to_remove:
            ss.deleted_hashes.add(h)
            ss.qa_jobs.pop(h, None)
            _safe_delete_dir(_doc_dir(h))
            ss.index_jobs.pop(h, None)
            ss.processed_files.pop(h, None)
            if ss.selected_file == h:
                ss.selected_file = None

        if uploaded:
            existing = list(ss.processed_files.keys())
            new_items = []
            for h in current_hashes:
                if h not in ss.processed_files and h not in [x[0] for x in new_items]:
                    name, b = uploaded_buffers[h]
                    new_items.append((h, name, b))
            if len(existing) + len(new_items) > 5:
                st.error("Máximo 5 archivos en total."); st.stop()

            for fhash, fname, b in new_items:
                doc_dir = _doc_dir(fhash)
                ss.processed_files[fhash] = {
                    'name': fname,
                    'size': get_file_size_bytes(b),
                    'status': 'processing',
                    'persist_dir': doc_dir,
                    '_start_index': time.time(),
                }
                fut = index_executor.submit(index_task, fhash, fname, b, ss.embeddings)
                ss.index_jobs[fhash] = fut

            if not ss.selected_file and ss.processed_files:
                ss.selected_file = next(iter(ss.processed_files.keys()))

        finished_index = []
        for h, fut in list(ss.index_jobs.items()):
            if fut.done():
                ok, result = fut.result()
                if h in ss.deleted_hashes:
                    _safe_delete_dir(_doc_dir(h))
                    finished_index.append(h)
                    continue
                if ok:
                    ss.processed_files[h]['vector_db'] = result
                    ss.processed_files[h]['status'] = 'ready'
                else:
                    ss.processed_files[h]['status'] = 'error'
                    ss.processed_files[h]['error'] = result
                finished_index.append(h)
        for h in finished_index:
            ss.index_jobs.pop(h, None)
        if finished_index:
            st.rerun()

        for h, fut in list(ss.index_jobs.items()):
            start = ss.processed_files[h].setdefault('_start_index', time.time())
            if time.time() - start > 180 and not fut.done():
                ss.processed_files[h]['status'] = 'error'
                ss.processed_files[h]['error'] = 'Indexado excedió 180s (posible lock). Quita/pon el archivo.'

        if ss.processed_files:
            st.caption("Selecciona un documento")
            items = list(ss.processed_files.items())
            hashes = [h for h, _ in items]
            try:
                current_index = hashes.index(ss.selected_file) if ss.selected_file else 0
            except ValueError:
                current_index = 0

            def label_for(i):
                info = items[i][1]
                base = f"{i+1}. {info['name']}"
                status = info.get('status')
                if status == 'ready':
                    badge = " (Listo)"
                elif status == 'processing':
                    elapsed = int(time.time() - info.get('_start_index', time.time()))
                    badge = f" (Indexando… {elapsed}s)"
                elif status == 'error':
                    badge = " (Error)"
                else:
                    badge = ""
                return base + badge

            selected_index = st.radio(
                "Selecciona un PDF",
                options=list(range(len(items))),
                index=min(current_index, len(items)-1) if items else 0,
                format_func=label_for, label_visibility="collapsed",
                horizontal=False, key="pdf_selector_radio"
            )
            ss.selected_file = hashes[selected_index]

        st.markdown(
            """<div class="sidebar-note" style="font-size:14px; opacity:.9;">
            <strong>Flujo:</strong><br>
            • Sube hasta 5 PDFs (index en serie, carpeta por PDF)<br>
            • La “x” del uploader elimina también el índice<br>
            • “Vaciar base vectorial” limpia ./chroma_db completo<br>
            </div>""",
            unsafe_allow_html=True
        )

    # ---------- MAIN ----------
    if not ss.processed_files:
        st.info("Sube tus archivos PDF en el panel izquierdo"); return
    if not ss.selected_file:
        st.info("Selecciona un documento en la barra izquierda"); return

    finfo = ss.processed_files[ss.selected_file]
    st.subheader(f"📑 {finfo['name']}")
    if finfo.get('status') == 'ready':
        st.markdown("<span class='status-chip status-ok'>Listo para preguntas</span>", unsafe_allow_html=True)
    elif finfo.get('status') == 'processing':
        elapsed = int(time.time() - finfo.get('_start_index', time.time()))
        st.markdown(f"<span class='status-chip status-warn'>Indexando… ({elapsed}s)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-chip status-err'>Error al indexar</span>", unsafe_allow_html=True)
    st.markdown("---")

    hist_key = f"hist_{ss.selected_file}"
    ss.setdefault(hist_key, [])
    for q, a in ss[hist_key]:
        display_chat_message("user", q)
        display_chat_message("assistant", a)

    can_ask = finfo.get('status') == 'ready' and 'vector_db' in finfo

    qa_jobs = ss.qa_jobs
    job = qa_jobs.get(ss.selected_file)
    job_running = bool(job and isinstance(job.get("future"), Future) and not job["future"].done())

    # ======= Acciones extra =======
    col_extra1, col_extra2, col_extra3 = st.columns(3)
    with col_extra1:
        if st.button("🧾 Resumen del PDF activo", disabled=not can_ask or job_running):
            fut = get_qa_executor().submit(summary_task, finfo['vector_db'], finfo['name'], ss.llm)
            qa_jobs[ss.selected_file] = {"future": fut, "question": "[RESUMEN]", "start": time.time()}
            st.rerun()
    with col_extra2:
        if st.button("🧩 Clasificar por temas", disabled=not can_ask or job_running):
            fut = get_qa_executor().submit(classify_task, finfo['vector_db'], finfo['name'], ss.llm)
            qa_jobs[ss.selected_file] = {"future": fut, "question": "[CLASIFICA]", "start": time.time()}
            st.rerun()
    with col_extra3:
        pass

    st.markdown("---")

    # ======= Form de preguntas =======
    with st.form(key=f"form_{ss.selected_file}"):
        question = st.text_area(
            "Escribe tu pregunta…",
            height=100, label_visibility="collapsed",
            disabled=not can_ask or job_running, key=f"q_{ss.selected_file}"
        )
        colA, colB = st.columns([1,1])
        with colA:
            submitted = st.form_submit_button("Enviar", disabled=(not can_ask) or job_running)
        with colB:
            cancel_clicked = st.form_submit_button("Cancelar respuesta", disabled=(not job_running))

        if submitted:
            q = (question or "").strip()
            if not q:
                st.warning("Escribe una pregunta antes de enviar.")
            else:
                fut = get_qa_executor().submit(qa_task, finfo['vector_db'], q, finfo['name'], ss.llm)
                qa_jobs[ss.selected_file] = {"future": fut, "question": q, "start": time.time()}
                st.rerun()

        if cancel_clicked and job_running:
            qa_jobs.pop(ss.selected_file, None)
            st.info("Respuesta cancelada.")
            st.rerun()

    # ======= Polling de todas las tareas =======
    finished_any = False
    for file_hash, j in list(qa_jobs.items()):
        fut: Future = j["future"]
        if fut.done():
            try:
                answer = fut.result(timeout=0)
            except Exception as e:
                msg = _friendly_openai_error(e)
                hkey = f"hist_{file_hash}"
                ss.setdefault(hkey, [])
                ss[hkey].append((j["question"], msg))
            else:
                hkey = f"hist_{file_hash}"
                ss.setdefault(hkey, [])
                ss[hkey].append((j["question"], answer if isinstance(answer, str) else str(answer)))
            qa_jobs.pop(file_hash, None)
            finished_any = True

    if finished_any:
        st.rerun()

    # ======= Timers en vivo =======
    any_index_running = any(not fut.done() for fut in ss.index_jobs.values())
    any_qa_running = any(isinstance(j.get("future"), Future) and not j["future"].done() for j in qa_jobs.values())
    if any_index_running or any_qa_running:
        st_autorefresh(interval=1000, key="auto_refresh")

    job = qa_jobs.get(ss.selected_file)
    if job and not job["future"].done():
        elapsed = int(time.time() - job.get("start", time.time()))
        st.info(f"🧠 Pensando… ({elapsed}s). Puedes seguir subiendo PDFs o navegar por la app.")

if __name__ == "__main__":
    main()
