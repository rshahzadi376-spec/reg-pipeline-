import os
import re
import json
import uuid
import hashlib
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

# =========================================================
# 1. CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

def secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value not in (None, ""):
            return str(value)
    except Exception:
        pass
    value = os.getenv(name)
    return value if value not in (None, "") else default

GEMINI_API_KEY = secret("GEMINI_API_KEY") or secret("GOOGLE_API_KEY")
QDRANT_URL = secret("QDRANT_URL")
QDRANT_API_KEY = secret("QDRANT_API_KEY")
QDRANT_COLLECTION = secret("QDRANT_COLLECTION", "company_knowledge_gemini")
GEMINI_EMBED_MODEL = secret("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_LLM_MODEL = secret("GEMINI_LLM_MODEL", "gemini-2.5-flash")

EMBEDDING_DIMENSIONS = 3072
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
EMBEDDING_BATCH_SIZE = 10

st.set_page_config(
    page_title="AI Knowledge Base System",
    page_icon="📚",
    layout="wide",
)

# =========================================================
# 2. STYLING
# =========================================================
st.markdown("""
<style>
    .stApp { background: #ffffff; }
    [data-testid="stHeader"] { background: #ffffff; }
    .main .block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1500px; }
    .ak-title { color:#123B68; font-size:2rem; font-weight:750; margin-bottom:0.15rem; }
    .ak-subtitle { color:#60758b; font-size:0.96rem; margin-bottom:1.35rem; }
    .ak-stat {
        background:#EAF4FF; border:1px solid #D4E8FC; border-radius:14px;
        padding:17px 18px; min-height:112px;
    }
    .ak-stat-label { color:#42627F; font-size:0.82rem; font-weight:650; }
    .ak-stat-value { color:#123B68; font-size:1.65rem; font-weight:750; margin-top:7px; overflow-wrap:anywhere; }
    .ak-stat-note { color:#71869A; font-size:0.75rem; margin-top:2px; }
    .right-panel {
        background:#F0F7FF; border:1px solid #D8EAFE; border-radius:16px;
        padding:18px 16px; margin-top:0.4rem;
    }
    .right-heading { color:#174B7A; font-size:1.08rem; font-weight:750; margin-bottom:0.7rem; }
    .doc-card {
        background:#F7FBFF; border:1px solid #DCEBFA; border-radius:12px;
        padding:12px 14px; margin:8px 0;
    }
    .doc-name { color:#183E61; font-weight:700; overflow-wrap:anywhere; }
    .doc-meta { color:#6C8195; font-size:0.82rem; margin-top:4px; }
    div.stButton > button[kind="primary"] {
        background:#1769AA; border-color:#1769AA; color:white; border-radius:9px;
    }
    div[data-testid="stExpander"] { border-color:#D8E7F5; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. CLIENTS
# =========================================================
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

@st.cache_resource
def get_qdrant_client(url: str, api_key: str):
    return QdrantClient(url=url, api_key=api_key, timeout=30)

def config_status():
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if missing:
        st.error("Configuration missing: " + ", ".join(missing))
        st.info("Add these values to your local .env file. Do not share API keys in chat or upload them to GitHub.")
        return False
    return True

# =========================================================
# 4. EMBEDDINGS AND GEMINI LLM
# =========================================================
def embed_texts(texts: List[str], task_type: str) -> List[List[float]]:
    client = get_gemini_client(GEMINI_API_KEY)
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        response = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        batch_vectors = [item.values for item in response.embeddings]
        if len(batch_vectors) != len(batch):
            raise RuntimeError("Gemini returned an unexpected number of embeddings.")
        for vector in batch_vectors:
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    f"Embedding dimension mismatch: got {len(vector)}, expected {EMBEDDING_DIMENSIONS}. "
                    "Check the Qdrant collection vector size."
                )
        vectors.extend(batch_vectors)
    return vectors

def gemini_generate(prompt: str) -> str:
    client = get_gemini_client(GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    answer = getattr(response, "text", None)
    if not answer or not answer.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return answer.strip()

def extract_json_object(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for i, char in enumerate(cleaned):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(cleaned[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return {}

def analyze_document(full_text: str) -> Dict[str, Any]:
    excerpt = full_text[:14000]
    prompt = f"""Analyze this PDF excerpt and return ONLY valid JSON with these keys:
subject: short subject name
topics: array of 3 to 6 short topics
organization: organization name only if clearly mentioned, otherwise null
summary: one concise sentence
The document can be about any subject. Do not assume it is a business document.

DOCUMENT:
{excerpt}
"""
    data = extract_json_object(gemini_generate(prompt))
    topics = data.get("topics", [])
    if isinstance(topics, str):
        topics = [x.strip() for x in re.split(r"[,;]", topics) if x.strip()]
    if not isinstance(topics, list):
        topics = []
    org = data.get("organization")
    return {
        "subject": str(data.get("subject") or "Not identified")[:160],
        "topics": [str(x).strip()[:100] for x in topics[:8] if str(x).strip()],
        "organization": str(org).strip()[:200] if org else None,
        "summary": str(data.get("summary") or "")[:600],
    }

# =========================================================
# 5. PDF CHUNKING
# =========================================================
def split_text(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    if reader.is_encrypted:
        try:
            result = reader.decrypt("")
            if not result:
                raise ValueError("This PDF is password-protected.")
        except Exception as exc:
            raise ValueError("This PDF is password-protected and could not be opened.") from exc

    all_chunks = []
    full_text_parts = []
    chunk_index = 0
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        full_text_parts.append(page_text)
        for chunk in split_text(page_text):
            all_chunks.append({
                "chunk_index": chunk_index,
                "page_number": page_number,
                "content": chunk,
            })
            chunk_index += 1
    return all_chunks, len(reader.pages), "\n".join(full_text_parts)

# =========================================================
# 6. QDRANT COLLECTION AND SESSION-ISOLATED DATA
# =========================================================
def get_qdrant():
    return get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)

def ensure_collection():
    client = get_qdrant()
    collections = client.get_collections().collections
    names = [item.name for item in collections]
    if QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=models.Distance.COSINE,
            ),
        )
    info = client.get_collection(QDRANT_COLLECTION)
    # A simple vector configuration is expected for this app.
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        size = getattr(first, "size", None)
    else:
        size = getattr(vectors, "size", None)
    if size and size != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Your Qdrant collection '{QDRANT_COLLECTION}' has vector size {size}; "
            f"this app's Gemini embeddings have {EMBEDDING_DIMENSIONS} dimensions. "
            "Use a collection configured for 3072 dimensions or change the embedding dimensions consistently."
        )
    return client

def session_id() -> str:
    if "ak_session_id" not in st.session_state:
        st.session_state.ak_session_id = str(uuid.uuid4())
    return st.session_state.ak_session_id

def qdrant_filter(doc_id: Optional[str] = None):
    must = [
        models.FieldCondition(
            key="session_id",
            match=models.MatchValue(value=session_id()),
        )
    ]
    if doc_id:
        must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=doc_id),
            )
        )
    return models.Filter(must=must)

def scroll_all(client, query_filter, limit=1000):
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return points

def list_documents() -> List[Dict[str, Any]]:
    client = get_qdrant()
    points = scroll_all(client, qdrant_filter(), limit=1000)
    grouped = {}
    for point in points:
        payload = point.payload or {}
        doc_id = payload.get("document_id")
        if not doc_id:
            continue
        if doc_id not in grouped:
            grouped[doc_id] = {
                "document_id": doc_id,
                "filename": payload.get("filename", "Unnamed PDF"),
                "organization": payload.get("organization", "Personal"),
                "total_pages": int(payload.get("total_pages") or 0),
                "subject": payload.get("subject", "Not analyzed"),
                "topics": payload.get("topics") or [],
                "summary": payload.get("summary") or "",
                "created_at": payload.get("created_at", ""),
                "chunks": 0,
                "file_hash": payload.get("file_hash", ""),
            }
        grouped[doc_id]["chunks"] += 1
    return sorted(grouped.values(), key=lambda x: x.get("created_at", ""), reverse=True)

def save_pdf_to_qdrant(filename: str, organization: str, chunks: List[dict],
                       total_pages: int, full_text: str, insights: dict):
    if not chunks:
        raise ValueError("No readable text was found. Scanned PDFs may need OCR first.")

    client = get_qdrant()
    file_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    # Duplicate check is scoped to this browser session.
    existing = scroll_all(client, qdrant_filter(), limit=1000)
    if any((p.payload or {}).get("file_hash") == file_hash for p in existing):
        return 0

    embeddings = embed_texts(
        [item["content"] for item in chunks],
        task_type="RETRIEVAL_DOCUMENT",
    )
    doc_id = str(uuid.uuid4())
    created_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    points = []
    for item, vector in zip(chunks, embeddings):
        payload = {
            "session_id": session_id(),
            "document_id": doc_id,
            "filename": filename,
            "organization": organization,
            "total_pages": total_pages,
            "file_hash": file_hash,
            "chunk_index": item["chunk_index"],
            "page_number": item["page_number"],
            "content": item["content"],
            "subject": insights.get("subject", "Not analyzed"),
            "topics": insights.get("topics", []),
            "summary": insights.get("summary", ""),
            "created_at": created_at,
        }
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{item['chunk_index']}"))
        points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

    # Upload in batches to avoid very large requests.
    for start in range(0, len(points), 100):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[start:start + 100], wait=True)
    return len(points)

# =========================================================
# 7. RETRIEVAL AND ANSWERS
# =========================================================
def search_chunks(question: str, doc_id: Optional[str]) -> List[Dict[str, Any]]:
    query_vector = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    client = get_qdrant()
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=qdrant_filter(doc_id),
        limit=TOP_K,
        with_payload=True,
    )
    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append({
            "content": payload.get("content", ""),
            "source": payload.get("filename", "Unknown document"),
            "page": payload.get("page_number"),
            "organization": payload.get("organization", ""),
            "score": float(point.score or 0),
        })
    return results

def ask_gemini(question: str, results: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"SOURCE {i}\nDocument: {item['source']}\nPage: {item['page'] or 'Not available'}\n"
        f"Content:\n{item['content']}"
        for i, item in enumerate(results, start=1)
    )
    prompt = f"""You are AI Knowledge Base System, a careful document research assistant.
Answer ONLY from the supplied document context. Do not invent facts, dates, figures, or citations.
If the answer is not in the context, say: "I could not find this information in the selected document."
When the question asks for a year, match values to the correct year and do not mix table columns.
Keep the answer clear and directly address the question.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return gemini_generate(prompt)

# =========================================================
# 8. PAGE LAYOUT: EMPTY LEFT GUTTER, CENTER WORKSPACE, RIGHT UPLOAD
# =========================================================
left_blank, center, right = st.columns([0.055, 0.665, 0.28], gap="large")

with center:
    st.markdown('<div class="ak-title">📚 AI Knowledge Base System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ak-subtitle">Upload your PDFs, analyze their content, and ask questions using Gemini AI.</div>',
        unsafe_allow_html=True,
    )

if not config_status():
    st.stop()

try:
    qdrant = ensure_collection()
except Exception as error:
    st.error(f"Qdrant connection/collection error: {str(error)[:600]}")
    st.stop()

try:
    docs = list_documents()
except Exception as error:
    st.error(f"Could not load your documents from Qdrant: {str(error)[:500]}")
    st.stop()

# Right-side upload panel
with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    st.markdown('<div class="right-heading">📤 Add a PDF</div>', unsafe_allow_html=True)

    existing_orgs = sorted({
        d["organization"] for d in docs
        if d.get("organization") and d["organization"] not in ("Personal", "Unspecified")
    })
    org_options = ["Personal", "Add an organization"] + existing_orgs
    selected_org = st.selectbox("Organization", org_options, key="ak_org_select")
    if selected_org == "Add an organization":
        organization = st.text_input("Organization name", placeholder="Enter company / organization")
    else:
        organization = selected_org

    uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"], key="ak_pdf_upload")
    analyze_clicked = st.button("Analyze PDF", type="primary", use_container_width=True)

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Please choose a PDF first.")
        elif selected_org == "Add an organization" and not organization.strip():
            st.warning("Please enter the organization name.")
        else:
            try:
                with st.spinner("Extracting PDF, analyzing it with Gemini and saving vectors to Qdrant..."):
                    chunks, pages, full_text = extract_pdf(uploaded_file)
                    if not full_text.strip() or not chunks:
                        raise ValueError("No readable text found. Scanned PDFs may require OCR.")
                    insights = analyze_document(full_text)
                    org_name = organization.strip() or insights.get("organization") or "Personal"
                    saved = save_pdf_to_qdrant(
                        filename=uploaded_file.name,
                        organization=org_name,
                        chunks=chunks,
                        total_pages=pages,
                        full_text=full_text,
                        insights=insights,
                    )
                if saved == 0:
                    st.warning("This PDF is already uploaded in your current session.")
                else:
                    st.success(f"PDF analyzed. {saved} chunks saved to Qdrant.")
                    st.rerun()
            except Exception as error:
                st.error(f"PDF processing failed: {str(error)[:600]}")
    st.markdown('</div>', unsafe_allow_html=True)

# Main workspace
with center:
    selected_doc_id = None
    selected_doc = None

    if docs:
        labels = {
            f"{d['filename']} — {d['organization']}": d["document_id"]
            for d in docs
        }
        selected_label = st.selectbox("Select Document", list(labels.keys()), key="ak_selected_doc")
        selected_doc_id = labels[selected_label]
        selected_doc = next((d for d in docs if d["document_id"] == selected_doc_id), None)

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown(
        f"<div class='ak-stat'><div class='ak-stat-label'>Uploaded Documents</div>"
        f"<div class='ak-stat-value'>{len(docs)}</div><div class='ak-stat-note'>Your uploaded PDFs</div></div>",
        unsafe_allow_html=True,
    )
    pages = selected_doc["total_pages"] if selected_doc else 0
    c2.markdown(
        f"<div class='ak-stat'><div class='ak-stat-label'>Document Pages</div>"
        f"<div class='ak-stat-value'>{pages if selected_doc else '—'}</div>"
        f"<div class='ak-stat-note'>Pages in selected PDF</div></div>",
        unsafe_allow_html=True,
    )
    filename = selected_doc["filename"] if selected_doc else "No document selected"
    c3.markdown(
        f"<div class='ak-stat'><div class='ak-stat-label'>PDF File Name</div>"
        f"<div class='ak-stat-value' style='font-size:1.05rem'>{filename}</div>"
        f"<div class='ak-stat-note'>Selected document</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Your Documents")
    if docs:
        for doc in docs:
            st.markdown(
                f"<div class='doc-card'><div class='doc-name'>📄 {doc['filename']}</div>"
                f"<div class='doc-meta'>{doc['total_pages']} pages · {doc['organization']} · "
                f"{doc['chunks']} chunks · {doc['subject']}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No PDFs uploaded yet. Use the upload panel on the right to add your first document.")

    if selected_doc:
        with st.expander("Document insights and keywords", expanded=False):
            st.markdown(f"**Subject:** {selected_doc.get('subject') or 'Not identified'}")
            if selected_doc.get("summary"):
                st.write(selected_doc["summary"])
            topics = selected_doc.get("topics") or []
            if topics:
                st.markdown("**Keywords / Topics:** " + " · ".join(str(x) for x in topics))

    st.divider()
    st.markdown("### Ask a Question")
    question = st.text_area(
        "Enter your question",
        placeholder="Ask something about the selected PDF...",
        height=100,
        label_visibility="collapsed",
        key="ak_question",
    )
    if st.button("Get Answer", type="primary", disabled=not bool(docs), key="ak_get_answer"):
        if not selected_doc_id:
            st.warning("Please select a PDF first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching the selected PDF and generating an answer with Gemini..."):
                    results = search_chunks(question.strip(), selected_doc_id)
                    if not results:
                        st.warning("No relevant chunks were found in this PDF.")
                    else:
                        answer = ask_gemini(question.strip(), results)
                        st.session_state["ak_last_answer"] = answer
                        st.session_state["ak_last_sources"] = results
                        st.session_state["ak_last_question"] = question.strip()
            except Exception as error:
                st.error(f"Question answering failed: {str(error)[:600]}")

    if st.session_state.get("ak_last_answer"):
        st.markdown("### Answer")
        st.write(st.session_state["ak_last_answer"])
        sources = st.session_state.get("ak_last_sources", [])
        with st.expander("Sources & Keywords — View 3 relevant chunks", expanded=False):
            for i, item in enumerate(sources[:TOP_K], start=1):
                st.markdown(f"**{i}. Source:** {item['source']}")
                st.caption(f"Page: {item['page'] or 'Not available'}")
                st.write(item["content"])
                st.divider()

st.caption("AI Knowledge Base System · Gemini AI · Qdrant Cloud")
