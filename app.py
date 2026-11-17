"""
AI Knowledge Base System - Streamlit app (connected to the RAG backend)

Flow: upload PDF -> chunk + embed + store in Qdrant -> ask a question ->
Top-3 chunk retrieval -> Gemini answer + 3 source boxes.
"""
import html
import logging
import time
from pathlib import Path

import streamlit as st

import rag_backend as rag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

st.set_page_config(page_title="AI Knowledge Base System", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

MAX_QUESTION_LEN = 500
MIN_SECONDS_BETWEEN_QUESTIONS = 3
MAX_QUESTIONS_PER_SESSION = 60


def load_css(path="style.css"):
    f = Path(__file__).parent / path
    if f.exists():
        st.markdown(f"<style>{f.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()

# ------------------------------------------------------------ session state
ss = st.session_state
ss.setdefault("sid", rag.new_session_id())   # private id for this visitor
ss.setdefault("docs", {})                    # {upload_key: {"name","pages","chunks","chars"}}
ss.setdefault("failed", {})                  # {upload_key: error message}
ss.setdefault("result", None)
ss.setdefault("last_q_time", 0.0)
ss.setdefault("q_count", 0)

missing = rag.missing_config()


def validate_question(q):
    q = " ".join(q.split())
    if not q:
        return False, "Please enter a question."
    if len(q) > MAX_QUESTION_LEN:
        return False, f"Question is too long ({len(q)}/{MAX_QUESTION_LEN} characters)."
    return True, q


# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="hub-title">📁 Document Hub</div>'
                '<div class="hub-sub">Upload the PDFs you want to query</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Organization / Topic</div>', unsafe_allow_html=True)
    org = st.text_input("Organization", placeholder="Enter your company / document name",
                        label_visibility="collapsed", autocomplete="off", max_chars=80, key="org_name")

    st.markdown('<div class="sb-label">Upload new PDF report</div>', unsafe_allow_html=True)
    files = st.file_uploader("PDF", type=["pdf"], accept_multiple_files=True,
                             label_visibility="collapsed", disabled=bool(missing))

    if not missing:
        current = {f"{f.name}:{f.size}": f for f in (files or [])}

        # files the visitor removed -> delete their vectors (privacy + free-tier space)
        for key in [k for k in ss["docs"] if k not in current]:
            rag.delete_file(ss["sid"], ss["docs"][key]["name"])
            del ss["docs"][key]
        for key in [k for k in ss["failed"] if k not in current]:
            del ss["failed"][key]

        # index new files
        for key, f in current.items():
            if key in ss["docs"] or key in ss["failed"]:
                continue
            with st.spinner(f"Reading & indexing {f.name} ..."):
                try:
                    stats = rag.index_pdf(ss["sid"], f.name, f.getvalue())
                    ss["docs"][key] = {"name": f.name, **stats}
                except rag.RAGError as e:
                    ss["failed"][key] = str(e)
                except Exception:
                    log.exception("indexing crashed")
                    ss["failed"][key] = "Something went wrong while indexing this file."

    for key, info in ss["docs"].items():
        st.markdown(
            f'<div class="loaded-card">✓ Loaded: {html.escape(info["name"])}<br>'
            f'<small>{info["pages"]} pages · {info["chunks"]} chunks indexed</small></div>',
            unsafe_allow_html=True)
    for key, msg in ss["failed"].items():
        st.error(f'{key.rsplit(":", 1)[0]}: {msg}')

# ------------------------------------------------------------ header
st.markdown('<h1 class="app-title">🛡️ AI Knowledge Base System</h1>'
            '<div class="app-sub">Intelligent Document Search &amp; Real-time Q&amp;A Command Center</div>',
            unsafe_allow_html=True)
st.divider()

if missing:
    st.error("Setup incomplete: these secrets are missing: " + ", ".join(missing) +
             ". Add them in Streamlit → Settings → Secrets.")

# ------------------------------------------------------------ query interface
st.markdown('<div class="section-title">💬 AI Query Interface</div>', unsafe_allow_html=True)

doc_names = sorted({d["name"] for d in ss["docs"].values()})
options = ["All Loaded Documents"] + doc_names

c1, c2 = st.columns([5, 1])
with c1:
    scope = st.selectbox("Select active report context:", options)
with c2:
    st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if doc_names:
        st.markdown('<span class="badge-ok">● Document indexed</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-wait">● No document yet</span>', unsafe_allow_html=True)

with st.form("query_form", clear_on_submit=False):
    question = st.text_input("Enter your question regarding the report:",
                             max_chars=MAX_QUESTION_LEN, autocomplete="off")
    submitted = st.form_submit_button("Get Answer")

if submitted:
    ok, value = validate_question(question)
    now = time.time()
    if missing:
        st.error("The app is not configured yet.")
    elif not ok:
        st.error(value)
    elif not doc_names:
        st.warning("Please upload at least one PDF in the Document Hub first.")
    elif now - ss["last_q_time"] < MIN_SECONDS_BETWEEN_QUESTIONS:
        st.warning("Please wait a few seconds before asking another question.")
    elif ss["q_count"] >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit for this session reached. Please refresh the page.")
    else:
        ss["last_q_time"] = now
        ss["q_count"] += 1
        t0 = time.time()
        with st.spinner("Searching your documents..."):
            try:
                file_filter = None if scope == "All Loaded Documents" else scope
                chunks = rag.retrieve(ss["sid"], value, file_filter)
                if not chunks:
                    raise rag.RAGError("No content found in your documents for this question.")
                answer, model = rag.generate_answer(value, chunks, org.strip())
                ss["result"] = {"question": value, "answer": answer, "chunks": chunks,
                                "model": model, "seconds": time.time() - t0, "scope": scope}
            except rag.RAGError as e:
                ss["result"] = None
                st.error(str(e))
            except Exception:
                log.exception("query crashed")
                ss["result"] = None
                st.error("Something went wrong while generating the answer. Please try again.")

# ------------------------------------------------------------ output
st.divider()
st.markdown('<div class="section-title">📄 Verified Intelligence Output</div>', unsafe_allow_html=True)

res = ss["result"]
if not res:
    st.markdown('<div class="answer-box">Your answer will appear here after you upload a PDF '
                'and click <b>Get Answer</b>.</div>', unsafe_allow_html=True)
else:
    st.caption(f"Question: {res['question']}")
    with st.container(border=True):
        st.markdown(res["answer"])

    st.markdown('<div class="section-title" style="margin-top:1.4rem">📚 Sources</div>',
                unsafe_allow_html=True)
    for ch in res["chunks"]:
        label = f'📎 Source {ch["rank"]}  ·  {ch["source"]}  ·  page {ch["page"]}'
        with st.expander(label):
            st.markdown(
                f'''<div class="chunk-card">
                      <div class="chunk-head">
                        <span>Retrieved chunk from <b>{html.escape(str(ch["source"]))}</b> (page {html.escape(str(ch["page"]))})</span>
                        <span class="chunk-score">Similarity {max(ch["score"], 0):.0%}</span>
                      </div>
                      <div class="chunk-text">{html.escape(ch["text"])}</div>
                    </div>''', unsafe_allow_html=True)

    with st.expander("🔒 Security & Full Vector Log"):
        st.markdown(
            f"**Search scope:** {html.escape(res['scope'])}  \n"
            f"**Chunks retrieved (Top-K):** {len(res['chunks'])}  \n"
            f"**Best match score:** {res['chunks'][0]['score']:.1%}  \n"
            f"**Answer model:** {html.escape(res['model'])} (auto-failover enabled)  \n"
            f"**Embedding model:** {rag.EMBED_MODEL}  \n"
            f"**Response time:** {res['seconds']:.1f} s  \n"
            f"**Isolation:** your documents are visible only to your session"
        )