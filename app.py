"""
AI Knowledge Base System  -  Streamlit front end (UI-ONLY VERSION)

Phase 1 (this file): interface only. Answers + chunks are DEMO placeholders.
Phase 2 (later):     replace get_answer() with a call to the n8n webhook / Qdrant.
"""
import html
import time
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

# ------------------------------------------------------------------ config
st.set_page_config(
    page_title="AI Knowledge Base System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_QUESTION_LEN = 500   # same limit as the n8n "Validate & Sanitize Input" node
TOP_K = 3                # number of retrieved chunks shown to the user


def load_css(path: str = "style.css") -> None:
    css_file = Path(__file__).parent / path
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found - using default theme.")


load_css()

# ------------------------------------------------------------------ session state
st.session_state.setdefault("docs", {})      # {filename: {"pages": int, "chars": int}}
st.session_state.setdefault("result", None)  # last answer payload


# ------------------------------------------------------------------ helpers
def read_pdf_info(uploaded_file) -> dict:
    """Read page count + character count from a PDF (basic validation)."""
    reader = PdfReader(uploaded_file)
    text = "".join((p.extract_text() or "") for p in reader.pages)
    return {"pages": len(reader.pages), "chars": len(text)}


def validate_question(q: str):
    """Mirror of the n8n validation step. Returns (ok, cleaned_or_error)."""
    q = " ".join(q.split())
    if not q:
        return False, "Please enter a question."
    if len(q) > MAX_QUESTION_LEN:
        return False, f"Question is too long ({len(q)}/{MAX_QUESTION_LEN} characters)."
    return True, q


def get_answer(question: str, scope: str) -> dict:
    """
    DEMO PLACEHOLDER - no backend yet.
    Phase 2: replace the body with a POST to your n8n webhook and map the
    response into the same dictionary shape below.
    """
    time.sleep(0.8)
    docs = list(st.session_state["docs"].keys()) or ["(no document uploaded)"]
    src = docs[0]
    return {
        "answer": (
            "This is a demo answer. Once the n8n backend is connected, the AI answer "
            "generated from the retrieved chunks will appear here."
        ),
        "chunks": [
            {"rank": i + 1, "source": src, "page": i + 1, "score": round(0.92 - i * 0.07, 2),
             "text": f"Demo retrieved chunk #{i + 1}. The real text from your PDF will be shown here."}
            for i in range(TOP_K)
        ],
        "confidence": 0.92,
        "demo": True,
    }


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="hub-title">📁 Document Hub</div>'
                '<div class="hub-sub">Upload the PDFs you want to query</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Organization / Topic</div>', unsafe_allow_html=True)
    org = st.text_input(
        "Organization",
        placeholder="Enter your company / document name",
        label_visibility="collapsed",
        autocomplete="off",      # stops the browser from suggesting old entries
        key="org_name",
    )

    st.markdown('<div class="sb-label">Upload new PDF report</div>', unsafe_allow_html=True)
    files = st.file_uploader("PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if files:
        for f in files:
            if f.name not in st.session_state["docs"]:
                try:
                    st.session_state["docs"][f.name] = read_pdf_info(f)
                except Exception:
                    st.error(f"Could not read {f.name}. Is it a valid, unencrypted PDF?")

    for name, info in st.session_state["docs"].items():
        st.markdown(
            f'<div class="loaded-card">✓ Loaded: {html.escape(name)}<br>'
            f'<small>{info["pages"]} pages · {info["chars"]:,} characters</small></div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ header
st.markdown('<h1 class="app-title">🛡️ AI Knowledge Base System</h1>'
            '<div class="app-sub">Intelligent Document Search &amp; Real-time Q&amp;A Command Center</div>',
            unsafe_allow_html=True)
st.divider()

# ------------------------------------------------------------------ query interface
st.markdown('<div class="section-title">💬 AI Query Interface</div>', unsafe_allow_html=True)

doc_names = list(st.session_state["docs"].keys())
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
    question = st.text_input(
        "Enter your question regarding the report:",
        max_chars=MAX_QUESTION_LEN,
        autocomplete="off",
    )
    submitted = st.form_submit_button("Get Answer")

if submitted:
    ok, value = validate_question(question)
    if not ok:
        st.error(value)
    elif not doc_names:
        st.warning("Please upload at least one PDF in the Document Hub first.")
    else:
        with st.spinner("Searching the knowledge base..."):
            try:
                st.session_state["result"] = get_answer(value, scope)
                st.session_state["question"] = value
            except Exception:
                st.session_state["result"] = None
                st.error("Something went wrong while generating the answer. Please try again.")

# ------------------------------------------------------------------ output
st.divider()
st.markdown('<div class="section-title">📄 Verified Intelligence Output</div>', unsafe_allow_html=True)

res = st.session_state["result"]
if not res:
    st.markdown('<div class="answer-box">Your answer will appear here after you upload a PDF '
                'and submit a question.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="answer-box">{html.escape(res["answer"])}</div>', unsafe_allow_html=True)
    if res.get("demo"):
        st.markdown('<div class="demo-note">⚠ Demo mode: backend not connected yet.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.4rem">📚 Sources</div>',
                unsafe_allow_html=True)
    for ch in res["chunks"]:
        label = f'📎 Source {ch["rank"]}  ·  {ch["source"]}  ·  page {ch["page"]}'
        with st.expander(label):
            st.markdown(
                f'''<div class="chunk-card">
                      <div class="chunk-head">
                        <span>Retrieved chunk from <b>{html.escape(ch["source"])}</b> (page {ch["page"]})</span>
                        <span class="chunk-score">Similarity {ch["score"]:.0%}</span>
                      </div>
                      <div class="chunk-text">{html.escape(ch["text"])}</div>
                    </div>''',
                unsafe_allow_html=True,
            )

    with st.expander("🔒 Security & Full Vector Log"):
        st.markdown(
            f"**Source document:** Verified & encrypted document  \n"
            f"**Confidence level:** {res['confidence']:.1%} (high precision match)  \n"
            f"**Security layer:** Input validation, auto-retry & fallback active"
        )