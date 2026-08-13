from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import FinancialDocumentStore
from src.qa_chain import answer_question


load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="FinSight AI | Financial Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #eaf2f0;
        --muted: #91a19d;
        --panel: #111b20;
        --line: rgba(255,255,255,.08);
        --accent: #31d6a6;
        --accent-soft: rgba(49,214,166,.12);
    }
    .stApp { background: #081015; color: var(--ink); }
    [data-testid="stSidebar"] { background: #0d171c; border-right: 1px solid var(--line); }
    [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem; }
    h1, h2, h3 { letter-spacing: -.025em; }
    .brand-kicker { color: var(--accent); font-size: .78rem; font-weight: 750; letter-spacing: .15em; text-transform: uppercase; margin-bottom: .55rem; }
    .hero-title { font-size: clamp(2.6rem, 5vw, 4.6rem); line-height: .98; font-weight: 760; letter-spacing: -.055em; margin: 0; }
    .hero-title span { color: var(--accent); }
    .hero-copy { color: var(--muted); max-width: 720px; font-size: 1.08rem; line-height: 1.65; margin: 1.25rem 0 1.8rem; }
    .trust-row { display:flex; gap:.65rem; flex-wrap:wrap; margin-bottom: 1.8rem; }
    .trust-chip { border: 1px solid var(--line); border-radius: 999px; padding: .38rem .75rem; color: #b9c8c4; font-size: .78rem; background: rgba(255,255,255,.025); }
    .workflow-card { min-height: 154px; border: 1px solid var(--line); border-radius: 16px; padding: 1.25rem; background: linear-gradient(145deg, var(--panel), #0d171c); }
    .workflow-number { color: var(--accent); font: 700 .75rem monospace; }
    .workflow-card h3 { font-size: 1rem; margin: .7rem 0 .35rem; }
    .workflow-card p { color: var(--muted); font-size: .85rem; line-height: 1.5; margin:0; }
    .section-label { color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 700; margin: 2rem 0 .8rem; }
    .ready-card { border: 1px solid rgba(49,214,166,.28); background: var(--accent-soft); border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: 1rem; }
    .ready-title { color: var(--accent); font-weight: 700; font-size: .9rem; }
    .ready-copy { color: #afc0bb; font-size: .8rem; margin-top:.2rem; }
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: 16px; background: rgba(17,27,32,.72); padding: .4rem .65rem; }
    [data-testid="stChatInput"] { border-color: rgba(49,214,166,.35); }
    .source-card { border-left: 2px solid var(--accent); padding: .72rem .9rem; margin: .5rem 0; background: rgba(255,255,255,.025); border-radius: 0 10px 10px 0; }
    .source-title { color: #dce9e5; font-size:.82rem; font-weight:700; }
    .source-copy { color: var(--muted); font-size:.76rem; line-height:1.45; margin-top:.25rem; }
    .tech-line { color: var(--muted); font-size: .76rem; line-height:1.6; border-top: 1px solid var(--line); margin-top: 1.5rem; padding-top: 1rem; }
    div.stButton > button { border-radius: 10px; border: 1px solid var(--line); font-weight: 650; }
    div.stButton > button[kind="primary"] { background: var(--accent); color: #062018; border: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTED_QUESTIONS = [
    "What was total revenue and how did it change year over year?",
    "What are the three most important financial risks?",
    "Summarise the company's cash flow performance.",
    "Which business segment grew the fastest?",
]


@st.cache_resource(show_spinner=False)
def get_store() -> FinancialDocumentStore:
    return FinancialDocumentStore()


def reset_chat() -> None:
    st.session_state.messages = []


def display_answer(content: str) -> None:
    # Streamlit treats unescaped dollar signs as LaTeX delimiters.
    st.markdown(content.replace("$", r"\$"))


def display_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"View supporting evidence · {len(sources)} excerpts"):
        for source in sources:
            filename = html.escape(str(source["filename"]))
            preview = html.escape(str(source["preview"]))
            score = int(float(source["score"]) * 100)
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{filename} · Page {source['page']} · {score}% match</div>
                    <div class="source-copy">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


store = get_store()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("documents_ready", False)
st.session_state.setdefault("document_summary", None)
st.session_state.setdefault("uploader_key", 0)

api_key_found = bool(os.getenv("OPENAI_API_KEY"))
model_name = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

with st.sidebar:
    st.markdown("### ◈ FinSight AI")
    st.caption("Financial intelligence, grounded in evidence.")
    st.markdown("<div class='section-label'>Document workspace</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Add annual reports or financial statements",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    if st.button("Analyse documents", type="primary", use_container_width=True, disabled=not uploaded_files):
        try:
            with st.spinner("Building your financial knowledge base..."):
                summary = store.ingest(uploaded_files)
            st.session_state.documents_ready = True
            st.session_state.document_summary = summary
            reset_chat()
            st.success("Analysis workspace ready")
        except Exception as exc:
            st.session_state.documents_ready = False
            st.error(f"The PDFs could not be processed: {exc}")

    if st.session_state.documents_ready:
        st.markdown(
            """
            <div class="ready-card">
                <div class="ready-title">● Knowledge base active</div>
                <div class="ready-copy">Your reports are indexed and ready for cited analysis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Replace documents", use_container_width=True):
            store.reset()
            reset_chat()
            st.session_state.documents_ready = False
            st.session_state.document_summary = None
            st.session_state.uploader_key += 1
            st.rerun()

    st.markdown("<div class='section-label'>Session</div>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        reset_chat()
        st.rerun()

    with st.expander("About this project"):
        st.write("FinSight AI is a retrieval-augmented financial research tool. It extracts page-aware text, creates semantic embeddings, retrieves relevant evidence and generates cited analysis.")

    status = "API connected" if api_key_found else "API key required"
    st.markdown(
        f"""
        <div class="tech-line">
            <strong>{status}</strong><br>
            Python · Streamlit · ChromaDB<br>
            Sentence Transformers · OpenAI
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="brand-kicker">AI-powered filing analysis</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Financial answers.<br><span>Traceable evidence.</span></h1>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="hero-copy">Turn dense annual reports into clear, source-grounded insight. FinSight AI retrieves the most relevant disclosures, analyses the numbers and cites the exact pages behind every answer.</p>
    <div class="trust-row">
        <span class="trust-chip">Page-level citations</span>
        <span class="trust-chip">Semantic retrieval</span>
        <span class="trust-chip">Multi-document analysis</span>
        <span class="trust-chip">Evidence-first responses</span>
    </div>
    """,
    unsafe_allow_html=True,
)

summary = st.session_state.document_summary
if summary:
    metric_columns = st.columns(3)
    metric_columns[0].metric("Documents", summary["files"])
    metric_columns[1].metric("Pages indexed", summary["pages"])
    metric_columns[2].metric("Evidence sections", summary["chunks"])
else:
    workflow_columns = st.columns(3)
    workflow = [
        ("01", "Upload", "Add one or more annual reports, results statements or financial PDFs."),
        ("02", "Index", "Page-aware text is embedded locally and organised for semantic search."),
        ("03", "Investigate", "Ask questions and receive concise answers with supporting page citations."),
    ]
    for column, (number, title, copy) in zip(workflow_columns, workflow):
        column.markdown(f'<div class="workflow-card"><div class="workflow-number">{number}</div><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)

question: str | None = None

if st.session_state.documents_ready and not st.session_state.messages:
    st.markdown("<div class='section-label'>Start an investigation</div>", unsafe_allow_html=True)
    prompt_columns = st.columns(2)
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        if prompt_columns[index % 2].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            question = suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_answer(message["content"])
            display_sources(message.get("sources", []))
        else:
            st.markdown(message["content"])

typed_question = st.chat_input(
    "Ask FinSight about revenue, margins, cash flow, risks or business segments...",
    disabled=not st.session_state.documents_ready,
)
question = typed_question or question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources: list[dict] = []
        if not api_key_found:
            response = "Add your OpenAI API key to `.env`, restart the app, and try again."
            st.error(response)
        else:
            try:
                with st.spinner("Finding and checking the strongest evidence..."):
                    result = answer_question(question=question, store=store, chat_history=st.session_state.messages[:-1], model_name=model_name)
                response = result.answer
                sources = result.sources
                display_answer(response)
                display_sources(sources)
            except Exception as exc:
                response = f"The analysis could not be completed: {exc}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
