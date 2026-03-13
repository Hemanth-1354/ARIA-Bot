"""
app.py — ARIA: AI Research Intelligence Assistant
Production-grade Streamlit application.

Features:
  · Streaming LLM responses (token by token)
  · Conversation history with named sessions in sidebar
  · RAG over uploaded PDFs (FAISS + HuggingFace embeddings)
  · Live web search via Tavily
  · Concise / Detailed response mode
  · Full try/except error handling on every function
"""

import os
import sys
import logging
import uuid
import json
from datetime import datetime
from typing import Generator

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.llm import get_chatgroq_model
from utils.rag import ingest_pdf, retrieve_relevant_chunks
from utils.web_search import search_web
from utils.prompt_builder import build_system_prompt
from config.config import GROQ_API_KEY, TAVILY_API_KEY

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #111318;
    color: #e2e4e9;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1c1f2a;
    width: 260px !important;
}
section[data-testid="stSidebar"] > div { padding: 0; }

/* ── Main area ── */
.main .block-container {
    max-width: 820px;
    padding: 0 2rem 6rem 2rem;
    margin: 0 auto;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar: brand ── */
.sb-brand {
    padding: 1.4rem 1.2rem 1rem;
    border-bottom: 1px solid #1c1f2a;
}
.sb-brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.sb-brand-sub {
    font-size: 0.68rem;
    color: #3d4255;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Sidebar: section label ── */
.sb-label {
    font-size: 0.62rem;
    font-weight: 600;
    color: #3d4255;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 1rem 1.2rem 0.4rem;
}

/* ── Sidebar: new chat button ── */
.stButton > button[kind="primary"],
div[data-testid="stSidebar"] .stButton > button {
    background: #1e2235 !important;
    border: 1px solid #2a2f47 !important;
    color: #9da5c7 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 0.45rem 1rem !important;
    width: 100%;
    text-align: left !important;
    transition: all 0.15s ease !important;
    cursor: pointer;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #252a42 !important;
    border-color: #818cf8 !important;
    color: #e2e4e9 !important;
}

/* ── Conversation history items ── */
.conv-item {
    display: flex;
    align-items: center;
    padding: 0.5rem 1.2rem;
    font-size: 0.78rem;
    color: #6b7280;
    cursor: pointer;
    border-radius: 0;
    transition: background 0.1s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    gap: 0.5rem;
    border-left: 2px solid transparent;
}
.conv-item:hover { background: #151820; color: #c4c9da; }
.conv-item.active {
    background: #151820;
    color: #e2e4e9;
    border-left-color: #818cf8;
}
.conv-item-icon { opacity: 0.4; font-size: 0.7rem; flex-shrink: 0; }
.conv-item-text { overflow: hidden; text-overflow: ellipsis; }
.conv-item-time { font-size: 0.62rem; color: #374151; margin-left: auto; flex-shrink: 0; }

/* ── Sidebar controls ── */
.sb-control {
    padding: 0 1.2rem;
    margin-bottom: 0.4rem;
}

/* ── Status indicator ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.74rem;
    color: #4b5563;
    padding: 0.25rem 0;
    font-family: 'Inter', sans-serif;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}
.sd-green  { background: #34d399; box-shadow: 0 0 4px #34d399aa; }
.sd-red    { background: #ef4444; }
.sd-gray   { background: #374151; }

/* ── Main: page header ── */
.page-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid #1c1f2a;
    margin-bottom: 1.5rem;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #4b5563;
    margin-top: 0.4rem;
    font-weight: 400;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
    border-bottom: 1px solid #1c1f2a !important;
}
[data-testid="stChatMessage"]:last-child {
    border-bottom: none !important;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"],
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #13151e !important;
    border-radius: 10px !important;
    border: 1px solid #1c1f2a !important;
    padding: 1rem !important;
    margin-bottom: 0.5rem !important;
}

/* ── Source pills ── */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.67rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 3px 9px;
    border-radius: 4px;
    margin-right: 5px;
    margin-top: 8px;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
}
.sp-doc { background: #0a1f12; color: #4ade80; border: 1px solid #14532d; }
.sp-web { background: #0a1525; color: #60a5fa; border: 1px solid #1e3a5f; }
.sp-llm { background: #13091f; color: #c084fc; border: 1px solid #3b1f5e; }

/* ── Empty state ── */
.empty-wrap {
    padding: 3rem 0 1rem;
}
.empty-greeting {
    font-size: 1.5rem;
    font-weight: 600;
    color: #e2e4e9;
    font-family: 'Inter', sans-serif;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.empty-sub {
    font-size: 0.88rem;
    color: #4b5563;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* ── Suggestion cards ── */
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 1.5rem;
}
.suggestion-card {
    background: #13151e;
    border: 1px solid #1e2235;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
}
.suggestion-card:hover {
    border-color: #818cf8;
    background: #161928;
}
.sc-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #c4c9da;
    margin-bottom: 3px;
    font-family: 'Inter', sans-serif;
}
.sc-sub {
    font-size: 0.72rem;
    color: #4b5563;
    font-family: 'Inter', sans-serif;
    line-height: 1.4;
}

/* ── Response mode toggle ── */
.stRadio > label { display: none !important; }
.stRadio > div {
    display: flex !important;
    gap: 6px !important;
    flex-direction: row !important;
}
.stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    background: #13151e !important;
    border: 1px solid #1e2235 !important;
    border-radius: 6px !important;
    padding: 0.3rem 0.8rem !important;
    font-size: 0.75rem !important;
    font-family: 'Inter', sans-serif !important;
    color: #6b7280 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
.stRadio > div > label:has(input:checked) {
    background: #1e2235 !important;
    border-color: #818cf8 !important;
    color: #e2e4e9 !important;
}

/* ── Toggle ── */
.stToggle > label { font-size: 0.78rem !important; color: #6b7280 !important; }

/* ── Expander ── */
details summary {
    font-size: 0.75rem !important;
    color: #4b5563 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.02em !important;
}
details summary:hover { color: #9da5c7 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0d0f14 !important;
    border: 1px dashed #1e2235 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2a2f47 !important;
}

/* ── Chat input ── */
[data-testid="stBottom"] {
    background: linear-gradient(to top, #111318 70%, transparent) !important;
    padding: 1rem 0 0.5rem !important;
}
[data-testid="stChatInput"] {
    background: #13151e !important;
    border: 1px solid #1e2235 !important;
    border-radius: 10px !important;
    box-shadow: 0 0 0 0 transparent !important;
    transition: border-color 0.15s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    color: #e2e4e9 !important;
    caret-color: #818cf8 !important;
}

/* ── Spinner ── */
.stSpinner { color: #818cf8 !important; }

/* ── Alert / warning ── */
.stAlert { border-radius: 8px !important; font-size: 0.82rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2235; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #2a2f47; }

/* ── Divider ── */
hr { border-color: #1c1f2a !important; margin: 0.6rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all session state keys with safe defaults."""
    defaults: dict = {
        # Conversations: dict of { id: { title, messages, created_at } }
        "conversations":     {},
        "active_conv_id":    None,
        # Per-session document state
        "vector_store":      None,
        "uploaded_filename": None,
        # UI settings
        "response_mode":     "detailed",
        "use_web_search":    True,
        # Pending prompt from suggestion click
        "pending_prompt":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Ensure there is always at least one active conversation
    if not st.session_state.active_conv_id or \
       st.session_state.active_conv_id not in st.session_state.conversations:
        _create_new_conversation()


def _create_new_conversation(title: str = "New conversation") -> str:
    """Create a fresh conversation, set it as active, return its id."""
    try:
        conv_id = str(uuid.uuid4())[:8]
        st.session_state.conversations[conv_id] = {
            "title":      title,
            "messages":   [],
            "created_at": datetime.now().strftime("%H:%M"),
        }
        st.session_state.active_conv_id = conv_id
        logger.info("New conversation created: %s", conv_id)
        return conv_id
    except Exception as e:
        logger.error("Failed to create conversation: %s", e)
        raise RuntimeError(f"Could not create conversation: {e}") from e


def _active_messages() -> list:
    """Return the messages list for the active conversation."""
    try:
        conv_id = st.session_state.active_conv_id
        return st.session_state.conversations[conv_id]["messages"]
    except (KeyError, TypeError):
        return []


def _add_message(role: str, content: str, source: str = "") -> None:
    """Append a message to the active conversation."""
    try:
        conv_id = st.session_state.active_conv_id
        msg: dict = {"role": role, "content": content}
        if source:
            msg["source"] = source
        st.session_state.conversations[conv_id]["messages"].append(msg)

        # Auto-title the conversation from the first user message
        if role == "user":
            msgs = st.session_state.conversations[conv_id]["messages"]
            if len(msgs) == 1:
                title = content[:38] + ("…" if len(content) > 38 else "")
                st.session_state.conversations[conv_id]["title"] = title

    except Exception as e:
        logger.error("Failed to add message: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_langchain_messages(system_prompt: str, messages: list) -> list:
    """Convert flat message dicts to LangChain message objects."""
    formatted = [SystemMessage(content=system_prompt)]
    for m in messages:
        if m["role"] == "user":
            formatted.append(HumanMessage(content=m["content"]))
        else:
            formatted.append(AIMessage(content=m["content"]))
    return formatted


def stream_response(system_prompt: str, messages: list) -> Generator:
    """
    Stream tokens from Groq.

    Yields:
        str token chunks as they arrive.

    Raises:
        RuntimeError: on model init or streaming failure.
    """
    try:
        model = get_chatgroq_model(mode=st.session_state.response_mode)
        lc_messages = _build_langchain_messages(system_prompt, messages)
        for chunk in model.stream(lc_messages):
            try:
                yield chunk.content
            except Exception as e:
                logger.warning("Chunk decode error: %s", e)
    except RuntimeError:
        raise
    except Exception as e:
        logger.error("Streaming failed: %s", e)
        raise RuntimeError(f"Streaming error: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Source pills HTML
# ─────────────────────────────────────────────────────────────────────────────
def _pills_html(source: str) -> str:
    mapping = {
        "rag":     '<span class="source-pill sp-doc">Document</span>',
        "web":     '<span class="source-pill sp-web">Web</span>',
        "llm":     '<span class="source-pill sp-llm">LLM</span>',
        "rag+web": '<span class="source-pill sp-doc">Document</span>'
                   '<span class="source-pill sp-web">Web</span>',
    }
    return mapping.get(source, "")


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(prompt: str) -> None:
    """
    Full pipeline: RAG → web search → system prompt → streamed LLM response.
    Must be called from within the main content area (not inside a with block).
    Handles its own chat_message rendering.
    """
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        pill_placeholder     = st.empty()

        try:
            # 1. RAG retrieval
            rag_context = ""
            if st.session_state.vector_store:
                try:
                    rag_context = retrieve_relevant_chunks(
                        st.session_state.vector_store, prompt
                    )
                except RuntimeError as e:
                    st.warning(f"Document retrieval issue: {e}")
                    logger.warning("RAG failed: %s", e)

            # 2. Web search
            web_context = ""
            if st.session_state.use_web_search and TAVILY_API_KEY:
                try:
                    web_context = search_web(prompt) or ""
                except RuntimeError as e:
                    st.warning(f"Web search issue: {e}")
                    logger.warning("Web search failed: %s", e)

            # 3. Build system prompt
            system_prompt = build_system_prompt(
                mode=st.session_state.response_mode,
                rag_context=rag_context,
                web_context=web_context,
            )

            # 4. Stream response
            full_response = ""
            with st.spinner(""):
                try:
                    for token in stream_response(system_prompt, _active_messages()):
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                except RuntimeError as e:
                    st.error(f"Error: {e}")
                    logger.error("Stream error: %s", e)
                    return

            # Final render (no cursor)
            response_placeholder.markdown(full_response)

            # 5. Determine source
            if rag_context and web_context:
                source = "rag+web"
            elif rag_context:
                source = "rag"
            elif web_context:
                source = "web"
            else:
                source = "llm"

            pill_placeholder.markdown(_pills_html(source), unsafe_allow_html=True)

            # 6. Optional: show retrieved sources
            if rag_context or web_context:
                with st.expander("View retrieved sources", expanded=False):
                    if rag_context:
                        st.caption("From document")
                        st.code(rag_context[:2000], language=None)
                    if web_context:
                        st.caption("From web")
                        st.markdown(web_context[:1500])

            # 7. Persist to conversation
            _add_message("assistant", full_response, source)

        except RuntimeError as e:
            st.error(f"Error: {e}")
            logger.error("Pipeline error: %s", e)
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            logger.exception("Unexpected pipeline error")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:

        # ── Brand ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="sb-brand">
            <div class="sb-brand-name">ARIA</div>
            <div class="sb-brand-sub">Research Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        # ── New chat button ────────────────────────────────────────────────────
        st.markdown('<div style="padding: 0.7rem 1rem 0.3rem;">', unsafe_allow_html=True)
        if st.button("+ New conversation", key="new_conv", use_container_width=True):
            try:
                _create_new_conversation()
                st.session_state.pending_prompt = None
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Conversation history ───────────────────────────────────────────────
        st.markdown('<div class="sb-label">Recent</div>', unsafe_allow_html=True)

        try:
            conversations = st.session_state.conversations
            # Show most recent first
            sorted_convs = sorted(
                conversations.items(),
                key=lambda x: x[1].get("created_at", ""),
                reverse=True,
            )

            for conv_id, conv in sorted_convs:
                is_active = conv_id == st.session_state.active_conv_id
                active_cls = "active" if is_active else ""
                title = conv.get("title", "Untitled")
                time  = conv.get("created_at", "")
                msg_count = len(conv.get("messages", []))

                # Render as a clickable button styled like a list item
                clicked = st.button(
                    label=f"{title}",
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                )
                if clicked and not is_active:
                    st.session_state.active_conv_id = conv_id
                    st.session_state.pending_prompt = None
                    st.rerun()

        except Exception as e:
            st.caption(f"Could not load history: {e}")
            logger.error("Conversation history render failed: %s", e)

        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
        st.divider()

        # ── Document ───────────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Document</div>', unsafe_allow_html=True)

        with st.container():
            uploaded = st.file_uploader(
                "PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="pdf_uploader",
            )

        if uploaded:
            if uploaded.name != st.session_state.uploaded_filename:
                with st.spinner("Indexing document…"):
                    try:
                        vs = ingest_pdf(uploaded)
                        st.session_state.vector_store      = vs
                        st.session_state.uploaded_filename = uploaded.name
                        logger.info("Indexed: %s", uploaded.name)
                    except RuntimeError as e:
                        st.error(f"Indexing failed: {e}")
                        logger.error("Index error: %s", e)

            if st.session_state.vector_store:
                name  = st.session_state.uploaded_filename or ""
                short = (name[:24] + "…") if len(name) > 24 else name
                st.markdown(
                    f'<div class="status-row">'
                    f'<span class="status-dot sd-green"></span>{short}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="status-row">'
                '<span class="status-dot sd-gray"></span>'
                '<span style="color:#3d4255;">No document</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Response mode ──────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Response mode</div>', unsafe_allow_html=True)
        with st.container():
            mode = st.radio(
                "mode",
                options=["concise", "detailed"],
                index=0 if st.session_state.response_mode == "concise" else 1,
                format_func=str.capitalize,
                label_visibility="collapsed",
                horizontal=True,
                key="response_mode_radio",
            )
        st.session_state.response_mode = mode

        st.divider()

        # ── Web search ─────────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Web search</div>', unsafe_allow_html=True)
        with st.container():
            web_enabled = bool(TAVILY_API_KEY)
            web_on = st.toggle(
                "Live search",
                value=st.session_state.use_web_search and web_enabled,
                disabled=not web_enabled,
                key="web_toggle",
                label_visibility="collapsed",
            )
        st.session_state.use_web_search = web_on and web_enabled

        dot = "sd-green" if st.session_state.use_web_search else "sd-gray"
        label = "On" if st.session_state.use_web_search else ("No API key" if not web_enabled else "Off")
        st.markdown(
            f'<div class="status-row">'
            f'<span class="status-dot {dot}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── API status ─────────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">API status</div>', unsafe_allow_html=True)
        for name, key in [("Groq", GROQ_API_KEY), ("Tavily", TAVILY_API_KEY)]:
            dot   = "sd-green" if key else "sd-red"
            state = "connected" if key else "missing key"
            st.markdown(
                f'<div class="status-row">'
                f'<span class="status-dot {dot}"></span>'
                f'<span>{name}</span>'
                f'<span style="margin-left:auto;font-size:0.68rem;color:#374151;">{state}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────
def render_empty_state() -> None:
    """
    Shown when the active conversation has no messages.

    Suggestions are split into two groups:
      - doc_suggestions: require a PDF to be loaded. Blocked otherwise.
      - general_suggestions: always available, no document needed.
    """
    has_doc = st.session_state.vector_store is not None

    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-greeting">What are you researching today?</div>
        <div class="empty-sub">Upload a paper or ask anything — ARIA will find the answer.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Document-specific suggestions ────────────────────────────────────────
    doc_suggestions = [
        ("Summarise the paper",     "Give me a summary of the key contributions and findings."),
        ("Explain the methodology", "Walk me through the methodology used in this paper."),
        ("Key limitations",         "What are the limitations and future work mentioned in this paper?"),
        ("Related work",            "What prior work does this paper build on?"),
    ]

    # ── General suggestions (no document required) ────────────────────────────
    general_suggestions = [
        ("Latest LLM research",  "What are the latest trends in large language model research?"),
        ("RAG vs Fine-tuning",   "What are the key differences between RAG and fine-tuning for LLMs?"),
    ]

    # ── Section: From your document ──────────────────────────────────────────
    if has_doc:
        doc_name = st.session_state.uploaded_filename or "document"
        short    = (doc_name[:32] + "…") if len(doc_name) > 32 else doc_name
        st.markdown(
            f'<div style="font-size:0.7rem;font-weight:600;color:#4b5563;'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.6rem;'
            f'font-family:Inter,sans-serif;">From · {short}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, (label, prompt) in enumerate(doc_suggestions):
            with cols[i % 2]:
                if st.button(label, key=f"doc_sug_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()
    else:
        # Show document suggestions as locked/greyed with a clear message
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:600;color:#374151;'
            'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.6rem;'
            'font-family:Inter,sans-serif;">From document — upload a PDF to unlock</div>',
            unsafe_allow_html=True,
        )
        # Render greyed-out non-clickable chips
        chips_html = "".join(
            f'<span style="display:inline-block;background:#0d0f14;border:1px solid #1a1d27;'
            f'color:#2a2f47;font-size:0.78rem;font-family:Inter,sans-serif;font-weight:500;'
            f'border-radius:7px;padding:0.4rem 0.9rem;margin:0 6px 6px 0;'
            f'cursor:not-allowed;user-select:none;">{label}</span>'
            for label, _ in doc_suggestions
        )
        st.markdown(
            f'<div style="margin-bottom:0.4rem;">{chips_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.75rem;color:#374151;font-family:Inter,sans-serif;'
            'margin-bottom:1rem;">Upload a PDF in the sidebar to ask document-specific questions.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)

    # ── Section: General questions ────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:600;color:#4b5563;'
        'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.6rem;'
        'font-family:Inter,sans-serif;">General</div>',
        unsafe_allow_html=True,
    )
    cols2 = st.columns(2)
    for i, (label, prompt) in enumerate(general_suggestions):
        with cols2[i % 2]:
            if st.button(label, key=f"gen_sug_{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Message history renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_messages() -> None:
    """Render the full conversation history for the active session."""
    try:
        for msg in _active_messages():
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("source"):
                    st.markdown(_pills_html(msg["source"]), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not render messages: {e}")
        logger.error("Message render failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Chat page
# ─────────────────────────────────────────────────────────────────────────────
def chat_page() -> None:
    # Page header — shows current conversation title
    try:
        conv  = st.session_state.conversations.get(st.session_state.active_conv_id, {})
        title = conv.get("title", "ARIA")
        msgs  = conv.get("messages", [])
    except Exception:
        title = "ARIA"
        msgs  = []

    if not msgs:
        # Fresh conversation — show big header + suggestions
        st.markdown("""
        <div class="page-header">
            <div class="page-title">ARIA</div>
            <div class="page-subtitle">AI Research Intelligence Assistant &nbsp;·&nbsp;
            Llama 3.3 70B &nbsp;·&nbsp; Groq</div>
        </div>
        """, unsafe_allow_html=True)
        render_empty_state()
    else:
        # Active conversation — compact header
        st.markdown(f"""
        <div style="padding: 1.2rem 0 0.5rem; border-bottom: 1px solid #1c1f2a; margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; font-weight: 500; color: #6b7280;
                        font-family: Inter, sans-serif; letter-spacing: 0.01em;">
                {title}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render message history
    render_messages()

    # Handle pending prompt (from suggestion card click)
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

        # Safety guard: doc-specific prompts require a loaded document
        doc_keywords = ["summary", "methodology", "limitations", "prior work", "findings", "paper"]
        needs_doc = any(kw in prompt.lower() for kw in doc_keywords)
        if needs_doc and not st.session_state.vector_store:
            st.info("Please upload a PDF in the sidebar before asking document-specific questions.")
            logger.info("Blocked doc suggestion — no document loaded.")
        else:
            _add_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            run_pipeline(prompt)

    # Chat input
    if prompt := st.chat_input("Ask ARIA…"):
        _add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        run_pipeline(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Instructions page
# ─────────────────────────────────────────────────────────────────────────────
def instructions_page() -> None:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Setup</div>
        <div class="page-subtitle">Get ARIA running locally or deployed in minutes</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
### Installation

```bash
pip install -r requirements.txt
```

---

### API Keys

| Key | Provider | Free tier |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | Yes |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | Yes |

---

### Run locally

```bash
GROQ_API_KEY=gsk_... TAVILY_API_KEY=tvly_... streamlit run app.py
```

---

### Deploy on Streamlit Cloud

1. Push the project folder to a **GitHub repository** (exclude `venv/`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → connect repo
3. Set entry point: `app.py`
4. Add secrets under **Settings → Secrets**:

```toml
GROQ_API_KEY    = "gsk_..."
TAVILY_API_KEY  = "tvly-..."
```

5. Click **Deploy** — done.

---

### Architecture

```
project/
├── config/
│   └── config.py           ← API keys & tuneable settings (env vars)
├── models/
│   ├── llm.py              ← Groq Llama 3.3 70B, mode-aware token limits
│   └── embeddings.py       ← HuggingFace all-MiniLM-L6-v2 (local, no key)
├── utils/
│   ├── rag.py              ← PDF → chunk → FAISS index → retrieve
│   ├── web_search.py       ← Tavily real-time search
│   └── prompt_builder.py   ← Dynamic system prompt (persona + mode + context)
├── app.py                  ← Streamlit UI (this file)
└── requirements.txt
```

---

### How it works

1. **Upload a PDF** → ARIA chunks it, embeds it with MiniLM, indexes with FAISS
2. **Ask a question** → ARIA retrieves the top relevant chunks (RAG)
3. **Web search** → If enabled, Tavily fetches real-time results
4. **LLM** → Groq Llama 3.3 70B synthesises a streamed answer from all context
5. **Source pills** show which context was used: Document / Web / LLM
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    try:
        init_session_state()
        render_sidebar()

        # Navigation at the bottom of sidebar
        with st.sidebar:
            st.divider()
            st.markdown('<div class="sb-label">Pages</div>', unsafe_allow_html=True)
            page = st.radio(
                "page",
                options=["Chat", "Setup"],
                label_visibility="collapsed",
                key="nav_radio",
            )

        if page == "Chat":
            chat_page()
        else:
            instructions_page()

    except Exception as e:
        st.error(f"Application error: {e}")
        logger.exception("Fatal application error")


if __name__ == "__main__":
    main()
