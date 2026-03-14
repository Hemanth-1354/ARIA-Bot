"""
app.py — ARIA: AI Research Intelligence Assistant

Features:
  · ChatGPT-style UI (no robot avatars, clean white/dark layout)
  · Streaming LLM responses (token by token)
  · Conversation history sidebar with named sessions
  · Multi-document RAG — upload N PDFs, search across all simultaneously
  · Paper comparison mode — pick any two loaded docs, ARIA compares them
  · Smart follow-up questions after every answer
  · Live web search via Tavily
  · Concise / Detailed response mode
  · Full try/except error handling on every function
"""

import os
import sys
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Generator

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.llm import get_chatgroq_model
from utils.rag import (
    ingest_pdf,
    merge_indexes,
    retrieve_relevant_chunks,
    retrieve_per_doc,
)
from utils.web_search import search_web
from utils.prompt_builder import build_system_prompt
from utils.followup import generate_followups
from config.config import GROQ_API_KEY, TAVILY_API_KEY

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — ChatGPT-style ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #212121;
    color: #ececec;
}

/* ── Sidebar — darker panel ── */
section[data-testid="stSidebar"] {
    background: #171717;
    border-right: 1px solid #2a2a2a;
    width: 260px !important;
}
section[data-testid="stSidebar"] > div { padding: 0; }

/* ── Main content area ── */
.main .block-container {
    max-width: 720px;
    padding: 0 1.5rem 7rem 1.5rem;
    margin: 0 auto;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar brand ── */
.sb-brand {
    padding: 1rem 0.9rem 0.85rem;
    border-bottom: 1px solid #2a2a2a;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sb-brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: #ececec;
    letter-spacing: -0.01em;
}
.sb-brand-sub {
    font-size: 0.6rem;
    color: #555;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 1px;
}

/* ── Sidebar section label ── */
.sb-label {
    font-size: 0.6rem;
    font-weight: 600;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.85rem 0.9rem 0.3rem;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar buttons (new chat, clear) ── */
div[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    color: #9a9a9a !important;
    font-size: 0.78rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    border-radius: 6px !important;
    padding: 0.38rem 0.8rem !important;
    width: 100%;
    text-align: left !important;
    transition: all 0.12s !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #2a2a2a !important;
    color: #ececec !important;
    border-color: #3a3a3a !important;
}

/* ── Conversation history items ── */
.conv-btn button {
    background: transparent !important;
    border: none !important;
    color: #9a9a9a !important;
    font-size: 0.78rem !important;
    text-align: left !important;
    border-radius: 6px !important;
    padding: 0.38rem 0.75rem !important;
    transition: background 0.1s !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.conv-btn button:hover {
    background: #2a2a2a !important;
    color: #ececec !important;
}
.conv-active button {
    background: #2a2a2a !important;
    color: #ececec !important;
}

/* ── Status dot ── */
.dot { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:5px; }
.dot-on  { background:#10a37f; }
.dot-off { background:#444; }
.dot-red { background:#ef4444; }

/* ── Status row text ── */
.status-row {
    font-size: 0.73rem;
    color: #666;
    padding: 0.2rem 0.9rem;
    font-family: 'Inter', sans-serif;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

/* ── Main: GPT-style empty state ── */
.empty-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 0 2rem;
    text-align: center;
}
.empty-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ececec;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}
.empty-sub {
    font-size: 0.88rem;
    color: #666;
    margin-bottom: 2rem;
}

/* ── Suggestion grid ── */
.sug-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; width: 100%; max-width: 580px; }
.sug-card {
    background: #2a2a2a;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    cursor: pointer;
    transition: background 0.12s, border-color 0.12s;
    text-align: left;
}
.sug-card:hover { background: #333; border-color: #444; }
.sug-title { font-size: 0.82rem; font-weight: 500; color: #ececec; margin-bottom: 2px; }
.sug-sub   { font-size: 0.72rem; color: #666; line-height: 1.35; }

/* ── Messages ── */
/* Hide default Streamlit chat avatars */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] { display: none !important; }

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 1.1rem 0 !important;
    border-bottom: 1px solid #2a2a2a !important;
    max-width: 100%;
}
[data-testid="stChatMessage"]:last-child { border-bottom: none !important; }

/* User message — slightly inset */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
div[class*="stChatMessage"][data-testid*="user"] {
    background: #2a2a2a !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.5rem !important;
}

/* ── Source pills ── */
.spill {
    display: inline-flex; align-items: center;
    font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px;
    margin-right: 4px; margin-top: 10px;
    font-family: 'Inter', sans-serif;
}
.sp-doc { background: #0a2b1e; color: #10a37f; border: 1px solid #134d35; }
.sp-web { background: #0a1a2e; color: #60a5fa; border: 1px solid #1e3a5f; }
.sp-llm { background: #1a1a1a; color: #888;    border: 1px solid #333; }
.sp-cmp { background: #1a0e2e; color: #c084fc; border: 1px solid #3b1f5e; }

/* ── Follow-up chips ── */
.followup-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 12px; padding-top: 10px;
    border-top: 1px solid #2a2a2a;
}
.followup-label {
    font-size: 0.62rem; color: #555; letter-spacing: 0.08em;
    text-transform: uppercase; width: 100%; margin-bottom: 2px;
    font-family: 'Inter', sans-serif;
}

/* ── Suggestion / follow-up buttons ── */
.stButton > button {
    background: #2a2a2a !important;
    border: 1px solid #333 !important;
    color: #9a9a9a !important;
    font-size: 0.78rem !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
    padding: 0.38rem 0.85rem !important;
    transition: all 0.12s !important;
    white-space: nowrap;
}
.stButton > button:hover {
    background: #333 !important;
    border-color: #444 !important;
    color: #ececec !important;
}

/* ── Comparison mode banner ── */
.cmp-banner {
    background: #1a0e2e;
    border: 1px solid #3b1f5e;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: #c084fc;
    font-family: 'Inter', sans-serif;
    margin-bottom: 1rem;
}

/* ── Doc chip (loaded doc badge) ── */
.doc-chip {
    display: inline-flex; align-items: center; gap: 4px;
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 999px; padding: 3px 9px;
    font-size: 0.7rem; color: #9a9a9a;
    font-family: 'Inter', sans-serif;
    margin: 2px 3px 2px 0;
}
.doc-chip-active { border-color: #10a37f; color: #10a37f; background: #0a2b1e; }

/* ── Chat input ── */
[data-testid="stBottom"] {
    background: linear-gradient(to top, #212121 75%, transparent) !important;
    padding: 0.75rem 0 0.5rem !important;
}
[data-testid="stChatInput"] {
    background: #2a2a2a !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #555 !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: #ececec !important;
    background: transparent !important;
}

/* ── Expander ── */
details summary {
    font-size: 0.72rem !important; color: #555 !important;
    font-family: 'Inter', sans-serif !important;
}
details summary:hover { color: #9a9a9a !important; }

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; font-size: 0.82rem !important; }

/* ── Radio ── */
.stRadio > label { display: none !important; }
.stRadio > div { display: flex !important; gap: 6px !important; flex-direction: row !important; }
.stRadio > div > label {
    background: #1a1a1a !important; border: 1px solid #2a2a2a !important;
    border-radius: 6px !important; padding: 0.28rem 0.75rem !important;
    font-size: 0.74rem !important; color: #666 !important;
    cursor: pointer !important; transition: all 0.12s !important;
}
.stRadio > div > label:has(input:checked) {
    background: #2a2a2a !important; border-color: #555 !important; color: #ececec !important;
}

/* ── Toggle ── */
.stToggle > label { font-size: 0.76rem !important; color: #666 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1a1a1a !important; border: 1px dashed #2a2a2a !important;
    border-radius: 8px !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] { font-size: 0.8rem !important; }

/* ── Divider ── */
hr { border-color: #2a2a2a !important; margin: 0.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #444; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    """Bootstrap all session state keys with safe defaults."""
    defaults: dict = {
        # Conversations: {id: {title, messages, created_at}}
        "conversations":     {},
        "active_conv_id":    None,
        # Multi-document store: {filename: FAISS index}
        "doc_indexes":       {},
        # Merged index across all docs (rebuilt on upload/remove)
        "merged_index":      None,
        # Comparison mode
        "compare_mode":      False,
        "compare_doc_a":     None,
        "compare_doc_b":     None,
        # UI settings
        "response_mode":     "detailed",
        "use_web_search":    True,
        # Pending prompt from button click
        "pending_prompt":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Ensure at least one active conversation exists
    if (
        not st.session_state.active_conv_id
        or st.session_state.active_conv_id not in st.session_state.conversations
    ):
        _new_conversation()


# ─────────────────────────────────────────────────────────────────────────────
# Conversation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _new_conversation(title: str = "New chat") -> str:
    """Create a fresh conversation and set it as active."""
    try:
        cid = str(uuid.uuid4())[:8]
        st.session_state.conversations[cid] = {
            "title":      title,
            "messages":   [],
            "created_at": datetime.now().strftime("%H:%M"),
        }
        st.session_state.active_conv_id = cid
        logger.info("New conversation: %s", cid)
        return cid
    except Exception as e:
        logger.error("Failed to create conversation: %s", e)
        raise RuntimeError(str(e)) from e


def _active_messages() -> list:
    try:
        return st.session_state.conversations[st.session_state.active_conv_id]["messages"]
    except (KeyError, TypeError):
        return []


def _add_message(role: str, content: str, **meta) -> None:
    """Append a message dict to the active conversation."""
    try:
        cid = st.session_state.active_conv_id
        msg = {"role": role, "content": content, **meta}
        st.session_state.conversations[cid]["messages"].append(msg)
        # Auto-title from first user message
        if role == "user" and len(st.session_state.conversations[cid]["messages"]) == 1:
            title = content[:40] + ("…" if len(content) > 40 else "")
            st.session_state.conversations[cid]["title"] = title
    except Exception as e:
        logger.error("Failed to add message: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Document management
# ─────────────────────────────────────────────────────────────────────────────
def _rebuild_merged_index() -> None:
    """Rebuild the merged FAISS index from all loaded doc indexes."""
    try:
        indexes = list(st.session_state.doc_indexes.values())
        st.session_state.merged_index = merge_indexes(indexes) if indexes else None
        logger.info("Merged index rebuilt from %d docs.", len(indexes))
    except RuntimeError as e:
        st.error(f"Could not merge document indexes: {e}")
        logger.error("Merge failed: %s", e)


def _remove_document(name: str) -> None:
    """Remove a document from the multi-doc store and rebuild merged index."""
    try:
        if name in st.session_state.doc_indexes:
            del st.session_state.doc_indexes[name]
            _rebuild_merged_index()
            # Clear comparison selection if removed doc was selected
            if st.session_state.compare_doc_a == name:
                st.session_state.compare_doc_a = None
            if st.session_state.compare_doc_b == name:
                st.session_state.compare_doc_b = None
            logger.info("Removed document: %s", name)
    except Exception as e:
        logger.error("Failed to remove document '%s': %s", name, e)


# ─────────────────────────────────────────────────────────────────────────────
# Source pills HTML
# ─────────────────────────────────────────────────────────────────────────────
def _pills(source: str) -> str:
    m = {
        "rag":     '<span class="spill sp-doc">Document</span>',
        "web":     '<span class="spill sp-web">Web</span>',
        "llm":     '<span class="spill sp-llm">LLM</span>',
        "rag+web": '<span class="spill sp-doc">Document</span><span class="spill sp-web">Web</span>',
        "compare": '<span class="spill sp-cmp">Comparison</span>',
    }
    return m.get(source, "")


# ─────────────────────────────────────────────────────────────────────────────
# LLM — streaming
# ─────────────────────────────────────────────────────────────────────────────
def _stream(system_prompt: str, messages: list) -> Generator:
    """
    Stream tokens from Groq.

    Yields:
        str token chunks.

    Raises:
        RuntimeError: on init or streaming failure.
    """
    try:
        model = get_chatgroq_model(mode=st.session_state.response_mode)
        formatted = [SystemMessage(content=system_prompt)]
        for m in messages:
            if m["role"] == "user":
                formatted.append(HumanMessage(content=m["content"]))
            else:
                formatted.append(AIMessage(content=m["content"]))
        for chunk in model.stream(formatted):
            try:
                yield chunk.content
            except Exception as ce:
                logger.warning("Chunk decode error: %s", ce)
    except RuntimeError:
        raise
    except Exception as e:
        logger.error("Streaming failed: %s", e)
        raise RuntimeError(f"Streaming error: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(prompt: str) -> None:
    """
    Normal pipeline: merged RAG + optional web search + streamed LLM response.
    Renders inside st.chat_message("assistant").
    """
    with st.chat_message("assistant"):
        resp_ph = st.empty()
        pill_ph = st.empty()

        try:
            # 1. RAG from merged index
            rag_context = ""
            if st.session_state.merged_index:
                try:
                    rag_context = retrieve_relevant_chunks(
                        st.session_state.merged_index, prompt
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

            # 3. System prompt
            system_prompt = build_system_prompt(
                mode=st.session_state.response_mode,
                rag_context=rag_context,
                web_context=web_context,
            )

            # 4. Stream
            full_response = ""
            with st.spinner(""):
                try:
                    for token in _stream(system_prompt, _active_messages()):
                        full_response += token
                        resp_ph.markdown(full_response + "▌")
                except RuntimeError as e:
                    st.error(f"Error: {e}")
                    return

            resp_ph.markdown(full_response)

            # 5. Source
            if rag_context and web_context:
                source = "rag+web"
            elif rag_context:
                source = "rag"
            elif web_context:
                source = "web"
            else:
                source = "llm"

            pill_ph.markdown(_pills(source), unsafe_allow_html=True)

            # 6. Retrieved sources expander
            if rag_context or web_context:
                with st.expander("View sources", expanded=False):
                    if rag_context:
                        st.caption("From documents")
                        st.code(rag_context[:2000], language=None)
                    if web_context:
                        st.caption("From web")
                        st.markdown(web_context[:1500])

            # 7. Follow-up suggestions
            followups = generate_followups(prompt, full_response)
            if followups:
                _render_followup_chips(followups)

            # 8. Persist
            _add_message("assistant", full_response, source=source)

        except RuntimeError as e:
            st.error(f"Error: {e}")
            logger.error("Pipeline error: %s", e)
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            logger.exception("Unexpected pipeline error")


def run_comparison_pipeline(prompt: str) -> None:
    """
    Comparison pipeline: retrieves from two selected docs separately,
    builds a structured compare prompt, streams the side-by-side answer.
    """
    doc_a = st.session_state.compare_doc_a
    doc_b = st.session_state.compare_doc_b

    if not doc_a or not doc_b:
        st.warning("Select two documents in the sidebar for comparison.")
        return

    with st.chat_message("assistant"):
        resp_ph = st.empty()
        pill_ph = st.empty()

        try:
            # Retrieve separately from each doc
            a_index = st.session_state.doc_indexes.get(doc_a)
            b_index = st.session_state.doc_indexes.get(doc_b)

            context_a, context_b = "", ""

            if a_index:
                try:
                    context_a = retrieve_relevant_chunks(a_index, prompt)
                except RuntimeError as e:
                    logger.warning("Retrieval failed for doc A: %s", e)

            if b_index:
                try:
                    context_b = retrieve_relevant_chunks(b_index, prompt)
                except RuntimeError as e:
                    logger.warning("Retrieval failed for doc B: %s", e)

            # Build comparison system prompt
            compare_prompt = f"""You are ARIA, an expert research assistant.
The user wants to COMPARE two research papers on the following question.

PAPER A — {doc_a}:
{context_a[:1500] if context_a else "No relevant content found."}

PAPER B — {doc_b}:
{context_b[:1500] if context_b else "No relevant content found."}

Instructions:
- Structure your answer clearly: first discuss Paper A, then Paper B, then a direct comparison
- Use headers: **Paper A**, **Paper B**, **Comparison**
- Be specific, cite page numbers or chunk references when possible
- Response mode: {"concise — keep it short" if st.session_state.response_mode == "concise" else "detailed — be thorough"}
"""

            # Stream comparison response
            full_response = ""
            with st.spinner(""):
                try:
                    model = get_chatgroq_model(mode=st.session_state.response_mode)
                    lc_messages = [
                        SystemMessage(content=compare_prompt),
                        HumanMessage(content=prompt),
                    ]
                    for chunk in model.stream(lc_messages):
                        try:
                            full_response += chunk.content
                            resp_ph.markdown(full_response + "▌")
                        except Exception as ce:
                            logger.warning("Chunk error: %s", ce)
                except RuntimeError as e:
                    st.error(f"Error: {e}")
                    return

            resp_ph.markdown(full_response)
            pill_ph.markdown(_pills("compare"), unsafe_allow_html=True)

            # Follow-ups
            followups = generate_followups(prompt, full_response)
            if followups:
                _render_followup_chips(followups)

            _add_message("assistant", full_response, source="compare")

        except RuntimeError as e:
            st.error(f"Comparison error: {e}")
            logger.error("Comparison pipeline error: %s", e)
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            logger.exception("Unexpected comparison error")


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up chips
# ─────────────────────────────────────────────────────────────────────────────
def _render_followup_chips(questions: List[str]) -> None:
    """Render follow-up question chips below an assistant message."""
    try:
        st.markdown(
            '<div class="followup-label">Suggested follow-ups</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(questions))
        for i, (col, q) in enumerate(zip(cols, questions)):
            with col:
                if st.button(q, key=f"fu_{uuid.uuid4().hex[:6]}", use_container_width=True):
                    st.session_state.pending_prompt = q
                    st.rerun()
    except Exception as e:
        logger.warning("Could not render follow-up chips: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:

        # ── Brand ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="sb-brand">
            <div>
                <div class="sb-brand-name">ARIA</div>
                <div class="sb-brand-sub">Research Assistant</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── New chat ───────────────────────────────────────────────────────────
        st.markdown('<div style="padding:0.6rem 0.9rem 0.2rem;">', unsafe_allow_html=True)
        if st.button("+ New chat", key="new_conv", use_container_width=True):
            try:
                _new_conversation()
                st.session_state.pending_prompt = None
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Conversation history ───────────────────────────────────────────────
        st.markdown('<div class="sb-label">Recent</div>', unsafe_allow_html=True)

        try:
            sorted_convs = sorted(
                st.session_state.conversations.items(),
                key=lambda x: x[1].get("created_at", ""),
                reverse=True,
            )
            for cid, conv in sorted_convs:
                is_active = cid == st.session_state.active_conv_id
                label = conv.get("title", "Untitled")
                css_class = "conv-active" if is_active else "conv-btn"
                with st.container():
                    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                    if st.button(label, key=f"conv_{cid}", use_container_width=True):
                        if not is_active:
                            st.session_state.active_conv_id = cid
                            st.session_state.pending_prompt = None
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"History error: {e}")
            logger.error("Conversation history render failed: %s", e)

        st.divider()

        # ── Documents (multi-upload) ───────────────────────────────────────────
        st.markdown('<div class="sb-label">Documents</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="multi_uploader",
        )

        if uploaded_files:
            for uf in uploaded_files:
                if uf.name not in st.session_state.doc_indexes:
                    with st.spinner(f"Indexing {uf.name[:20]}…"):
                        try:
                            vs = ingest_pdf(uf)
                            st.session_state.doc_indexes[uf.name] = vs
                            _rebuild_merged_index()
                            logger.info("Indexed: %s", uf.name)
                        except RuntimeError as e:
                            st.error(f"Failed to index {uf.name}: {e}")
                            logger.error("Index failed for %s: %s", uf.name, e)

        # Loaded docs list with remove buttons
        if st.session_state.doc_indexes:
            for name in list(st.session_state.doc_indexes.keys()):
                short = (name[:22] + "…") if len(name) > 22 else name
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f'<div class="status-row">'
                        f'<span class="dot dot-on"></span>{short}</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("✕", key=f"rm_{name}", help=f"Remove {name}"):
                        _remove_document(name)
                        st.rerun()
        else:
            st.markdown(
                '<div class="status-row">'
                '<span class="dot dot-off"></span>'
                '<span style="color:#555;">No documents loaded</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Comparison mode ────────────────────────────────────────────────────
        doc_names = list(st.session_state.doc_indexes.keys())
        has_two   = len(doc_names) >= 2

        st.markdown('<div class="sb-label">Compare mode</div>', unsafe_allow_html=True)

        compare_on = st.toggle(
            "Paper comparison",
            value=st.session_state.compare_mode and has_two,
            disabled=not has_two,
            label_visibility="collapsed",
            key="compare_toggle",
        )
        st.session_state.compare_mode = compare_on and has_two

        if not has_two:
            st.caption("Upload at least 2 PDFs to compare.")
        elif compare_on:
            st.markdown(
                '<div style="font-size:0.7rem;color:#666;padding:0.2rem 0.9rem 0.3rem;">'
                'Select two papers:</div>',
                unsafe_allow_html=True,
            )
            doc_a = st.selectbox(
                "Paper A", options=doc_names, key="sel_a",
                label_visibility="collapsed",
            )
            remaining = [d for d in doc_names if d != doc_a]
            doc_b = st.selectbox(
                "Paper B", options=remaining, key="sel_b",
                label_visibility="collapsed",
            )
            st.session_state.compare_doc_a = doc_a
            st.session_state.compare_doc_b = doc_b

        st.divider()

        # ── Response mode ──────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Response mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "mode",
            options=["concise", "detailed"],
            index=0 if st.session_state.response_mode == "concise" else 1,
            format_func=str.capitalize,
            label_visibility="collapsed",
            horizontal=True,
            key="mode_radio",
        )
        st.session_state.response_mode = mode

        st.divider()

        # ── Web search ─────────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Web search</div>', unsafe_allow_html=True)
        web_ok = bool(TAVILY_API_KEY)
        web_on = st.toggle(
            "Live web search",
            value=st.session_state.use_web_search and web_ok,
            disabled=not web_ok,
            label_visibility="collapsed",
            key="web_toggle",
        )
        st.session_state.use_web_search = web_on and web_ok
        dot   = "dot-on" if st.session_state.use_web_search else "dot-off"
        label = "On" if st.session_state.use_web_search else ("No API key" if not web_ok else "Off")
        st.markdown(
            f'<div class="status-row"><span class="dot {dot}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── API status ─────────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Status</div>', unsafe_allow_html=True)
        for svc, key in [("Groq", GROQ_API_KEY), ("Tavily", TAVILY_API_KEY)]:
            dot   = "dot-on" if key else "dot-red"
            state = "connected" if key else "missing key"
            st.markdown(
                f'<div class="status-row">'
                f'<span class="dot {dot}"></span>{svc} — {state}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────
def render_empty_state() -> None:
    """ChatGPT-style empty state with suggestion cards."""
    has_doc = bool(st.session_state.doc_indexes)

    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-logo">ARIA</div>
        <div class="empty-sub">Ask anything about your research papers, or pick a suggestion below.</div>
    </div>
    """, unsafe_allow_html=True)

    # Doc-specific suggestions (locked without docs)
    doc_suggestions = [
        ("Summarise the paper",     "Give me a summary of the key contributions and findings."),
        ("Explain the methodology", "Walk me through the methodology used in this paper."),
        ("Key limitations",         "What are the limitations and future work mentioned?"),
        ("Related work",            "What prior work does this paper build on?"),
    ]
    # General suggestions (always available)
    general_suggestions = [
        ("Latest LLM research",  "What are the latest trends in large language model research?"),
        ("RAG vs Fine-tuning",   "What are the differences between RAG and fine-tuning?"),
    ]

    if has_doc:
        cols = st.columns(2)
        for i, (label, prompt) in enumerate(doc_suggestions + general_suggestions):
            with cols[i % 2]:
                if st.button(label, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()
    else:
        # Show doc suggestions greyed out
        chips = "".join(
            f'<span style="display:inline-block;background:#1a1a1a;border:1px solid #2a2a2a;'
            f'color:#3a3a3a;font-size:0.78rem;font-family:Inter,sans-serif;font-weight:500;'
            f'border-radius:8px;padding:0.38rem 0.85rem;margin:3px 4px 3px 0;'
            f'cursor:not-allowed;">{label}</span>'
            for label, _ in doc_suggestions
        )
        st.markdown(
            f'<div style="text-align:center;margin-bottom:0.3rem;">'
            f'<div style="font-size:0.62rem;color:#555;letter-spacing:0.08em;'
            f'text-transform:uppercase;margin-bottom:0.5rem;font-family:Inter,sans-serif;">'
            f'Document questions — upload a PDF to unlock</div>'
            f'{chips}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (label, prompt) in enumerate(general_suggestions):
            with cols[i % 2]:
                if st.button(label, key=f"gen_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Message renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_messages() -> None:
    """Render full conversation history with source pills."""
    try:
        for msg in _active_messages():
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("source"):
                    st.markdown(_pills(msg["source"]), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not render messages: {e}")
        logger.error("Message render failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Chat page
# ─────────────────────────────────────────────────────────────────────────────
def chat_page() -> None:
    try:
        conv  = st.session_state.conversations.get(st.session_state.active_conv_id, {})
        msgs  = conv.get("messages", [])
        title = conv.get("title", "ARIA")
    except Exception:
        msgs, title = [], "ARIA"

    # Compact header for active conversations
    if msgs:
        st.markdown(
            f'<div style="padding:1rem 0 0.6rem;border-bottom:1px solid #2a2a2a;'
            f'margin-bottom:0.8rem;font-size:0.78rem;font-weight:500;color:#666;'
            f'font-family:Inter,sans-serif;">{title}</div>',
            unsafe_allow_html=True,
        )

    # Comparison mode banner
    if st.session_state.compare_mode and st.session_state.compare_doc_a and st.session_state.compare_doc_b:
        a = st.session_state.compare_doc_a
        b = st.session_state.compare_doc_b
        a_short = (a[:25] + "…") if len(a) > 25 else a
        b_short = (b[:25] + "…") if len(b) > 25 else b
        st.markdown(
            f'<div class="cmp-banner">Comparison mode active — '
            f'<strong>{a_short}</strong> vs <strong>{b_short}</strong></div>',
            unsafe_allow_html=True,
        )

    # Empty state
    if not msgs and not st.session_state.pending_prompt:
        render_empty_state()

    # Message history
    render_messages()

    # Handle pending prompt (from suggestion / follow-up button)
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

        # Guard: doc-specific prompts need at least one document
        doc_kw = ["summary", "methodology", "limitations", "prior work", "findings", "paper"]
        needs_doc = any(kw in prompt.lower() for kw in doc_kw)
        if needs_doc and not st.session_state.doc_indexes:
            st.info("Upload a PDF first to ask document-specific questions.")
        else:
            _add_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            if st.session_state.compare_mode:
                run_comparison_pipeline(prompt)
            else:
                run_pipeline(prompt)

    # Chat input
    if prompt := st.chat_input(
        "Compare papers, ask about your documents, or search the web…"
        if st.session_state.compare_mode
        else "Ask ARIA…"
    ):
        _add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        if st.session_state.compare_mode:
            run_comparison_pipeline(prompt)
        else:
            run_pipeline(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Setup page
# ─────────────────────────────────────────────────────────────────────────────
def setup_page() -> None:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem;">
        <div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
                    color:#ececec;letter-spacing:-0.03em;">Setup</div>
        <div style="font-size:0.83rem;color:#666;margin-top:4px;font-family:Inter,sans-serif;">
            Get ARIA running in minutes
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
### Installation
```bash
pip install -r requirements.txt
```

### API Keys

| Key | Where | Free |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | Yes |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | Yes |

### Run locally
```bash
GROQ_API_KEY=gsk_... TAVILY_API_KEY=tvly_... streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub → connect at [share.streamlit.io](https://share.streamlit.io)
2. Entry point: `app.py`
3. Add secrets:
```toml
GROQ_API_KEY   = "gsk_..."
TAVILY_API_KEY = "tvly-..."
```

### Features
- **Multi-document RAG** — upload multiple PDFs, ARIA searches across all simultaneously
- **Paper comparison** — enable comparison mode, select two papers, ask any question
- **Follow-up questions** — ARIA suggests 3 smart follow-ups after every answer
- **Streaming** — responses render token by token
- **Conversation history** — all sessions saved in the sidebar

### Architecture
```
project/
├── config/config.py         ← API keys & settings (env vars)
├── models/llm.py            ← Groq Llama 3.3 70B, mode-aware
├── models/embeddings.py     ← HuggingFace all-MiniLM-L6-v2
├── utils/rag.py             ← Multi-doc: ingest, merge, retrieve, compare
├── utils/followup.py        ← Follow-up question generation
├── utils/web_search.py      ← Tavily real-time search
├── utils/prompt_builder.py  ← Dynamic system prompt builder
├── app.py                   ← Streamlit UI
└── requirements.txt
```
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    try:
        init_session_state()
        render_sidebar()

        with st.sidebar:
            st.divider()
            st.markdown('<div class="sb-label">Pages</div>', unsafe_allow_html=True)
            page = st.radio(
                "page", options=["Chat", "Setup"],
                label_visibility="collapsed", key="nav",
            )

        if page == "Chat":
            chat_page()
        else:
            setup_page()

    except Exception as e:
        st.error(f"Application error: {e}")
        logger.exception("Fatal error")


if __name__ == "__main__":
    main()
