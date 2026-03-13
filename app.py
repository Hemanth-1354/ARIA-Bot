"""
app.py — ARIA: AI Research Intelligence Assistant
Clean, minimal, component-based Streamlit UI.
"""

import os
import sys
import logging

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.llm import get_chatgroq_model
from utils.rag import ingest_pdf, retrieve_relevant_chunks
from utils.web_search import search_web
from utils.prompt_builder import build_system_prompt
from config.config import GROQ_API_KEY, TAVILY_API_KEY

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA — Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

section[data-testid="stSidebar"] {
    background-color: #0f1117;
    border-right: 1px solid #1e2130;
}
.main .block-container {
    max-width: 780px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}
.aria-wordmark {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1;
}
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
.dot-on  { background: #34d399; }
.dot-off { background: #374151; }
.pill {
    display: inline-block;
    font-size: 0.65rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 999px;
    margin-right: 4px;
}
.pill-doc { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.pill-web { background: #0c1a2e; color: #60a5fa; border: 1px solid #1e3a5f; }
.pill-llm { background: #1a0e2e; color: #a78bfa; border: 1px solid #3b2a5e; }
.section-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4b5563;
    font-family: 'Inter', sans-serif;
    margin-bottom: 6px;
}
.stButton > button {
    background: #1a1d27;
    border: 1px solid #2d3148;
    color: #9ca3af;
    font-size: 0.8rem;
    font-family: 'Inter', sans-serif;
    border-radius: 6px;
    padding: 0.4rem 0.9rem;
    transition: border-color 0.15s, color 0.15s;
}
.stButton > button:hover {
    border-color: #60a5fa;
    color: #e5e7eb;
    background: #1e2235;
}
[data-testid="stChatInput"] textarea { font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important; }
[data-testid="stChatMessage"] { padding: 0.9rem 1rem !important; }
.stAlert { font-size: 0.85rem; border-radius: 8px; }
hr { border-color: #1e2130 !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "messages":          [],
        "vector_store":      None,
        "uploaded_filename": None,
        "response_mode":     "detailed",
        "use_web_search":    True,
        "pending_prompt":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def source_pills(source: str) -> str:
    """Return HTML source pills for a given source label."""
    pills = {
        "rag":     '<span class="pill pill-doc">Document</span>',
        "web":     '<span class="pill pill-web">Web</span>',
        "llm":     '<span class="pill pill-llm">LLM</span>',
        "rag+web": '<span class="pill pill-doc">Document</span>'
                   '<span class="pill pill-web">Web</span>',
    }
    return pills.get(source, "")


def get_chat_response(system_prompt: str, messages: list) -> str:
    """Call Groq and return the assistant reply."""
    try:
        model = get_chatgroq_model(mode=st.session_state.response_mode)
        formatted = [SystemMessage(content=system_prompt)]
        for m in messages:
            if m["role"] == "user":
                formatted.append(HumanMessage(content=m["content"]))
            else:
                formatted.append(AIMessage(content=m["content"]))
        return model.invoke(formatted).content

    except RuntimeError:
        raise
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise RuntimeError(f"LLM error: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(prompt: str) -> None:
    """
    Full pipeline: RAG retrieval + web search + LLM response.
    Must be called inside an active st.chat_message("assistant") block.
    """
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # 1. RAG
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

                # 4. LLM response
                response = get_chat_response(system_prompt, st.session_state.messages)

                # 5. Determine source
                if rag_context and web_context:
                    source = "rag+web"
                elif rag_context:
                    source = "rag"
                elif web_context:
                    source = "web"
                else:
                    source = "llm"

                # Render response
                st.markdown(response)
                st.markdown(source_pills(source), unsafe_allow_html=True)

                # Optional: show retrieved context
                if rag_context or web_context:
                    with st.expander("View sources", expanded=False):
                        if rag_context:
                            st.caption("From document")
                            st.code(rag_context[:1500], language=None)
                        if web_context:
                            st.caption("From web")
                            st.markdown(web_context[:1200])

                # Save to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "source": source}
                )

            except RuntimeError as e:
                st.error(f"Error: {e}")
                logger.error("Pipeline error: %s", e)
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                logger.exception("Unexpected error: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar component
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # Wordmark
        st.markdown("""
            <div style="padding: 1rem 0 1.2rem;">
                <div class="aria-wordmark">ARIA</div>
                <div style="font-size:0.7rem; color:#4b5563; font-family:Inter,sans-serif;
                            letter-spacing:0.06em; text-transform:uppercase; margin-top:3px;">
                    Research Assistant
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # PDF Upload
        st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

        if uploaded:
            if uploaded.name != st.session_state.uploaded_filename:
                with st.spinner("Indexing…"):
                    try:
                        vs = ingest_pdf(uploaded)
                        st.session_state.vector_store = vs
                        st.session_state.uploaded_filename = uploaded.name
                        logger.info("Indexed: %s", uploaded.name)
                    except RuntimeError as e:
                        st.error(f"Could not index PDF: {e}")
                        logger.error("Index failed: %s", e)

            if st.session_state.vector_store:
                name = st.session_state.uploaded_filename or ""
                short = name[:26] + "…" if len(name) > 26 else name
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#4ade80;margin-top:4px;">'
                    f'<span class="dot dot-on"></span>{short}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:0.78rem;color:#4b5563;margin-top:4px;">'
                '<span class="dot dot-off"></span>No document loaded</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Response mode
        st.markdown('<div class="section-label">Response mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "mode",
            options=["concise", "detailed"],
            index=0 if st.session_state.response_mode == "concise" else 1,
            format_func=str.capitalize,
            label_visibility="collapsed",
            horizontal=True,
        )
        st.session_state.response_mode = mode
        st.caption("Short answers." if mode == "concise" else "In-depth explanations.")

        st.divider()

        # Web search
        st.markdown('<div class="section-label">Web search</div>', unsafe_allow_html=True)
        web_on = st.toggle(
            "Live web search",
            value=st.session_state.use_web_search,
            disabled=not bool(TAVILY_API_KEY),
            label_visibility="collapsed",
        )
        st.session_state.use_web_search = web_on and bool(TAVILY_API_KEY)

        dot_cls = "dot-on" if st.session_state.use_web_search else "dot-off"
        label   = "On" if st.session_state.use_web_search else ("No API key" if not TAVILY_API_KEY else "Off")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#6b7280;margin-top:4px;">'
            f'<span class="dot {dot_cls}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Clear chat
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

        # API status
        st.divider()
        st.markdown('<div class="section-label">API status</div>', unsafe_allow_html=True)
        for label, key in [("Groq", GROQ_API_KEY), ("Tavily", TAVILY_API_KEY)]:
            dot = "dot-on" if key else "dot-off"
            state = "connected" if key else "missing key"
            st.markdown(
                f'<div style="font-size:0.75rem;color:#6b7280;margin-bottom:3px;">'
                f'<span class="dot {dot}"></span>{label} — {state}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Chat page component
# ─────────────────────────────────────────────────────────────────────────────
def render_empty_state() -> None:
    """Suggestion buttons shown on first load."""
    st.markdown(
        '<div style="color:#4b5563;font-size:0.88rem;margin-bottom:1.2rem;'
        'font-family:Inter,sans-serif;">Pick a question to get started, or type your own below.</div>',
        unsafe_allow_html=True,
    )
    suggestions = [
        ("Summarise this paper",     "Give me a summary of the key contributions and findings in this paper."),
        ("Explain the methodology",  "Explain the methodology used in this research paper in detail."),
        ("Latest LLM research",      "What are the latest trends in large language model research?"),
        ("RAG vs Fine-tuning",       "What are the differences between RAG and fine-tuning for LLMs?"),
    ]
    cols = st.columns(2)
    for i, (label, prompt) in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(label, key=f"s_{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()


def render_messages() -> None:
    """Render full chat history with source pills."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "source" in msg:
                st.markdown(source_pills(msg["source"]), unsafe_allow_html=True)


def chat_page() -> None:
    # Page header
    st.markdown("""
        <div style="margin-bottom:1.8rem;">
            <div class="aria-wordmark" style="font-size:2rem;">ARIA</div>
            <div style="font-size:0.85rem;color:#6b7280;margin-top:4px;font-family:Inter,sans-serif;">
                AI Research Intelligence Assistant
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Show empty state only when no messages and nothing pending
    if not st.session_state.messages and not st.session_state.pending_prompt:
        render_empty_state()

    # Render chat history
    render_messages()

    # Process suggestion button click from previous run
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        run_pipeline(prompt)

    # Chat input
    if prompt := st.chat_input("Ask ARIA…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        run_pipeline(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Instructions page component
# ─────────────────────────────────────────────────────────────────────────────
def instructions_page() -> None:
    st.markdown("""
        <div style="margin-bottom:1.8rem;">
            <div class="aria-wordmark" style="font-size:2rem;">Setup</div>
            <div style="font-size:0.85rem;color:#6b7280;margin-top:4px;font-family:Inter,sans-serif;">
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

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) — free |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) — free |

### Run locally

```bash
GROQ_API_KEY=gsk_... TAVILY_API_KEY=tvly_... streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect repo → set `app.py` as entry point
3. Add secrets under Settings → Secrets:

```toml
GROQ_API_KEY = "gsk_..."
TAVILY_API_KEY = "tvly-..."
```

### Project structure

```
project/
├── config/config.py        ← API keys & settings
├── models/llm.py           ← Groq Llama 3.3 70B
├── models/embeddings.py    ← HuggingFace MiniLM embeddings
├── utils/rag.py            ← PDF → chunks → FAISS → retrieve
├── utils/web_search.py     ← Tavily web search
├── utils/prompt_builder.py ← System prompt builder
├── app.py                  ← Streamlit UI
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
            st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
            page = st.radio("nav", options=["Chat", "Instructions"], label_visibility="collapsed")

        if page == "Chat":
            chat_page()
        else:
            instructions_page()

    except Exception as e:
        st.error(f"Application error: {e}")
        logger.exception("Fatal error: %s", e)


if __name__ == "__main__":
    main()
