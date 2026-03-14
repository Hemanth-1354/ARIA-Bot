# ARIA - AI Research Intelligence Assistant

ARIA is an intelligent, modular research assistant that lets you upload academic papers, ask deep questions about them, and get real-time web search results - all powered by a Groq LLM, a local FAISS vector store, and HuggingFace embeddings.

---

## Features

- **PDF Q&A** - Upload one or multiple research papers and ask questions using Retrieval-Augmented Generation (RAG)
- **Real-time Web Search** - Augment answers with live results via the Tavily API
- **Smart Follow-ups** - Auto-generates 3 follow-up questions after every response
- **Dual Response Modes** - Switch between Concise (quick answers) and Detailed (in-depth analysis) instantly
- **Multi-PDF Comparison** - Ingest multiple papers and search across all of them simultaneously
- **Fully Local Embeddings** - No external API needed for embeddings; runs on CPU

---

## System Architecture

```
User Query
    │
    ▼
Streamlit UI (app.py)
    │
    ├──► PDF Upload ──► PyPDFLoader ──► Chunking ──► FAISS Index
    │                                                      │
    │                                               Similarity Search
    │                                                      │
    ├──► Tavily Web Search ────────────────────────────────┤
    │                                                      │
    └──► Prompt Builder ◄──────────── RAG Context ─────────┘
              │
              ▼
        Groq LLM (Llama 3.3 70B)
              │
              ▼
       Response + Follow-up Questions
```

---

## Project Structure

```
project/
├── config/
│   └── config.py           ← API keys, model names & RAG settings
├── models/
│   ├── llm.py              ← Groq LLM initialisation (concise/detailed modes)
│   └── embeddings.py       ← HuggingFace embedding model loader
├── utils/
│   ├── rag.py              ← Full RAG pipeline (load, chunk, index, retrieve)
│   ├── web_search.py       ← Tavily real-time web search
│   ├── prompt_builder.py   ← Dynamic system prompt construction
│   └── followup.py         ← Auto follow-up question generator
├── app.py                  ← Streamlit UI entry point
├── requirements.txt        ← Python dependencies
└── README.md
```

---


## Configuration

All settings are managed in `config/config.py` via environment variables:

| Variable          | Default                   | Description                              |
| ----------------- | ------------------------- | ---------------------------------------- |
| `GROQ_API_KEY`    | —                         | Groq API key (required)                  |
| `GROQ_MODEL`      | `llama-3.3-70b-versatile` | Groq model name                          |
| `TAVILY_API_KEY`  | —                         | Tavily API key (required for web search) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`        | HuggingFace embedding model              |
| `CHUNK_SIZE`      | `800`                     | PDF chunk size (characters)              |
| `CHUNK_OVERLAP`   | `100`                     | Overlap between chunks                   |
| `TOP_K_RESULTS`   | `4`                       | Number of chunks retrieved per query     |

---

## Module Overview

### `models/embeddings.py`

Loads the `all-MiniLM-L6-v2` HuggingFace model locally on CPU. No API key required. Produces normalized embeddings for semantic similarity search.

### `models/llm.py`

Initialises the Groq `ChatGroq` model with two modes:

- **Concise** — max 300 tokens, fast replies
- **Detailed** — max 1500 tokens, comprehensive analysis

### `utils/rag.py`

Full RAG pipeline:

1. Load PDF with `PyPDFLoader`
2. Chunk with `RecursiveCharacterTextSplitter` (size=800, overlap=100)
3. Build FAISS index with HuggingFace embeddings
4. Retrieve top-k relevant chunks for any query
5. Supports multi-PDF merge and per-document retrieval for comparison mode

### `utils/web_search.py`

Calls the Tavily API to fetch real-time web results. Returns a synthesised answer plus top sources with snippets and URLs.

### `utils/prompt_builder.py`

Dynamically assembles the system prompt with:

- ARIA persona
- Response mode instruction (concise/detailed)
- RAG document context (if a PDF is uploaded)
- Web search results (if enabled)

### `utils/followup.py`

After every assistant reply, generates 3 short follow-up questions using the same Groq model in concise mode. Fails silently - never disrupts the main Q&A pipeline.

---

## Tech Stack

| Component    | Technology                                                             |
| ------------ | ---------------------------------------------------------------------- |
| LLM          | [Groq](https://groq.com) · Llama 3.3 70B Versatile                     |
| Embeddings   | [HuggingFace](https://huggingface.co) · all-MiniLM-L6-v2               |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) (local, CPU)        |
| Web Search   | [Tavily API](https://tavily.com)                                       |
| Framework    | [LangChain](https://langchain.com) + [Streamlit](https://streamlit.io) |
| PDF Loader   | LangChain `PyPDFLoader`                                                |

---

## Known Limitations & Assumptions

- Embeddings run on **CPU only** - sufficient for MVP but slower on large PDFs
- PDFs must be **text-based** (not scanned images); OCR is not supported
- API keys must be set in the environment before running the app
- Follow-up generation is **non-critical** and will fail silently without affecting core Q&A

---

## Future Improvements

- [ ] GPU acceleration for faster embedding on large document sets
- [ ] OCR support for scanned PDFs
- [ ] Automatic citation extraction and referencing
- [ ] Multi-language document support
- [ ] Persistent vector store (save and reload FAISS indexes)
- [ ] User authentication and session management

---

