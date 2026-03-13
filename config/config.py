import os

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Web Search ────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── RAG ───────────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS: int = int(os.environ.get("TOP_K_RESULTS", "4"))

# ── Response Modes ────────────────────────────────────────────────────────────
CONCISE_MAX_TOKENS: int = 300
DETAILED_MAX_TOKENS: int = 1500
