import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialise and return a HuggingFace sentence-transformer embedding model.

    The model runs entirely locally — no external API key required.
    Default: 'all-MiniLM-L6-v2'  (fast, ~80 MB, high quality for semantic search)

    Returns:
        HuggingFaceEmbeddings instance ready for use with FAISS.

    Raises:
        RuntimeError: If the embedding model fails to load.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model '%s' loaded successfully.", EMBEDDING_MODEL)
        return embeddings

    except Exception as e:
        logger.error("Failed to load embedding model '%s': %s", EMBEDDING_MODEL, e)
        raise RuntimeError(f"Failed to load embedding model: {e}") from e
