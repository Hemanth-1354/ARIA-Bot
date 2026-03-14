"""
utils/rag.py
RAG pipeline — supports multiple PDFs and paper comparison mode.

Public API:
  ingest_pdf(file)                → FAISS index for one PDF
  merge_indexes(indexes)          → merged FAISS index across all docs
  retrieve_relevant_chunks(vs, q) → top-k chunks as string
  retrieve_per_doc(docs_dict, q)  → {name: chunks} dict for comparison
"""

import os
import sys
import logging
import tempfile
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from models.embeddings import get_embedding_model
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS

logger = logging.getLogger(__name__)


def load_pdf_documents(uploaded_file) -> List[Document]:
    """
    Load a Streamlit UploadedFile PDF into LangChain Documents.
    Tags each document chunk with the original filename as metadata.

    Raises:
        RuntimeError: on any load failure.
    """
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)

        if not documents:
            raise ValueError("No content extracted from PDF.")

        # Tag every chunk with the original filename for multi-doc attribution
        for doc in documents:
            doc.metadata["filename"] = uploaded_file.name

        logger.info("Loaded %d pages from '%s'.", len(documents), uploaded_file.name)
        return documents

    except ValueError as ve:
        logger.warning("PDF load warning: %s", ve)
        raise RuntimeError(str(ve)) from ve
    except Exception as e:
        logger.error("Failed to load PDF '%s': %s", getattr(uploaded_file, "name", "?"), e)
        raise RuntimeError(f"Failed to load PDF: {e}") from e


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks.

    Raises:
        RuntimeError: if chunking fails or produces no chunks.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No chunks created from documents.")

        logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))
        return chunks

    except ValueError as ve:
        logger.warning("Chunking warning: %s", ve)
        raise RuntimeError(str(ve)) from ve
    except Exception as e:
        logger.error("Failed to chunk documents: %s", e)
        raise RuntimeError(f"Failed to chunk documents: {e}") from e


def build_faiss_index(chunks: List[Document]) -> FAISS:
    """
    Build a FAISS vector store from document chunks.

    Raises:
        RuntimeError: if index creation fails.
    """
    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS index built with %d vectors.", len(chunks))
        return vector_store
    except Exception as e:
        logger.error("Failed to build FAISS index: %s", e)
        raise RuntimeError(f"Failed to build FAISS index: {e}") from e


def ingest_pdf(uploaded_file) -> FAISS:
    """
    Full single-PDF pipeline: load → chunk → FAISS index.

    Returns:
        FAISS vector store.

    Raises:
        RuntimeError: on any pipeline failure.
    """
    try:
        documents = load_pdf_documents(uploaded_file)
        chunks = chunk_documents(documents)
        return build_faiss_index(chunks)
    except RuntimeError:
        raise
    except Exception as e:
        logger.error("RAG ingestion failed: %s", e)
        raise RuntimeError(f"Ingestion failed: {e}") from e


def merge_indexes(indexes: List[FAISS]) -> Optional[FAISS]:
    """
    Merge multiple FAISS indexes into one for cross-document search.

    Args:
        indexes: List of FAISS vector stores (one per uploaded PDF).

    Returns:
        A single merged FAISS index, or None if list is empty.

    Raises:
        RuntimeError: if merge fails.
    """
    try:
        if not indexes:
            return None
        if len(indexes) == 1:
            return indexes[0]

        base = indexes[0]
        for idx in indexes[1:]:
            base.merge_from(idx)

        logger.info("Merged %d FAISS indexes into one.", len(indexes))
        return base

    except Exception as e:
        logger.error("Failed to merge FAISS indexes: %s", e)
        raise RuntimeError(f"Failed to merge indexes: {e}") from e


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    k: int = TOP_K_RESULTS,
) -> str:
    """
    Retrieve top-k relevant chunks from a FAISS index.

    Returns:
        Formatted string with chunk text and source metadata.

    Raises:
        RuntimeError: if retrieval fails.
    """
    try:
        docs = vector_store.similarity_search(query, k=k)
        if not docs:
            return ""

        parts = []
        for i, doc in enumerate(docs, 1):
            page     = doc.metadata.get("page", "?")
            filename = doc.metadata.get("filename", "document")
            parts.append(f"[Chunk {i} | {filename} | Page {page}]\n{doc.page_content.strip()}")

        context = "\n\n---\n\n".join(parts)
        logger.info("Retrieved %d chunks for query.", len(docs))
        return context

    except Exception as e:
        logger.error("Failed to retrieve chunks: %s", e)
        raise RuntimeError(f"Failed to retrieve relevant chunks: {e}") from e


def retrieve_per_doc(
    doc_indexes: Dict[str, FAISS],
    query: str,
    k: int = TOP_K_RESULTS,
) -> Dict[str, str]:
    """
    Retrieve top-k chunks separately from each named FAISS index.
    Used for paper comparison mode — keeps each paper's context separate.

    Args:
        doc_indexes: dict mapping filename → FAISS index.
        query: user's question.
        k: chunks per document.

    Returns:
        dict mapping filename → retrieved context string.

    Raises:
        RuntimeError: if retrieval fails for any document.
    """
    try:
        results: Dict[str, str] = {}
        for name, vs in doc_indexes.items():
            try:
                results[name] = retrieve_relevant_chunks(vs, query, k=k)
            except RuntimeError as e:
                logger.warning("Retrieval failed for '%s': %s", name, e)
                results[name] = ""

        logger.info("Per-doc retrieval done for %d documents.", len(doc_indexes))
        return results

    except Exception as e:
        logger.error("Per-doc retrieval failed: %s", e)
        raise RuntimeError(f"Per-doc retrieval failed: {e}") from e
