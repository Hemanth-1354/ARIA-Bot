"""
utils/rag.py
RAG pipeline: PDF ingestion → chunking → FAISS vector store → retrieval.
"""

import sys
import os
import logging
import tempfile
from typing import List, Optional

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
    Load a Streamlit UploadedFile PDF into LangChain Document objects.

    Args:
        uploaded_file: Streamlit UploadedFile object (PDF).

    Returns:
        List of LangChain Document objects with page content and metadata.

    Raises:
        RuntimeError: If the PDF cannot be loaded or parsed.
    """
    try:
        # Write uploaded bytes to a temp file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Clean up temp file
        os.unlink(tmp_path)

        if not documents:
            raise ValueError("No content could be extracted from the PDF.")

        logger.info(
            "Loaded %d pages from '%s'.", len(documents), uploaded_file.name
        )
        return documents

    except ValueError as ve:
        logger.warning("PDF load warning: %s", ve)
        raise RuntimeError(str(ve)) from ve

    except Exception as e:
        logger.error("Failed to load PDF '%s': %s", getattr(uploaded_file, "name", "unknown"), e)
        raise RuntimeError(f"Failed to load PDF: {e}") from e


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping text chunks for embedding.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of smaller Document chunks.

    Raises:
        RuntimeError: If chunking fails.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No chunks were created from the documents.")

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

    Args:
        chunks: List of text chunk Documents.

    Returns:
        FAISS vector store ready for similarity search.

    Raises:
        RuntimeError: If index creation fails.
    """
    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS index built with %d vectors.", len(chunks))
        return vector_store

    except Exception as e:
        logger.error("Failed to build FAISS index: %s", e)
        raise RuntimeError(f"Failed to build FAISS index: {e}") from e


def retrieve_relevant_chunks(
    vector_store: FAISS, query: str, k: int = TOP_K_RESULTS
) -> str:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Args:
        vector_store: FAISS vector store.
        query: User's question string.
        k: Number of chunks to retrieve.

    Returns:
        Concatenated string of relevant chunks with source labels.

    Raises:
        RuntimeError: If retrieval fails.
    """
    try:
        docs = vector_store.similarity_search(query, k=k)

        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "document")
            context_parts.append(
                f"[Chunk {i} | Page {page}]\n{doc.page_content.strip()}"
            )

        context = "\n\n---\n\n".join(context_parts)
        logger.info("Retrieved %d chunks for query.", len(docs))
        return context

    except Exception as e:
        logger.error("Failed to retrieve chunks: %s", e)
        raise RuntimeError(f"Failed to retrieve relevant chunks: {e}") from e


def ingest_pdf(uploaded_file) -> Optional[FAISS]:
    """
    Full pipeline: load PDF → chunk → build FAISS index.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        FAISS vector store, or None if any step fails.
    """
    try:
        documents = load_pdf_documents(uploaded_file)
        chunks = chunk_documents(documents)
        vector_store = build_faiss_index(chunks)
        return vector_store

    except RuntimeError as re:
        logger.error("RAG ingestion pipeline failed: %s", re)
        raise
