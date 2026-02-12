"""
query.py — Similarity-search interface over the career-guides ChromaDB collection.

Typical usage::

    from app.rag.query import query_knowledge_base

    results = query_knowledge_base("How do I negotiate salary?", k=4)
    for r in results:
        print(r["content"][:120], r["source"], r["score"])

If the ChromaDB collection has not been built yet, the function returns an
empty list and logs a warning — it never raises on a missing collection.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Configuration — mirrors create_database.py
# ------------------------------------------------------------------ #

_THIS_DIR = Path(__file__).resolve().parent
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", str(_THIS_DIR.parent.parent / "chroma")))
COLLECTION_NAME = "career_guides"
EMBEDDING_MODEL = "text-embedding-3-small"

# Module-level cache so the vectorstore is loaded once per process
_vectorstore: Optional[Chroma] = None


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

def _get_vectorstore() -> Optional[Chroma]:
    """Lazily load (and cache) the persisted ChromaDB collection.

    Returns ``None`` if the persisted directory does not exist yet.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not CHROMA_DIR.exists():
        logger.warning(
            "ChromaDB directory %s does not exist. "
            "Run `python -m app.rag.create_database` first.",
            CHROMA_DIR,
        )
        return None

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        logger.info(
            "Loaded ChromaDB collection '%s' from %s",
            COLLECTION_NAME,
            CHROMA_DIR,
        )
        return _vectorstore
    except Exception as exc:
        logger.error("Failed to load ChromaDB collection: %s", exc)
        return None


def reload_vectorstore() -> Optional[Chroma]:
    """Force-reload the vector store (useful after re-ingesting docs)."""
    global _vectorstore
    _vectorstore = None
    return _get_vectorstore()


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

def query_knowledge_base(
    query: str,
    k: int = 4,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run a similarity search and return the top-*k* relevant chunks.

    Parameters
    ----------
    query:
        The natural-language question.
    k:
        Maximum number of chunks to return.
    score_threshold:
        Minimum relevance score (0-1, cosine distance).  Chunks below this
        threshold are discarded.  Default ``0.0`` means return everything.

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``content``  – the chunk text
        - ``source``   – originating file path
        - ``score``    – relevance score (higher = more similar)
        - ``metadata`` – full LangChain metadata dict
    """
    store = _get_vectorstore()
    if store is None:
        logger.warning("Knowledge base not available — returning empty results.")
        return []

    try:
        # similarity_search_with_relevance_scores returns (Document, score) pairs
        raw_results = store.similarity_search_with_relevance_scores(query, k=k)
    except Exception as exc:
        logger.error("Similarity search failed: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    for doc, score in raw_results:
        if score < score_threshold:
            continue
        source_path = doc.metadata.get("source", "unknown")
        # Keep only the filename for cleaner attribution
        source_name = Path(source_path).stem.replace("_", " ").title()
        results.append(
            {
                "content": doc.page_content,
                "source": source_name,
                "score": round(float(score), 4),
                "metadata": doc.metadata,
            }
        )

    logger.info(
        "Knowledge base query returned %d result(s) for: %.80s",
        len(results),
        query,
    )
    return results


def get_formatted_context(query: str, k: int = 4) -> str:
    """Return a pre-formatted context string suitable for LLM injection.

    This is a convenience wrapper around :func:`query_knowledge_base` that
    concatenates the retrieved chunks into a single text block with source
    attribution, ready to drop into a prompt.
    """
    results = query_knowledge_base(query, k=k)
    if not results:
        return ""

    sections: List[str] = []
    for i, r in enumerate(results, 1):
        sections.append(
            f"[Source {i}: {r['source']}]\n{r['content']}"
        )
    return "\n\n---\n\n".join(sections)
