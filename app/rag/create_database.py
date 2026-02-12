"""
create_database.py — Ingest career-guide documents into a ChromaDB vector store.

Usage (standalone)::

    python -m app.rag.create_database          # from project root
    python app/rag/create_database.py           # also works

The script:
1. Loads all Markdown files from ``app/rag/data/career_guides/``.
2. Splits them into overlapping chunks using a token-aware splitter.
3. Generates OpenAI embeddings (``text-embedding-3-small``).
4. Upserts the chunks into a persistent ChromaDB collection.
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

# Resolve paths relative to this file so it works from any working directory
_THIS_DIR = Path(__file__).resolve().parent
GUIDES_DIR = _THIS_DIR / "data" / "career_guides"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", str(_THIS_DIR.parent.parent / "chroma")))

COLLECTION_NAME = "career_guides"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800        # tokens — keeps each chunk focused
CHUNK_OVERLAP = 150     # tokens — preserves context across boundaries


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

def build_vectorstore(
    guides_dir: Path = GUIDES_DIR,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Load career guides, chunk, embed, and persist to ChromaDB.

    Returns the ``Chroma`` vectorstore instance so callers can query
    immediately if they want.
    """
    guides_dir = Path(guides_dir)
    chroma_dir = Path(chroma_dir)

    if not guides_dir.exists() or not any(guides_dir.glob("*.md")):
        raise FileNotFoundError(
            f"No .md files found in {guides_dir}. "
            "Add career guide documents before building the database."
        )

    logger.info("Loading documents from %s …", guides_dir)
    loader = DirectoryLoader(
        str(guides_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    logger.info("Loaded %d document(s)", len(documents))

    # ---- chunk -------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))

    # ---- embed + persist ---------------------------------------------
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Wipe the old collection so we get a clean rebuild
    if chroma_dir.exists():
        # Chroma.from_documents will overwrite if same collection name
        logger.info("Persisting to %s (collection: %s)", chroma_dir, collection_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(chroma_dir),
    )

    logger.info(
        "ChromaDB collection '%s' created with %d vectors at %s",
        collection_name,
        len(chunks),
        chroma_dir,
    )
    return vectorstore


# ------------------------------------------------------------------ #
#  CLI entry-point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    try:
        vs = build_vectorstore()
        # Quick sanity check
        results = vs.similarity_search("How do I negotiate a higher salary?", k=2)
        print(f"\n✅  Vector store built successfully ({vs._collection.count()} vectors)")
        print("Sample query — 'How do I negotiate a higher salary?':")
        for i, doc in enumerate(results, 1):
            src = Path(doc.metadata.get("source", "?")).name
            print(f"  [{i}] ({src}) {doc.page_content[:120]}…")
    except Exception as exc:
        print(f"\n❌  Failed: {exc}", file=sys.stderr)
        sys.exit(1)
