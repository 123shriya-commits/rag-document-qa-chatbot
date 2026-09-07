"""
rag_engine.py
--------------
Core Retrieval-Augmented Generation (RAG) logic.

Pipeline:
1. Load a document (PDF or TXT)
2. Split it into overlapping chunks
3. Embed each chunk using a sentence-transformer model
4. Store embeddings in a FAISS vector index
5. On a user query: embed the query, retrieve the top-k most similar chunks,
   and feed them + the question into a local text-generation model
"""

import os
import re
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"      # small, fast, good quality
GENERATION_MODEL_NAME = "google/flan-t5-base"  # free, no API key, CPU-friendly
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
TOP_K = 3               # number of chunks retrieved per query


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_document_text(file_path: str) -> str:
    """Extract raw text from a PDF or TXT file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Collapse excessive whitespace/newlines."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks so context isn't lost at boundaries."""
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# RAG Pipeline class
# ---------------------------------------------------------------------------
class RAGPipeline:
    """
    Wraps the embedding model, FAISS index, and generation model into
    a single reusable object. Instantiate once per document.
    """

    def __init__(self):
        self.embedder = None
        self.generator = None
        self.index = None
        self.chunks: List[str] = []

    # -- lazy loading so Streamlit doesn't reload models on every rerun --
    def _load_models(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if self.generator is None:
            self.generator = pipeline(
                "text2text-generation",
                model=GENERATION_MODEL_NAME,
            )

    def build_index(self, text: str):
        """Chunk the document, embed each chunk, and build a FAISS index."""
        self._load_models()

        self.chunks = chunk_text(text)
        if not self.chunks:
            raise ValueError("No text could be extracted from the document.")

        embeddings = self.embedder.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Return the top_k most relevant chunks for a query, with distances."""
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        query_embedding = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        return results

    def generate_answer(self, query: str, top_k: int = TOP_K) -> Tuple[str, List[str]]:
        """
        Full RAG step: retrieve relevant chunks, build a prompt, and
        generate an answer grounded in those chunks.
        Returns (answer, list_of_source_chunks_used).
        """
        retrieved = self.retrieve(query, top_k=top_k)
        context_chunks = [chunk for chunk, _ in retrieved]
        context = "\n\n".join(context_chunks)

        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        result = self.generator(prompt, max_length=256, do_sample=False)
        answer = result[0]["generated_text"].strip()

        return answer, context_chunks
