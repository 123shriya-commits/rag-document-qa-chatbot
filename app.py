"""
app.py
------
Streamlit front-end for the RAG Document Q&A Chatbot.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from rag_engine import RAGPipeline, load_document_text

st.set_page_config(page_title="RAG Document Q&A Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 RAG Document Q&A Chatbot")
st.caption("Upload a PDF or TXT file, then ask questions about its content — powered by a local, "
           "open-source retrieval-augmented generation pipeline (no API key required).")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

# ---------------------------------------------------------------------------
# Sidebar: document upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None and st.button("Process document"):
        with st.spinner("Reading and indexing document... (first run also downloads models, may take a minute)"):
            # Save upload to a temp file so PdfReader/loader can access a real path
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                text = load_document_text(tmp_path)
                st.session_state.pipeline.build_index(text)
                st.session_state.document_loaded = True
                st.session_state.chat_history = []
                st.success(f"Indexed {len(st.session_state.pipeline.chunks)} chunks from '{uploaded_file.name}'")
            except Exception as e:
                st.error(f"Failed to process document: {e}")
            finally:
                os.remove(tmp_path)

    st.divider()
    st.caption(
        "**Tech stack:** sentence-transformers (embeddings) · FAISS (vector search) "
        "· flan-t5-base (generation) · Streamlit (UI)"
    )

# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------
if not st.session_state.document_loaded:
    st.info("👈 Upload a document and click **Process document** to get started.")
else:
    question = st.chat_input("Ask a question about the document...")

    if question:
        with st.spinner("Thinking..."):
            try:
                answer, sources = st.session_state.pipeline.generate_answer(question)
                st.session_state.chat_history.append((question, answer, sources))
            except Exception as e:
                st.error(f"Error generating answer: {e}")

    # Render chat history, most recent last
    for q, a, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            with st.expander("View retrieved source chunks"):
                for i, s in enumerate(sources, start=1):
                    st.markdown(f"**Chunk {i}:** {s}")
