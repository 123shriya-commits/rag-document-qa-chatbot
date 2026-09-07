# RAG Document Q&A Chatbot

A local, fully open-source Retrieval-Augmented Generation (RAG) chatbot that answers questions
about any PDF or TXT document you upload — no API key required.

## How it works

1. **Upload** a PDF or TXT document through the Streamlit UI
2. **Chunking** — the document is split into overlapping ~500-character chunks
3. **Embedding** — each chunk is converted into a vector using `sentence-transformers` (all-MiniLM-L6-v2)
4. **Indexing** — vectors are stored in a FAISS index for fast similarity search
5. **Retrieval** — when you ask a question, it's embedded and matched against the index to find the most relevant chunks
6. **Generation** — the retrieved chunks + your question are passed to `google/flan-t5-base`, which generates an answer grounded in the actual document content

## Tech stack

| Component        | Tool                          |
|-------------------|-------------------------------|
| Embeddings        | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search      | FAISS (IndexFlatL2)           |
| Text generation    | google/flan-t5-base           |
| UI                 | Streamlit                     |
| PDF parsing        | pypdf                         |

## Setup

1. Clone this repo / copy this folder to your machine
2. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. On first run, the embedding and generation models will download automatically from Hugging Face
   (~250 MB total) — this only happens once and requires an internet connection.

## Usage

1. Upload a PDF or TXT file in the sidebar (or try the included `sample_docs/sample_ai_notes.txt`)
2. Click **Process document**
3. Ask questions in the chat box — answers are generated from the document content only
4. Expand **View retrieved source chunks** under any answer to see exactly which parts of the
   document were used to generate it (this is what makes RAG explainable, unlike a plain chatbot)

## Project structure

```
rag_chatbot/
├── app.py              # Streamlit UI
├── rag_engine.py        # Core RAG logic (chunking, embedding, retrieval, generation)
├── requirements.txt
├── sample_docs/
│   └── sample_ai_notes.txt
└── README.md
```

## Possible improvements (good talking points in interviews)

- Swap `flan-t5-base` for a larger instruction-tuned model (e.g. Mistral-7B-Instruct) for higher-quality answers
- Add a re-ranking step after retrieval to improve relevance before generation
- Support multi-document upload with source-document attribution
- Add conversation memory so follow-up questions understand prior context
- Deploy to Streamlit Community Cloud or HuggingFace Spaces for a public live demo
- Add evaluation metrics (e.g. answer relevance scoring, retrieval precision@k) against a small labeled Q&A set

## Notes for resume / portfolio use

Once you've tested this on a real document, fill in your own numbers before adding it to your resume:
- Number of chunks indexed for a typical document
- Retrieval latency (time to return top-k chunks)
- End-to-end response time (retrieval + generation)
- Any accuracy/relevance testing you do manually against a sample Q&A set

Never state metrics you haven't actually measured — be ready to explain any number on your resume.
