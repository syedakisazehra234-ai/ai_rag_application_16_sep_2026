import os
import hashlib
import tempfile
from typing import List, Dict

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide",
)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


# -----------------------------
# PDF + RAG functions
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF uploaded through Streamlit."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text}")

    return "\n\n".join(pages)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Create overlapping word-based chunks."""
    words = normalize_text(text).split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_faiss_index(chunks: List[str], model):
    """Embed chunks and store them in a FAISS cosine-similarity index."""
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve(query: str, index, chunks: List[str], model, top_k: int = TOP_K):
    """Retrieve the most semantically similar chunks."""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append(
                {
                    "chunk": chunks[idx],
                    "score": float(score),
                    "index": int(idx),
                }
            )

    return results


def answer_with_groq(question: str, retrieved_chunks: List[Dict], client):
    """Generate an answer using only the retrieved PDF context."""
    context = "\n\n---\n\n".join(
        f"Context {i + 1}:\n{item['chunk']}"
        for i, item in enumerate(retrieved_chunks)
    )

    system_prompt = """You are a helpful document question-answering assistant.

Answer the user's question using the supplied document context.
Rules:
1. Use the document context as the primary source of truth.
2. Do not invent facts that are not supported by the context.
3. If the answer is not present in the context, clearly say that the
   information was not found in the uploaded document.
4. Keep the answer clear and reasonably concise.
5. When useful, mention the page number shown in the retrieved context.
"""

    user_prompt = f"""Document context:
{context}

Question:
{question}
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


def get_file_id(uploaded_file) -> str:
    """Create a stable ID so the same uploaded PDF isn't processed repeatedly."""
    file_bytes = uploaded_file.getvalue()
    return hashlib.sha256(file_bytes).hexdigest()


def process_pdf(uploaded_file, embedding_model):
    text = extract_pdf_text(uploaded_file)

    if not text.strip():
        raise ValueError(
            "No extractable text was found. This app currently supports "
            "text-based PDFs, not scanned/image-only PDFs."
        )

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("The PDF did not produce any usable text chunks.")

    index, _ = build_faiss_index(chunks, embedding_model)
    return text, chunks, index


# -----------------------------
# UI
# -----------------------------
st.title("📚 PDF RAG Chat")
st.caption(
    "Upload a PDF → extract text → chunk → embed with an open-source "
    "embedding model → search with FAISS → answer with Groq."
)

# API key: Streamlit Cloud uses st.secrets; local development can use env var.
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

if not groq_api_key:
    st.warning(
        "GROQ_API_KEY is not configured. Add it to Streamlit Secrets before "
        "asking questions."
    )

with st.sidebar:
    st.header("⚙️ RAG Settings")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Embedding:** `{EMBEDDING_MODEL}`")
    st.write(f"**Chunk size:** `{CHUNK_SIZE}` words")
    st.write(f"**Chunk overlap:** `{CHUNK_OVERLAP}` words")
    st.write(f"**Top-K:** `{TOP_K}`")

    if st.button("🗑️ Clear current document", use_container_width=True):
        for key in [
            "file_id",
            "pdf_name",
            "pdf_text",
            "chunks",
            "faiss_index",
            "messages",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a text-based PDF. Scanned/image-only PDFs require OCR.",
)

if uploaded_file is not None:
    file_id = get_file_id(uploaded_file)

    # Process only when a new/different PDF is uploaded.
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Processing PDF and building FAISS index..."):
            try:
                pdf_text, chunks, index = process_pdf(
                    uploaded_file,
                    load_embedding_model(),
                )

                st.session_state.file_id = file_id
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_text = pdf_text
                st.session_state.chunks = chunks
                st.session_state.faiss_index = index
                st.session_state.messages = []

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")
                st.stop()

    st.success(
        f"Loaded **{st.session_state.pdf_name}** — "
        f"{len(st.session_state.chunks)} chunks indexed in FAISS."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Extracted characters", f"{len(st.session_state.pdf_text):,}")
    with col2:
        st.metric("Indexed chunks", f"{len(st.session_state.chunks):,}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        if not groq_api_key:
            answer = "Please configure `GROQ_API_KEY` in Streamlit Secrets first."
            with st.chat_message("assistant"):
                st.error(answer)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching the document and generating an answer..."):
                    try:
                        embedding_model = load_embedding_model()

                        retrieved = retrieve(
                            question,
                            st.session_state.faiss_index,
                            st.session_state.chunks,
                            embedding_model,
                            TOP_K,
                        )

                        client = get_groq_client(groq_api_key)
                        answer = answer_with_groq(
                            question,
                            retrieved,
                            client,
                        )

                        st.markdown(answer)

                        with st.expander("🔎 Retrieved context"):
                            for i, item in enumerate(retrieved, start=1):
                                st.markdown(
                                    f"**Result {i} — similarity: "
                                    f"{item['score']:.3f}**"
                                )
                                st.write(item["chunk"])

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")

else:
    st.info("Upload a PDF above to start chatting with your document.")

    st.markdown(
        """
### How this RAG pipeline works

1. **PDF extraction** — `pypdf` extracts text from each page.
2. **Chunking** — the text is split into overlapping chunks.
3. **Embeddings** — `all-MiniLM-L6-v2` creates open-source vector embeddings.
4. **Vector database** — FAISS stores the embeddings locally in memory.
5. **Retrieval** — the user's question is embedded and the closest chunks are found.
6. **Generation** — Groq runs `openai/gpt-oss-120b` using the retrieved context.
7. **Answer** — the model answers from the uploaded document context.
"""
    )
