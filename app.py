import os
import streamlit as st

from Retriever import (
    extract_text_from_pdf,
    create_chunks,
    create_vector_store,
    retrieve_documents,
)
from Generator import generate_answer


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload a PDF and ask questions about it using "
    "FAISS retrieval and an open-weight LLM through Groq."
)

# Support both Streamlit Cloud secrets and local environment variables.
try:
    secret_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    secret_key = None

if secret_key:
    os.environ["GROQ_API_KEY"] = secret_key

if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY is not configured. Add it to "
        ".streamlit/secrets.toml locally or Streamlit Cloud Secrets."
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks sent to the generator.",
    )
    st.markdown("---")
    st.markdown(
        """
**Pipeline**

PDF → PyMuPDF → Chunks → MiniLM Embeddings → FAISS → Groq LLM
"""
    )

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Text-based PDFs are supported. Scanned/image-only PDFs need OCR.",
)

if uploaded_file is not None:
    if st.session_state.document_name != uploaded_file.name:
        with st.spinner("Extracting, chunking, embedding, and indexing PDF..."):
            try:
                pages = extract_text_from_pdf(uploaded_file)

                if not pages:
                    st.error(
                        "No extractable text was found. "
                        "This may be a scanned/image-only PDF."
                    )
                    st.stop()

                chunks = create_chunks(
                    pages,
                    chunk_size=500,
                    overlap=100,
                )

                if not chunks:
                    st.error("No text chunks could be created.")
                    st.stop()

                index, chunks = create_vector_store(chunks)

                st.session_state.vector_index = index
                st.session_state.chunks = chunks
                st.session_state.document_name = uploaded_file.name

            except Exception as exc:
                st.error(f"Error processing PDF: {exc}")
                st.stop()

    st.success(
        f"Loaded **{st.session_state.document_name}** — "
        f"{len(st.session_state.chunks)} chunks indexed."
    )

if st.session_state.vector_index is not None:
    st.divider()
    st.subheader("💬 Ask a question")

    question = st.text_input(
        "Question",
        placeholder="What is the main objective of this document?",
    )

    ask = st.button("Ask", type="primary", use_container_width=False)

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Retrieving relevant passages..."):
            retrieved_documents = retrieve_documents(
                question,
                st.session_state.vector_index,
                st.session_state.chunks,
                top_k=top_k,
            )

        if not retrieved_documents:
            st.warning("No relevant passages were retrieved.")
            st.stop()

        with st.spinner("Generating answer with Groq..."):
            try:
                answer = generate_answer(
                    question,
                    retrieved_documents,
                )
            except Exception as exc:
                st.error(f"Error generating answer: {exc}")
                st.stop()

        st.subheader("🤖 Answer")
        st.write(answer)

        st.divider()
        st.subheader("📖 Retrieved Sources")

        for number, document in enumerate(retrieved_documents, start=1):
            page = document["page"]
            score = document["score"]

            with st.expander(
                f"Source {number} — Page {page} — Similarity {score:.3f}"
            ):
                st.write(document["text"])
