import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loaded once when the module is imported.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def extract_text_from_pdf(pdf_file):
    """Extract text page-by-page from an uploaded PDF."""
    pdf_bytes = pdf_file.getvalue()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )
    finally:
        document.close()

    return pages


def create_chunks(pages, chunk_size=500, overlap=100):
    """
    Create word-based overlapping chunks while preserving page metadata.

    Example:
        chunk 1: words 1-500
        chunk 2: words 401-900
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                    }
                )

            if end >= len(words):
                break

    return chunks


def create_vector_store(chunks):
    """Embed chunks and create a normalized FAISS cosine-similarity index."""
    if not chunks:
        raise ValueError("Cannot create a vector store from zero chunks.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, chunks


def retrieve_documents(query, index, chunks, top_k=5):
    """Retrieve the top-k most semantically similar chunks."""
    if index is None or not chunks:
        return []

    top_k = max(1, min(top_k, len(chunks)))

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, position in zip(scores[0], indices[0]):
        if position == -1:
            continue

        results.append(
            {
                "text": chunks[position]["text"],
                "page": chunks[position]["page"],
                "score": float(score),
            }
        )

    return results
