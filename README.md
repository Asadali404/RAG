# 📚 PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its contents.

The application extracts text from the PDF, creates overlapping chunks, converts the chunks into semantic embeddings, stores the embeddings in a FAISS vector index, retrieves the most relevant passages for a question, and sends those passages to an open-weight language model through Groq.

## 🚀 Architecture

```text
User
 │
 ▼
Streamlit
 │
 ├── Upload PDF
 │
 ▼
PyMuPDF
 │
 ▼
Text Extraction
 │
 ▼
Overlapping Chunks
 │
 ▼
Sentence Transformers
(all-MiniLM-L6-v2)
 │
 ▼
Embeddings
 │
 ▼
FAISS
 │
 ▼
Semantic Retrieval
 │
 ▼
Top-K Relevant Chunks
 │
 ▼
Groq API
 │
 ▼
Llama 3.3 70B
 │
 ▼
Grounded Answer
 │
 ▼
Source Pages
```

## ✨ Features

- Upload a PDF directly from the Streamlit UI
- Extract PDF text with PyMuPDF
- Create overlapping text chunks
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Store vectors in FAISS
- Perform cosine-similarity retrieval
- Generate answers using an open-weight LLM through Groq
- Display retrieved source passages and PDF page numbers
- Adjustable Top-K retrieval
- Deployable to Streamlit Community Cloud

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| PDF extraction | PyMuPDF |
| Chunking | Python |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | FAISS |
| LLM inference | Groq |
| Generator model | Llama 3.3 70B |
| Deployment | Streamlit Community Cloud |
| Source code | GitHub |

## 📁 Project Structure

```text
pdf-rag-assistant/
│
├── app.py
├── Retriever.py
├── Generator.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml       # local only; never commit
```

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-assistant.git
cd pdf-rag-assistant
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key

Create this file:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Never commit this file to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## 🔐 Streamlit Cloud Deployment

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select your GitHub repository.
5. Select the `main` branch.
6. Set the main file to `app.py`.
7. Open the app's Secrets settings.
8. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

9. Deploy.

Do not upload `.streamlit/secrets.toml` to GitHub.

## 🔑 Environment Variable Alternative

The application also supports the standard environment variable:

```text
GROQ_API_KEY
```

The application checks Streamlit Secrets first and then the environment.

## 🧠 How the RAG Pipeline Works

### 1. PDF extraction

PyMuPDF extracts text page-by-page and preserves the page number as metadata.

### 2. Chunking

Text is split into approximately 500-word chunks with a 100-word overlap.

The overlap helps preserve context across chunk boundaries.

### 3. Embeddings

Each chunk is converted into a dense vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model handles tokenization internally.

### 4. FAISS retrieval

The embeddings are normalized and stored in a FAISS `IndexFlatIP` index.

Because the vectors are normalized, inner-product similarity corresponds to cosine similarity.

### 5. Question retrieval

The user's question is embedded using the same embedding model.

FAISS returns the most similar document chunks.

### 6. Generation

The retrieved chunks and user's question are sent to the Groq API.

The LLM generates an answer based on the retrieved context.

## ⚠️ Current Limitations

### Scanned PDFs

The current version extracts text from PDFs that contain a text layer.

Image-only/scanned PDFs require OCR and are not handled by this version.

### In-memory FAISS index

The FAISS index is created in memory after a PDF is uploaded.

If the application restarts, the user needs to upload the document again.

This is intentional for a simple deployment-friendly first version.

### Chunking

The current implementation uses word-based chunking rather than sentence-aware or semantic chunking.

## 🔮 Possible Improvements

Future versions could add:

- Chat history
- Multiple PDF support
- OCR for scanned PDFs
- Persistent vector storage
- Sentence-aware chunking
- Hybrid keyword + semantic search
- Reranking
- Streaming LLM responses
- Better citation formatting
- Document metadata
- PDF page previews
- Authentication
- Conversation memory
- Evaluation metrics for retrieval quality

## 📝 Notes

The embedding model runs locally in the Streamlit application. The retrieved context is sent to Groq for generation.

Do not upload confidential documents unless you understand the privacy and data-handling implications of the services used by your deployment.

## 📄 License

Add the license that matches how you want others to use your project. For example, an MIT License can be added as `LICENSE`.
