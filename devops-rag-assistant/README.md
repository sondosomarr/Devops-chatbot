# DevOps RAG Assistant
A DevOps Q&A Assistant using Retrieval-Augmented Generation (RAG) with local execution.

## Features
- **Local Execution:** Uses ChromaDB for vector storage and Ollama (qwen2.5:7b) for generation.
- **English Only:** Focused specifically on DevOps documentation in English.
- **Minimal UI:** Streamlit interface for uploading PDFs and asking questions.

## Setup
1. **Ensure Ollama is installed** and the `qwen2.5:7b` model is pulled:
   ```bash
   ollama pull qwen2.5:7b
   ```
2. **Setup Python environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
To run the Streamlit UI:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.
You can upload DevOps PDFs (like Kubernetes & Docker documentation) via the sidebar.

## Project Structure
- `data/`: Contains uploaded PDF documents.
- `src/`: 
  - `ingestion.py`: Handles loading PDFs, chunking, and embedding creation.
  - `retrieval.py`: Setup for ChromaDB retriever.
  - `generation.py`: LangChain RAG pipeline connecting ChromaDB and Ollama.
  - `utils.py`: Common helper functions (logging, embeddings).
- `vectorstore/`: Persisted ChromaDB embeddings.
- `app.py`: Streamlit frontend.
