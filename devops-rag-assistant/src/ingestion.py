import os

import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.utils import get_logger

logger = get_logger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(BASE_DIR) != "devops-rag-assistant":
    BASE_DIR = os.path.join(os.getcwd(), "devops-rag-assistant")

DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore_v3")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def ocr_pdf(pdf_path: str) -> list:
    """
    OCR fallback for image-based PDFs.
    Converts each page to an image using pdf2image, then runs Tesseract OCR.
    Returns a list of Document objects with page_content and metadata.
    """
    from pdf2image import convert_from_path
    import pytesseract

    # Configure Tesseract paths (conda install location)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    tesseract_bin = os.path.join(conda_prefix, "Library", "bin", "tesseract.exe")
    tessdata_dir = os.path.join(conda_prefix, "Library", "share", "tessdata")
    
    if os.path.exists(tesseract_bin):
        pytesseract.pytesseract.tesseract_cmd = tesseract_bin
    if os.path.exists(tessdata_dir):
        os.environ["TESSDATA_PREFIX"] = tessdata_dir

    logger.info(f"Running OCR on: {pdf_path}")
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        logger.error(f"pdf2image failed for {pdf_path}: {e}")
        return []

    ocr_docs = []
    for page_num, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                }
            )
            ocr_docs.append(doc)
            logger.info(f"  OCR page {page_num}: {len(text)} chars extracted")
        else:
            logger.warning(f"  OCR page {page_num}: no text found")

    logger.info(f"OCR complete for {os.path.basename(pdf_path)}: {len(ocr_docs)} pages with text")
    return ocr_docs


def load_all_pdfs(data_dir: str) -> list:
    """
    Load all PDFs from data_dir. For each PDF:
    1. Try PyPDFLoader (text extraction).
    2. If total extracted chars == 0, fall back to OCR.
    """
    all_docs = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDFs found in {data_dir}.")
        return []

    for filename in pdf_files:
        pdf_path = os.path.join(data_dir, filename)
        logger.info(f"Processing: {filename}")

        # Step 1: Try text extraction
        try:
            loader = PyPDFLoader(pdf_path, extract_images=True)
            docs = loader.load()
        except Exception as e:
            logger.error(f"PyPDFLoader failed for {filename}: {e}")
            docs = []

        total_chars = sum(len(d.page_content.strip()) for d in docs)
        logger.info(f"  PyPDFLoader: {len(docs)} pages, {total_chars} chars")

        # Step 2: If no text, fall back to OCR
        if total_chars == 0:
            logger.warning(f"  {filename} appears to be image-based (0 chars). Falling back to OCR...")
            docs = ocr_pdf(pdf_path)

        all_docs.extend(docs)

    return all_docs


def _compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def _get_existing_doc_info(vectorstore) -> dict:
    """
    Returns a dict of {doc_title: doc_id} for all documents already in the vectorstore.
    """
    try:
        items = vectorstore._collection.get(include=["metadatas"])
        metadatas = items["metadatas"]
    except Exception:
        return {}
    
    existing = {}
    for m in metadatas:
        if m and "doc_title" in m and "doc_id" in m:
            existing[m["doc_title"]] = m["doc_id"]
    return existing


def _remove_doc_chunks(vectorstore, doc_title: str):
    """Remove all chunks for a given doc_title from the vectorstore."""
    try:
        items = vectorstore._collection.get(
            where={"doc_title": doc_title},
            include=[]
        )
        ids_to_delete = items["ids"]
        if ids_to_delete:
            vectorstore._collection.delete(ids=ids_to_delete)
            logger.info(f"  Removed {len(ids_to_delete)} old chunks for '{doc_title}'")
    except Exception as e:
        logger.error(f"  Failed to remove old chunks for '{doc_title}': {e}")


def load_single_pdf(pdf_path: str) -> list:
    """
    Load a single PDF. Try text extraction first, fall back to OCR if needed.
    """
    filename = os.path.basename(pdf_path)
    logger.info(f"Processing: {filename}")

    try:
        loader = PyPDFLoader(pdf_path, extract_images=True)
        docs = loader.load()
    except Exception as e:
        logger.error(f"PyPDFLoader failed for {filename}: {e}")
        docs = []

    total_chars = sum(len(d.page_content.strip()) for d in docs)
    logger.info(f"  PyPDFLoader: {len(docs)} pages, {total_chars} chars")

    if total_chars == 0:
        logger.warning(f"  {filename} appears to be image-based (0 chars). Falling back to OCR...")
        docs = ocr_pdf(pdf_path)

    return docs


def ingest_documents():
    """
    Incremental ingestion:
    - Opens existing vectorstore (or creates a new one).
    - Checks which doc_titles are already indexed and their doc_id (hash).
    - Only processes NEW or CHANGED PDFs.
    - Removes old chunks for changed PDFs before adding new ones.
    """
    logger.info(f"Loading PDFs from {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    embeddings = get_embeddings()

    # Open or create vectorstore
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

    # Get existing documents in the vectorstore
    existing_docs = _get_existing_doc_info(vectorstore)
    logger.info(f"Existing documents in vectorstore: {list(existing_docs.keys())}")

    # Scan data directory for PDFs
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDFs found in {DATA_DIR}.")
        return

    new_splits = []
    for filename in pdf_files:
        pdf_path = os.path.join(DATA_DIR, filename)
        current_hash = _compute_file_hash(pdf_path)

        if filename in existing_docs:
            if existing_docs[filename] == current_hash:
                logger.info(f"SKIP '{filename}' — already indexed with same hash.")
                continue
            else:
                logger.info(f"UPDATE '{filename}' — file changed (hash mismatch). Re-indexing...")
                _remove_doc_chunks(vectorstore, filename)
        else:
            logger.info(f"NEW '{filename}' — not yet indexed. Adding...")

        # Load the PDF
        docs = load_single_pdf(pdf_path)
        if not docs:
            logger.warning(f"  No text extracted from '{filename}'. Skipping.")
            continue

        # Chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(docs)
        splits = [s for s in splits if s.page_content.strip()]

        # Tag metadata
        for s in splits:
            s.metadata["doc_id"] = current_hash
            s.metadata["doc_title"] = filename

        logger.info(f"  Created {len(splits)} chunks for '{filename}'")
        new_splits.extend(splits)

    if not new_splits:
        logger.info("No new documents to index. Vectorstore is up to date.")
        return

    logger.info(f"Adding {len(new_splits)} new chunks to the vectorstore...")
    vectorstore.add_documents(new_splits)
    logger.info("Ingestion complete. Vector store updated.")

if __name__ == "__main__":
    ingest_documents()

