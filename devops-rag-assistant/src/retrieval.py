import os
from langchain_community.vectorstores import Chroma
from src.utils import get_logger
from src.ingestion import get_embeddings

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(BASE_DIR) != "devops-rag-assistant":
    BASE_DIR = os.path.join(os.getcwd(), "devops-rag-assistant")

VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore_v3")

def get_retriever():
    """Returns the Chroma vectorstore directly, so we can do custom similarity searches."""
    if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
        logger.warning(f"Vector store directory {VECTORSTORE_DIR} is empty or missing. Please run ingestion first.")
        return None
        
    logger.info("Loading Chroma vector store...")
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_relevant_context(vectorstore, query: str, active_doc_titles: list, k: int = 5, distance_threshold: float = 1.3):
    """
    Retrieves the top_k chunks from the vector store, explicitly filtering by active_doc_titles.
    Returns both the threshold-filtered docs and the raw docs (for fallback/short-circuit logic).
    """
    logger.info(f"Querying for: '{query}' within active docs: {active_doc_titles}")
    
    filter_dict = None
    if active_doc_titles:
        filter_dict = {"doc_title": {"$in": active_doc_titles}}
        
    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter=filter_dict
    )
    
    logger.info(f"Active docs have {len(results)} chunks returned before threshold.")
    
    valid_docs = []
    raw_docs = []
    for i, (doc, score) in enumerate(results):
        raw_docs.append(doc)
        logger.info(f"Retrieved chunk {i+1} | Score: {score:.3f} | Title: {doc.metadata.get('doc_title')} | Page: {doc.metadata.get('page')}")
        
        # Relevance Gate
        if score <= distance_threshold:
            valid_docs.append(doc)
        else:
            logger.warning(f"Chunk REJECTED: Score {score:.3f} is above threshold {distance_threshold}")
            
    logger.info(f"Top chunks AFTER threshold: {len(valid_docs)}")
    return valid_docs, raw_docs

if __name__ == "__main__":
    vectorstore = get_retriever()
    if vectorstore:
        query = "What is a Kubernetes pod?"
        valid, raw = retrieve_relevant_context(vectorstore, query, active_doc_titles=[], k=3)
        print(f"Total valid docs retrieved: {len(valid)}")
        for doc in valid:
            print(f"- {doc.metadata.get('doc_title')} (Page {doc.metadata.get('page')}): {doc.page_content[:50]}...")
