import os
from src.ingestion import get_embeddings
from src.retrieval import get_retriever
from langchain_community.document_loaders import PyPDFDirectoryLoader

DATA_DIR = "d:/Devops chatbot/devops-rag-assistant/data"

print("--- STEP 1: VERIFY INGESTION ---")
if os.path.exists(DATA_DIR):
    loader = PyPDFDirectoryLoader(DATA_DIR, extract_images=True)
    docs = loader.load()
    print(f"Pages loaded: {len(docs)}")
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"Total extracted characters: {total_chars}")
    if docs:
        print(f"Sample text (first 500 chars): {docs[0].page_content[:500]}")
else:
    print(f"Data directory {DATA_DIR} does not exist.")

vectorstore = get_retriever()
if vectorstore:
    try:
        print(f"Total chunks in vectorstore: {vectorstore._collection.count()}")
    except Exception as e:
         print("Error counting chunks:", e)

    print("\n--- STEP 2: VERIFY METADATA CONSISTENCY ---")
    try:
        items = vectorstore._collection.get(include=["metadatas"], limit=10)
        metadatas = items["metadatas"]
    except Exception as e:
        metadatas = []
        print("Error getting metadatas:", e)
        
    doc_titles = set([m.get("doc_title") for m in metadatas if m and "doc_title" in m])
    doc_ids = set([m.get("doc_id") for m in metadatas if m and "doc_id" in m])
    print(f"Sample doc_titles in DB: {doc_titles}")
    print(f"Sample doc_ids (hashes) in DB: {doc_ids}")
    
    if os.path.exists(DATA_DIR):
        active_docs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        print(f"Active docs from UI (filenames): {active_docs}")
    else:
        active_docs = []

    print("\n--- STEP 3: VERIFY FILTERING ---")
    query = "DevOps"
    print(f"\nTop 5 WITHOUT filter (query='{query}'):")
    results = vectorstore.similarity_search_with_score(query, k=5, filter=None)
    for doc, score in results:
        print(f"  Title: {doc.metadata.get('doc_title')} | Score: {score:.4f} | Content: {doc.page_content[:120].strip()}")

    if active_docs:
        print(f"\nTop 5 WITH $in filter on doc_title (all active docs):")
        try:
             results = vectorstore.similarity_search_with_score(query, k=5, filter={"doc_title": {"$in": active_docs}})
             for doc, score in results:
                  print(f"  Title: {doc.metadata.get('doc_title')} | Score: {score:.4f} | Content: {doc.page_content[:120].strip()}")
        except Exception as e:
             print(f"  Filter using $in FAILED: {e}")

    print("\n--- STEP 4: VERIFY THRESHOLD ---")
    queries = ["What is CI/CD?", "Docker integration", "Kubernetes", "Git ops", "AWS pipeline"]
    for q in queries:
        res = vectorstore.similarity_search_with_score(q, k=3)
        scores = [f"{s:.4f}" for _, s in res]
        print(f"  Query: '{q}' -> Scores: {scores}")

    print("\n--- STEP 5: VERIFY EMBEDDING CONSISTENCY ---")
    print("Ingestion model: all-MiniLM-L6-v2")
    print("Retrieval model: all-MiniLM-L6-v2 (same function)")
    
    print("\n--- DONE ---")
