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
        items = vectorstore._collection.get(include=["metadatas"])
        metadatas = items["metadatas"]
    except Exception as e:
        metadatas = []
        print("Error getting metadatas:", e)
        
    doc_ids = set([m.get("doc_id") for m in metadatas if m and "doc_id" in m])
    print(f"Distinct doc_ids in DB: {doc_ids}")
    
    if os.path.exists(DATA_DIR):
        active_docs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        print(f"Active docs from UI (list_uploaded_pdfs): {active_docs}")
    else:
        active_docs = []

    print("\n--- STEP 3: VERIFY FILTERING ---")
    query = "DevOps" # known valid query
    results = vectorstore.similarity_search_with_score(query, k=5, filter=None)
    print("Top 5 without filter:")
    for doc, score in results:
        print(f"Doc: {doc.metadata.get('doc_id')} | Score: {score} | Content: {doc.page_content[:120].strip()}")

    if active_docs:
        print("\nTop 5 WITH string filter (first active doc):")
        results = vectorstore.similarity_search_with_score(query, k=5, filter={"doc_id": active_docs[0]})
        for doc, score in results:
             print(f"Doc: {doc.metadata.get('doc_id')} | Score: {score}")
             
        print("\nTop 5 WITH $in filter (all active docs):")
        try:
             results = vectorstore.similarity_search_with_score(query, k=5, filter={"doc_id": {"$in": active_docs}})
             for doc, score in results:
                  print(f"Doc: {doc.metadata.get('doc_id')} | Score: {score}")
        except Exception as e:
             print(f"Filter using $in failed: {e}")

    print("\n--- STEP 4: VERIFY THRESHOLD ---")
    queries = ["What is CI/CD?", "Docker integration", "Kubernetes", "Git ops", "AWS pipeline"]
    for q in queries:
        res = vectorstore.similarity_search_with_score(q, k=3)
        scores = [s for _, s in res]
        print(f"Query: '{q}' -> Scores: {scores}")

    print("\n--- STEP 5: VERIFY EMBEDDING CONSISTENCY ---")
    print(f"Ingestion explicit model: all-MiniLM-L6-v2")
