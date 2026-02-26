from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.retrieval import get_retriever, retrieve_relevant_context
from src.utils import get_logger

logger = get_logger(__name__)

def get_llm():
    logger.info("Initializing Ollama (qwen2.5:7b)...")
    return Ollama(model="qwen2.5:7b")

def get_rag_chain():
    """
    Returning vectorstore and llm to be executed imperatively due to active_doc_ids
    dynamic requirement and relevance threshold short-circuiting.
    """
    vectorstore = get_retriever()
    if not vectorstore:
        return None, None
        
    llm = get_llm()
    return vectorstore, llm
    
template = """You are a senior DevOps RAG/LLM engineer assistant.
You must answer ONLY from the provided CONTEXT. 
If the context does not contain strictly enough information, you may attempt to answer using the context provided, but note your limitations.
If it is completely unrelated, then you must categorically refuse to answer by responding exactly: "I can’t find relevant info in the currently selected document."
You must cite the document title and page number for your evidence.

Response Format Guidelines:
- Commands / Steps: (List actionable steps if any)
- Explanation: (Short explanation)
- Evidence: (Quoted snippet <= 25 words under the citation: `[Source: {{doc_title}} | Page {{page_number}}]`)

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

def ask_question(query: str, active_doc_titles: list):
    vectorstore, llm = get_rag_chain()
    if not vectorstore or not llm:
        return "RAG chain not initialized. Please ensure documents are uploaded and processed.", []
        
    # 1. Retrieval + Relevance Gate
    valid_docs, raw_docs = retrieve_relevant_context(vectorstore, query, active_doc_titles=active_doc_titles, k=5, distance_threshold=1.3)
    
    docs_to_use = valid_docs
    
    # 2. Smart Short Circuit Logic
    if not valid_docs:
        logger.warning(f"No chunks passed relevance gate.")
        # Check if we at least got raw chunks back (meaning the active doc is valid and queried)
        if raw_docs:
            logger.info("Evaluating fallback chunk due to strict threshold drop...")
            best_chunk = raw_docs[0]
            
            # Simple heuristic keyword overlap matching
            query_words = set(query.lower().split())
            chunk_words = set(best_chunk.page_content.lower().split())
            
            common_words = query_words.intersection(chunk_words)
            # Remove generic stop words if needed, but for now just length check
            if len(common_words) > 0:
                logger.info(f"Fallback chunk has {len(common_words)} matching terms with query. Proceeding to LLM.")
                docs_to_use = [best_chunk]
            else:
                logger.warning("Fallback chunk has no matching keywords. Aborting.")
                return "I can’t find relevant info in the currently selected document.", []
        else:
             return "I can’t find relevant info in the currently selected document.", []
        
    # 3. Formatting Context
    formatted_context_parts = []
    for doc in docs_to_use:
        doc_title = doc.metadata.get('doc_title', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        formatted_context_parts.append(f"--- Document: {doc_title} | Page {page} ---\n{doc.page_content}\n")
        
    formatted_context = "\n".join(formatted_context_parts)
    logger.info(f"Assembled context size: {len(formatted_context)} characters")
    
    # 4. Generate Response
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": formatted_context, "question": query})
    logger.info("Response generated.")
    
    return response, [doc.metadata for doc in docs_to_use]

if __name__ == "__main__":
    resp, meta = ask_question("What is a Docker container?", active_doc_titles=[])
    print("\nResponse:\n", resp)
