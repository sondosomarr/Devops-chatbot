import streamlit as st
import os
from src.ingestion import ingest_documents
from src.generation import ask_question

st.set_page_config(page_title="DevOps RAG Assistant", page_icon="🤖")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def list_uploaded_pdfs():
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]

def main():
    st.title("🤖 DevOps RAG Assistant")
    st.write("Upload your DevOps PDFs and ask questions! Powered by Qwen 2.5 7B & ChromaDB.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Process Documents"):
            if uploaded_files:
                if not os.path.exists(DATA_DIR):
                    os.makedirs(DATA_DIR)
                with st.spinner("Saving files and building vector index..."):
                    # Save files
                    for file in uploaded_files:
                        file_path = os.path.join(DATA_DIR, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                    
                    # Run ingestion
                    try:
                        ingest_documents()
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
            else:
                st.warning("Please upload some PDFs first.")
                
        st.divider()
        st.header("Active Documents")
        st.write("Select which PDFs to query against.")
        available_docs = list_uploaded_pdfs()
        
        # User explicitly chooses the active documents
        active_docs = st.multiselect(
            "Select Active Documents",
            options=available_docs,
            default=available_docs,
            help="Queries will only search these selected documents. Unselect documents you want to ignore."
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Retrieved Sources"):
                    for src in message["sources"]:
                        st.caption(f"Doc: {src.get('doc_title', 'Unknown')} | Page: {src.get('page')}")
                        
    # Chat input
    if prompt := st.chat_input("What is your DevOps question?"):
        
        if not active_docs:
            st.warning("Please select at least one Active Document from the sidebar before asking a question.")
            return

        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = ask_question(prompt, active_docs)
                    st.markdown(response)
                    
                    # Persist response and sources in history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                    if sources:
                        with st.expander("View Retrieved Sources"):
                            for src in sources:
                                st.caption(f"Doc: {src.get('doc_title', 'Unknown')} | Page: {src.get('page')}")
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
