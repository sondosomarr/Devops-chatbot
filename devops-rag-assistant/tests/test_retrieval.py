import os
import sys
import unittest

# Add project root to sys.path so we can import src modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.retrieval import get_retriever, retrieve_relevant_context

class TestRetrieval(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load the vectorstore once for all tests."""
        cls.vectorstore = get_retriever()
        
    def test_vectorstore_loaded(self):
        """Ensure vector store is initialized properly."""
        self.assertIsNotNone(self.vectorstore, "Vector store should not be None; ensure ingestion has run.")

    def test_single_active_document_routing(self):
        """Ensure chunks retrieved belong ONLY to the active document."""
        if not self.vectorstore:
            self.skipTest("Vectorstore not loaded")
            
        test_query = "What is a Kubernetes Pod?"
        
        # We assume there are some PDFs ingested. Let's find an available doc_id or fake one.
        # If no documents are heavily populated, the distance threshold might block all results.
        # For this test, we just want to verify the filter works.
        mock_active_doc_id = "test_document.pdf"
        
        # Execute retrieve
        docs = retrieve_relevant_context(
            self.vectorstore, 
            test_query, 
            active_doc_ids=[mock_active_doc_id], 
            k=5, 
            distance_threshold=2.0 # lenient threshold to just check metadata filtering
        )
        
        # Verify metadata
        for doc in docs:
            self.assertEqual(
                doc.metadata.get("doc_id"), 
                mock_active_doc_id, 
                f"Retrieved chunk from {doc.metadata.get('doc_id')}, expected {mock_active_doc_id}"
            )
            
    def test_multi_active_document_routing(self):
        """Ensure chunks retrieved belong to one of the active documents."""
        if not self.vectorstore:
            self.skipTest("Vectorstore not loaded")
            
        test_query = "Docker vs Kubernetes?"
        active_docs = ["docker_guide.pdf", "k8s_guide.pdf"]
        
        docs = retrieve_relevant_context(
            self.vectorstore, 
            test_query, 
            active_doc_ids=active_docs, 
            k=5, 
            distance_threshold=2.0
        )
        
        for doc in docs:
            self.assertIn(
                doc.metadata.get("doc_id"), 
                active_docs, 
                "Retrieved chunk from a document not in the active doc list."
            )
            
if __name__ == "__main__":
    unittest.main()
