import os
import google.generativeai as genai
from app.core.config import CONFIG
from app.services.azure_search import Azure_Search
from typing import Dict, List

class ClassifierService:
    """
    Service for classifying user queries to determine which RAG index to use.
    Maps user queries to appropriate indexes: lodgeit-help-guides, pricing, Taxgenii, or logit-website
    """
    
    def __init__(self):
        """Initialize the classifier with Gemini API"""
        try:
            # Use API key from config
            api_key = CONFIG.GEMINI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY not configured!")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Initialize Azure Search service for document retrieval
            self.azure_search = Azure_Search()
            
            # Load index descriptions
            self._load_index_descriptions()
            
        except Exception as e:
            print(f"Error initializing ClassifierService: {e}")
            raise
    
    def _load_index_descriptions(self):
        """Load all index descriptions from files"""
        descriptions_dir = "index_descriptions"
        index_names = ["helpguide", "pricing", "taxgenii", "website"]
        
        self.all_descriptions = []
        try:
            for name in index_names:
                filepath = os.path.join(descriptions_dir, f"{name}.txt")
                with open(filepath, 'r') as f:
                    self.all_descriptions.append(f.read().strip())
        except FileNotFoundError as e:
            print(f"Error: Description file not found - {e}")
            raise
        
        # Join descriptions for the prompt
        self.formatted_descriptions = "\n---\n".join(self.all_descriptions)
    
    def _fetch_documents_from_all_indexes(self, user_query: str) -> Dict[str, List[Dict]]:
        """
        Fetch top 2 documents from all indexes to provide context for classification.
        
        Args:
            user_query: The user's query
            
        Returns:
            Dict mapping index names to lists of documents
        """
        documents = {}
        
        # Define indexes and their search configurations
        indexes = [
            ("lodgeit-help-guides", "default"),
            ("lodgeit-pricing", "default"),
            ("ato_complete_data2", "taxgenisemantic"),
            ("logit-website", "default")  # Will be handled specially
        ]
        
        for index_name, semantic_config in indexes:
            try:
                if index_name == "logit-website":
                    # Special handling for website index - search chunks and edges
                    chunks = self.azure_search.search_website_chunks(user_query, top=2)
                    parent_id = chunks[0].get("parent_id") if chunks else None
                    edges = self.azure_search.fetch_website_edges(parent_id, top=5) if parent_id else []
                    
                    # Format chunks as documents
                    docs = []
                    for chunk in chunks:
                        docs.append({
                            "title": chunk.get("title", ""),
                            "content": chunk.get("content", "")[:300] + "..." if len(chunk.get("content", "")) > 300 else chunk.get("content", ""),
                            "url": chunk.get("url", ""),
                            "hierarchy": chunk.get("hierarchy", "")
                        })
                    
                    # Add edges as additional context
                    if edges:
                        edge_context = "Related concepts: " + ", ".join([f"{e.get('source_label', '')} -> {e.get('target_label', '')}" for e in edges[:3]])
                        if docs:
                            docs[0]["content"] += f"\n\n{edge_context}"
                    
                    documents[index_name] = docs
                    
                else:
                    # Use semantic search for other indexes
                    docs = self.azure_search.semantic_search_documents(
                        keywords=user_query,
                        class_filters=[],
                        index_name=index_name,
                        limit=2,
                        semantic_configuration_name=semantic_config
                    )
                    
                    # Format documents with title and content snippet
                    formatted_docs = []
                    for doc in docs:
                        formatted_docs.append({
                            "title": doc.get("title", ""),
                            "content": doc.get("content", "")[:300] + "..." if len(doc.get("content", "")) > 300 else doc.get("content", ""),
                            "url": doc.get("url", ""),
                            "hierarchy": doc.get("hierarchy", "")
                        })
                    
                    documents[index_name] = formatted_docs
                    
            except Exception as e:
                print(f"Error fetching documents from {index_name}: {e}")
                documents[index_name] = []
        
        return documents
    
    def classify_query(self, user_query: str) -> str:
        """
        Classify a user query to determine the appropriate RAG index.
        Now includes document context from all indexes for better classification.
        
        Args:
            user_query: The user's question/query
            
        Returns:
            str: The classified index name (helpguide, pricing, taxgenii, or website)
        """
        # Fetch documents from all indexes for context
        documents = self._fetch_documents_from_all_indexes(user_query)
        
        # Build document context for the prompt
        document_context = ""
        for index_name, docs in documents.items():
            if docs:
                document_context += f"\n\n=== {index_name.upper()} INDEX SAMPLE DOCUMENTS ===\n"
                for i, doc in enumerate(docs, 1):
                    document_context += f"\nDocument {i}:\n"
                    document_context += f"Title: {doc.get('title', 'No title')}\n"
                    document_context += f"Content: {doc.get('content', 'No content')}\n"
                    if doc.get('hierarchy'):
                        document_context += f"Hierarchy: {doc.get('hierarchy')}\n"
        
        prompt = f"""You are an expert query routing agent. Your task is to classify a user's query into one of the following categories and return only the corresponding index name.

Analyze the user's query to determine their core intent. Is it a "how-to" question, a pricing question, a general tax law question, or a high-level product feature question?

Here are the descriptions of the available indexes:

{self.formatted_descriptions}

{document_context}

---

User Query: "{user_query}"

Based on your analysis of the user's query and the sample documents from each index, which index contains the most relevant content to answer this query? Respond with the index name ONLY.
"""
        
        try:
            response = self.model.generate_content(prompt)
            classified_index = response.text.strip().lower()

            # Handle both short and full index names
            if classified_index in ["lodgeit-help-guides", "helpguide"]:
                return "lodgeit-help-guides"
            elif classified_index in ["lodgeit-pricing", "pricing"]:
                return "lodgeit-pricing"
            elif classified_index in ["ato_complete_data2", "taxgenii"]:
                return "ato_complete_data2"
            elif classified_index in ["logit-website", "website"]:
                return "logit-website"
            else:
                # Default fallback
                return "lodgeit-help-guides"
            
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            # Return default index on error
            return "lodgeit-help-guides"
    
    def get_index_mapping(self) -> Dict[str, str]:
        """
        Get the mapping of classifier outputs to actual Azure Search index names.
        
        Returns:
            Dict mapping classifier output to Azure Search index name
        """
        return {
            "helpguide": "lodgeit-help-guides",
            "pricing": "lodgeit-pricing",
            "taxgenii": "ato_complete_data2", 
            "website": "logit-website"
        }
    
    def test_classification(self, test_queries: List[str]) -> Dict[str, str]:
        """
        Test the classifier with a list of queries.
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Dict mapping queries to classified indexes
        """
        results = {}
        for query in test_queries:
            classified_index = self.classify_query(query)
            results[query] = classified_index
        return results
