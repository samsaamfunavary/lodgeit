import os
import google.generativeai as genai
from app.core.config import CONFIG
from app.services.azure_search import Azure_Search
from typing import Dict, List
import asyncio


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
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            
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
    
    async def _fetch_documents_from_all_indexes(self, user_query: str) -> Dict[str, List[Dict]]:
        """
        Asynchronously fetches top documents from all indexes in parallel.
        """
        indexes_to_search = [
            ("lodgeit-help-guides", "default"),
            ("lodgeit-pricing", "default"),
            ("ato_complete_data2", "taxgenisemantic"),
            ("lodgeit-website", "default")
        ]
        
        # --- ASYNCIO IMPLEMENTATION ---
        # 1. Create a list of tasks (coroutines) to run.
        tasks = []
        for index_name, semantic_config in indexes_to_search:
            # We create a small helper coroutine to keep the results organized
            async def fetch_and_package(name, config):
                try:
                    # Your existing logic for each index is now wrapped in this async function
                    if name == "lodgeit-website":
                        docs = self.azure_search.search_website_chunks(user_query, top=2) # Assuming this can be made async
                    elif name == "lodgeit-pricing":
                        docs = self.azure_search.search_pricing_data(user_query, max_results=2) # Assuming this can be made async
                    else:
                        docs = self.azure_search.semantic_search_documents(
                            keywords=user_query, class_filters=[], index_name=name, limit=2, semantic_configuration_name=config
                        )
                    return name, docs
                except Exception as e:
                    print(f"Error fetching from index {name}: {e}")
                    return name, [] # Return empty list on error

            tasks.append(fetch_and_package(index_name, semantic_config))

        # 2. Run all the tasks concurrently and wait for them all to complete.
        results = await asyncio.gather(*tasks)

        # 3. Process the results into the final dictionary.
        documents = {index_name: docs for index_name, docs in results}
        return documents
    
    async def classify_query(self, user_query: str) -> str:
        """
        Classifies a user query using the high-accuracy RAG-for-RAG approach.
        """
        documents = await self._fetch_documents_from_all_indexes(user_query)
        
        document_context = ""
        # ... (Your existing logic to build document_context string)

        prompt = f"""You are an expert query routing agent...
        
        Here are sample documents from each index:
        {document_context}

        User Query: "{user_query}"

        Respond with the index name ONLY."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            classified_index = response.choices[0].message.content.strip().lower()
            
            # Use the mapping to return the full, correct index name
            mapping = self.get_index_mapping()
            # Find the full name that matches the short name from the LLM
            for short_name, full_name in mapping.items():
                if short_name in classified_index or full_name in classified_index:
                    return full_name
            
            return "lodgeit-help-guides" # Default fallback
            
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return "lodgeit-help-guides"
    
    def get_index_mapping(self) -> Dict[str, str]:
        """Maps classifier short names to actual Azure Search index names."""
        return {
            "helpguide": "lodgeit-help-guides",
            "pricing": "lodgeit-pricing",
            "taxgenii": "ato_complete_data2", 
            "website": "lodgeit-website" # Corrected from logit-website
        }

    
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
            "website": "lodgeit-website"
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
