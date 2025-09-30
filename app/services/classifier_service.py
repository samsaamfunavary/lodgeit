import os
import asyncio
from typing import Dict, List
from openai import AsyncAzureOpenAI

from app.core.config import CONFIG
from app.services.azure_search import Azure_Search

class ClassifierService:
    """
    Service for classifying user queries using Azure OpenAI and parallel document fetching.
    """
    def __init__(self):
        """Initialize the classifier with the Azure OpenAI client."""
        try:
            self.openai_client = AsyncAzureOpenAI(
                api_key=CONFIG.AZURE_OPENAI_API_KEY,
                azure_endpoint=CONFIG.AZURE_OPEN_API_ENDPOINT,
                api_version=CONFIG.AZURE_OPENAI_API_VERSION
            )
            self.openai_deployment = CONFIG.AZURE_OPENAI_DEPLOYMENT
            self.azure_search = Azure_Search()
            self._load_index_descriptions()
            
        except Exception as e:
            print(f"Error initializing ClassifierService: {e}")
            raise
    
    def _load_index_descriptions(self):
        """Load all index descriptions from files."""
        # This function is correct and needs no changes.
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
        self.formatted_descriptions = "\n---\n".join(self.all_descriptions)
    
    async def _fetch_documents_from_all_indexes(self, user_query: str) -> Dict[str, List[Dict]]:
        """
        Asynchronously fetches top documents from all indexes in parallel.
        Uses asyncio.to_thread to avoid blocking the event loop with synchronous calls.
        """
        indexes_to_search = [
            ("lodgeit-help-guides", "default"),
            ("lodgeit-pricing", "default"),
            ("ato_complete_data2", "taxgenisemantic"),
            ("lodgeit-website", "default")
        ]
        
        async def fetch_for_index(index_name, semantic_config):
            try:
                if index_name == "lodgeit-website":
                    # Run the synchronous search function in a separate thread
                    return index_name, await asyncio.to_thread(self.azure_search.search_website_chunks, user_query, 2)
                elif index_name == "lodgeit-pricing":
                    return index_name, await asyncio.to_thread(self.azure_search.search_pricing_data, user_query, 2)
                else:
                    return index_name, await asyncio.to_thread(
                        self.azure_search.semantic_search_documents,
                        user_query, [], index_name, 2, semantic_config
                    )
            except Exception as e:
                print(f"Error fetching from index {index_name}: {e}")
                return index_name, []

        tasks = [fetch_for_index(name, config) for name, config in indexes_to_search]
        results = await asyncio.gather(*tasks)
        
        return {index_name: docs for index_name, docs in results}
    
    async def classify_query(self, user_query: str) -> str:
        """
        Classifies a user query using the high-accuracy RAG-for-RAG approach.
        """
        documents = await self._fetch_documents_from_all_indexes(user_query)
        
        document_context = ""
        for index_name, docs in documents.items():
            if docs:
                document_context += f"\n\n=== {index_name.upper()} SAMPLE DOCUMENTS ===\n"
                for i, doc in enumerate(docs, 1):
                    # Simplified formatting for clarity
                    content_snippet = (doc.get("content", "") or "")[:200]
                    document_context += f"Doc {i}: {doc.get('title', '')} - {content_snippet}...\n"

        prompt = f"""You are an expert query routing agent. Your task is to classify a user's query into one of the following categories and return only the corresponding index name.

Here are the descriptions of the available indexes:
{self.formatted_descriptions}

Here are sample documents from each index to help you classify the query:
{document_context}

---
User Query: "{user_query}"

Based on your analysis of the user's query and the sample documents from each index, which index contains the most relevant content to answer this query? Respond with the index name ONLY.
"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            classified_index = response.choices[0].message.content.strip().lower()
            
            mapping = self.get_index_mapping()
            for short_name, full_name in mapping.items():
                if short_name in classified_index or full_name in classified_index:
                    return full_name
            
            return "lodgeit-help-guides"
            
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return "lodgeit-help-guides"
    
    def get_index_mapping(self) -> Dict[str, str]:
        """Maps classifier short names to actual Azure Search index names."""
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
