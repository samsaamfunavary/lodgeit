from openai import AzureOpenAI
from app.services.azure_search import Azure_Search
from app.services.classifier_service import ClassifierService
from app.core.config import CONFIG
import requests
import json
from typing import List, Dict, Any, AsyncGenerator
import textwrap
from openai import AsyncAzureOpenAI # <--- Change this import


class ChatService:
    def __init__(self):
        """Initializes the Chat Service and its clients."""
        
        # Initialize the Azure OpenAI client
        self.azure_search = Azure_Search()
        self.classifier = ClassifierService()
        self.openai_deployment = CONFIG.AZURE_OPENAI_DEPLOYMENT

        # --- THIS IS THE FIX ---
        # Use the Asynchronous client for async functions
        self.openai_client = AsyncAzureOpenAI(
            api_key=CONFIG.AZURE_OPENAI_API_KEY,
            azure_endpoint=CONFIG.AZURE_OPEN_API_ENDPOINT,
            api_version=CONFIG.AZURE_OPENAI_API_VERSION
        )
    def _classify_and_get_index(self, message: str, provided_index: str = None) -> str:
        """Classifies the user query to determine the appropriate index."""
        if provided_index:
            return provided_index
        return self.classifier.classify_query(message)

    def _create_rag_prompt(self, message: str, relevant_docs: List[Dict[str, Any]], index_name: str) -> str:
        """Creates a comprehensive RAG prompt for the LLM."""
        system_prompts = {
            "lodgeit-help-guides": textwrap.dedent("""\
                You are a LodgeiT Help Guides assistant. Answer using ONLY the provided context and reference documents.
                - Use clear, well-structured markdown.
                - If the context is insufficient, say so.
                - Cite documents by their TITLE with a clickable markdown link when a URL is present.
                """),
            "lodgeit-pricing": textwrap.dedent("""\
                You are a LodgeiT Pricing assistant. Answer using ONLY the pricing context provided.
                - Provide prices in AUD.
                - If comparing plans, provide a concise comparison.
                """),
            "ato_complete_data2": textwrap.dedent("""\
                You are a Taxgenii assistant for ATO operational guidance. Answer using ONLY the provided ATO/practice context.
                - Focus on ATO portals, agent workflows, and compliance.
                - When steps are relevant, provide clear, ordered instructions.
                """),
            "logit-website": textwrap.dedent("""\
                You are a LodgeiT Product & Website assistant. Answer using ONLY the provided context.
                - Explain what LodgeiT does, who it is for, and which features apply.
                - Must follow: Clear, readable markdown. Strictly do NOT include any images.
                """)
        }
        base_system_prompt = system_prompts.get(index_name, system_prompts["lodgeit-help-guides"])

        if not relevant_docs and index_name not in ["logit-website", "lodgeit-pricing"]:
            return f"{base_system_prompt}\n\n**User Question:** {message}\n\n**Note:** No relevant documents were found."

        context = ""
        if index_name == "lodgeit-pricing":
            try:
                pricing_results = self.azure_search.search_pricing_data(message, max_results=5)
                context = self.azure_search.format_pricing_results(pricing_results)
            except Exception as e:
                context = f"Error fetching pricing data: {e}"
        elif index_name == "logit-website":
            try:
                chunks = self.azure_search.search_website_chunks(message, top=3)
                parent_ids = {chunk.get("parent_id") for chunk in chunks if chunk.get("parent_id")}
                all_edges = [edge for parent_id in parent_ids for edge in self.azure_search.fetch_website_edges(parent_id, top=15)]
                context = self.azure_search.build_website_context_markdown(chunks, all_edges, question=message)
            except Exception as e:
                context = f"Error fetching website data: {e}"
        else:
            for i, doc in enumerate(relevant_docs, 1):
                context += f"**Document {i} - {doc.get('title', 'Untitled')}:**\n- Content: {doc.get('content', 'N/A')}\n"
                if doc.get('url'):
                    context += f"- URL: {doc.get('url')}\n"
                context += "\n"
        
        return f"""{base_system_prompt}

**Context from knowledge base:**
{context}

**User Question:** {message}

**Instructions:**
1. Use ONLY the provided context to answer. If the context is insufficient, politely say so.
2. All responses must be in properly formatted markdown.
3. Reference documents by their TITLE and include clickable markdown links if a URL is present.

**Answer:**"""

    def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Calls the Azure OpenAI API for a non-streaming response."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=messages,
                temperature=0,
                max_tokens=3500 
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    # In your ChatService class (app/services/chat_service.py)

    async def _call_openai_api_streaming(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Calls the Azure OpenAI API with streaming enabled and yields content chunks."""
        try:
            stream = await self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=messages,
                temperature=0,
                max_tokens=3500,
                stream=True 
            )
            
            # Asynchronously iterate over the stream chunks
            async for chunk in stream:
                # --- FIX: Check if the 'choices' list is not empty ---
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        # Yield each chunk of text as it arrives
                        yield content

        except Exception as e:
            print(f"Azure OpenAI streaming error: {e}")
            yield f"**Error:** An error occurred during the API call: {e}"

    def prepare_rag_for_streaming(self, message: str, hierarchy_filters: List[str], index_name: str = None, limit: int = 3) -> dict:
        """
        Performs fast, non-LLM steps: classification and document retrieval.
        Returns data needed for the RAG call.
        """
        if isinstance(message, list):
            message = " ".join(map(str, message))

        classified_index = self._classify_and_get_index(message, index_name)

        if classified_index == "ato_complete_data2":
            llm_response, relevant_docs = self._get_taxgenii_response(message)
            return {
                "is_external_api": True,
                "response": llm_response,
                "relevant_documents": relevant_docs,
                "classified_index": classified_index
            }

        relevant_docs = self.azure_search.semantic_search_documents(
            keywords=message,
            class_filters=hierarchy_filters,
            index_name=classified_index,
            limit=limit
        )
        
        system_prompt = self._create_rag_prompt(message, relevant_docs, classified_index)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        return {
            "is_external_api": False,
            "relevant_documents": relevant_docs,
            "messages": messages,
            "classified_index": classified_index
        }

    async def chat_with_rag_streaming(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Takes prepared messages and streams the LLM response."""
        async for chunk in self._call_openai_api_streaming(messages):
            yield chunk

    async def chat_with_rag(self, message: str, hierarchy_filters: List[str], index_name: str = None, limit: int = 3) -> Dict[str, Any]:
        """Non-streaming chat with RAG using the unified preparation logic."""
        prep_data = self.prepare_rag_for_streaming(
            message=message,
            hierarchy_filters=hierarchy_filters,
            index_name=index_name,
            limit=limit
        )

        if prep_data.get("is_external_api"):
            return {
                "response": prep_data["response"],
                "relevant_documents": prep_data["relevant_documents"],
                "query": message,
                "classified_index": prep_data.get("classified_index")
            }

        messages = prep_data.get("messages", [])
        llm_response = self._call_openai_api(messages)
        
        return {
            "response": llm_response,
            "relevant_documents": prep_data.get("relevant_documents", []),
            "query": message,
            "classified_index": prep_data.get("classified_index")
        }

    # --- External API Helper Methods ---
    def _get_taxgenii_response(self, message: str) -> tuple[str, list]:
        """Gets response from the external Taxgenii API."""
        try:
            return self._call_taxgenii_response_api(message)
        except Exception as e:
            return f"Error getting Taxgenii response: {str(e)}", []

            
    async def chat_with_taxgenii_streaming(self, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Calls the TaxGenii API and yields a true stream of its raw Markdown content,
        followed by the reference documents.
        """
        try:
            url = "https://api.taxgenii.lodgeit.net.au/api/chat/get-response-message"
            payload = {"username": "user", "prompt": message, "learn": False, "stream": True}
            
            # Use requests with streaming. We use 'with' to ensure the connection is closed.
            with requests.post(url, json=payload, timeout=60, stream=True) as response:
                response.raise_for_status()
                
                # --- CORRECTED STREAM HANDLING ---
                # The API streams raw text, so we iterate over lines/content.
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        # Yield each piece of text as a content chunk
                        yield {"type": "content", "data": line + "\n"}
                
                # --- PROCESS HEADERS AFTER STREAM ---
                # The headers are only fully available after the stream is consumed.
                reference_docs = []
                if x_metainfo := response.headers.get('x-metainfo'):
                    try:
                        metainfo = json.loads(x_metainfo)
                        if 'urls' in metainfo:
                            for url_info in metainfo['urls']:
                                reference_docs.append({
                                    "title": url_info.get('hierrachy', ''), # Note API typo
                                    "url": url_info.get('url', ''),
                                })
                    except json.JSONDecodeError:
                        print(f"Failed to parse x-metainfo header: {x_metainfo}")
                
                # Yield the collected references as the final event in the stream
                yield {"type": "references", "data": reference_docs}

        except Exception as e:
            yield {"type": "content", "data": f"**Error:** TaxGenii API error: {e}"}

        
    
    def _call_taxgenii_response_api(self, message: str) -> tuple[str, list]:
        """Makes the HTTP call to the Taxgenii API."""
        try:
            url = "https://api.taxgenii.lodgeit.net.au/api/chat/get-response-message"
            payload = {"username": "user", "prompt": message, "learn": False, "stream": True}
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            reference_docs = []
            if x_metainfo := response.headers.get('x-metainfo'):
                try:
                    metainfo = json.loads(x_metainfo)
                    if 'urls' in metainfo:
                        for url_info in metainfo['urls']:
                            reference_docs.append({
                                "title": url_info.get('hierrachy', ''),
                                "url": url_info.get('url', ''),
                            })
                except json.JSONDecodeError:
                    print(f"Failed to parse x-metainfo header: {x_metainfo}")
            
            try:
                result = response.json()
                return result.get("response", str(result)), reference_docs
            except json.JSONDecodeError:
                return response.text, reference_docs
        except Exception as e:
            raise Exception(f"Taxgenii API error: {str(e)}")