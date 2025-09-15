from openai import AzureOpenAI
from app.services.azure_search import Azure_Search
from app.services.classifier_service import ClassifierService
from app.core.config import CONFIG
import requests
import json
from typing import List, Dict, Any, AsyncGenerator

class ChatService:
    def __init__(self):
        self.azure_search = Azure_Search()
        self.classifier = ClassifierService()
        # Use Google Gemini API
        self.gemini_api_key = CONFIG.GEMINI_API_KEY
        self.gemini_model = CONFIG.GEMINI_MODEL
        self.gemini_base_url = CONFIG.GEMINI_BASE_URL
    
    def _classify_and_get_index(self, message: str, provided_index: str = None) -> str:
        """
        Classify the user query to determine the appropriate index.
        If an index is explicitly provided, use that instead.
        
        Args:
            message: User's query message
            provided_index: Optional explicit index name
            
        Returns:
            str: The appropriate index name for the query
        """
        if provided_index:
            return provided_index
        
        # Use classifier to determine the best index
        return self.classifier.classify_query(message)
    
    def _create_rag_prompt(self, message: str, relevant_docs: List[Dict[str, Any]], index_name: str = "lodgeit-help-guides") -> str:
        """Create a RAG-enhanced prompt for the LLM with markdown formatting"""
        
        # Define system prompts for different indexes
        system_prompts = {
            "lodgeit-help-guides": """You are a LodgeiT Help Guides assistant. Answer using ONLY the provided context and reference documents.

Formatting and behavior:
- Use clear, well-structured markdown with headings, lists, and links.
- If the context is insufficient, say so and suggest next steps or keywords.
- Cite documents by their TITLE with a clickable markdown link when a URL is present.
- When an image is relevant, include it inline where it best supports the explanation using: ![Alt text](Image_URL)
- Keep tone professional, concise, and accurate. Do not invent facts or documents.

Must follow:
- Proper markdown formatting and spacing; clear line breaks.
- Correct list/numbered list formatting.
- Helpful, actionable steps for how-to and troubleshooting responses.
- When images are relevant: Insert them like an article after the bullet/paragraph at the point they are most useful in your explanation.
- Use standard markdown format: ![Alt text](Image_URL)
- The image should appear big and natural like in an article when markdown is rendered (do not shrink or place in a separate section).
- In the markdown it should be properly spaced after every image and paragraph add multiple line breaks.
""",
            "lodgeit-pricing": """You are a LodgeiT Pricing assistant. Answer using ONLY the pricing context provided.

Formatting and behavior:
- Provide prices in AUD; mention GST where applicable.
- If comparing plans, provide a concise comparison and call out key differences.
- When a plan is asked about, include the plan name, price, included allowances, notable features, and overage/extra usage fees.
- If bundles or e-sign packages are relevant, list the available tiers and per-unit costs.
- If information is missing from the context, state that it is not available.

Must follow:
- Use clean markdown with sections, bullet lists, and tables when helpful.
- Do not include non-pricing topics; redirect such questions to the appropriate resource.
""",
            "ato_complete_data2": """You are a Taxgenii assistant for ATO operational guidance. Answer using ONLY the provided ATO/practice context.

Formatting and behavior:
- Focus on ATO portals, agent workflows, lodgment programs, client-to-agent linking, deferrals, POI, RAM/myGovID, and compliance.
- When steps are relevant, provide clear, ordered step-by-step instructions.
- Reference named ATO areas or programs as they appear in the context. Include links when URLs are present.
- If the question is outside operational guidance (e.g., tax law interpretations), state it is out of scope and suggest where to look.

Must follow:
- Professional, succinct markdown with headings and lists.
- No speculation; do not provide financial or legal advice.
""",
            "logit-website": """You are a LodgeiT Product & Website assistant. Answer using ONLY the provided product/features/resources context.

Formatting and behavior:
- Explain what LodgeiT does, who it is for, and which features/integrations apply.
- Use role-oriented framing when relevant (Accountants, Bookkeepers, Businesses/Family Offices).
- Link to resources (Knowledge Base, YouTube, Workshops) when URLs are present.
- Do NOT discuss pricing; direct pricing questions to the pricing resources.

Must follow:
- Clear, readable markdown with headings and bullets.
- Include inline links and images from the context when useful.
"""
        }

        # Get the appropriate system prompt for the index
        base_system_prompt = system_prompts.get(index_name, system_prompts["lodgeit-help-guides"])
        
        if not relevant_docs:
            system_prompt = f"""{base_system_prompt}

**User Question:** {message}

**Note:** No relevant documents were found in the knowledge base for your question.

**Instructions:**
1. Acknowledge that no specific documentation was found
2. Provide general guidance if possible based on your knowledge of {index_name}
3. Suggest that the user try different keywords or check if the topic exists in the knowledge base
4. Be helpful and professional
5. Give the image as markdown also if you find the image relevant to user query for the image description present just after the image in reference documents

**Answer:**"""
            return system_prompt
        
        context = ""
        if index_name == "lodgeit-pricing":
            try:
                pricing_results = self.azure_search.search_pricing_data(message, max_results=5)
                context = self.azure_search.format_pricing_results(pricing_results)
            except Exception:
                context = ""
        elif index_name == "logit-website":
            try:
                chunks = self.azure_search.search_website_chunks(message, top=5)
                parent_id = chunks[0].get("parent_id") if chunks else None
                edges = self.azure_search.fetch_website_edges(parent_id, top=20) if parent_id else []
                context = self.azure_search.build_website_context_markdown(chunks, edges, question=message)
            except Exception:
                context = ""
        else:
            for i, doc in enumerate(relevant_docs, 1):
                doc_title = doc.get('title', 'Untitled Document')
                doc_url = doc.get('url', '')
                context += f"**Document {i} - {doc_title}:**\n"
                context += f"- **Title:** {doc_title}\n"
                context += f"- **Hierarchy:** {doc.get('hierarchy', 'N/A')}\n"
                context += f"- **Content:** {doc.get('content', 'N/A')}\n"
                if doc_url:
                    context += f"- **URL:** {doc_url}\n"
                context += "\n"
        
        system_prompt = f"""{base_system_prompt}

**Context from knowledge base:**
{context}

**User Question:** {message}

**Instructions:**
1. Use ONLY the provided context to answer.  
   - If the context is insufficient, politely say so.  
2. Be concise, accurate, and professional, while keeping a friendly tone.  
3. Reference specific documents by their **TITLE** (not numbers) and include **clickable markdown links**.  
   - Example: [How to Reduce Medicare Levy](https://help.lodgeit.net.au/support/solutions/articles/60000720420-how-to-reduce-medicare-levy)  
4. When images are relevant:  
   - Insert them **Like a article after the bullet/paragraph at the point they are most useful** in your explanation.  
   - Use standard markdown format: `![Alt text](Image_URL)`  
   - The image should appear **big and natural** like in an article when markdown is rendered (do not shrink or place in a separate section).  
5. All responses must be in **properly formatted markdown**.  
6. Highlight important terms or links in the response for readability.  
7. Do NOT invent documents, titles, or images—only use what exists in the provided context.

Must follow Instructions :
- The response must be in properly formatted markdown.
- With Proper Line breaks
- If bullets or numbering, proper format to be followed
- No special characters or symbols
- Proper Headings, and spacing after every Image and paragraph
- Must follow the proper format for the response
- In the markdown it should be properly spaced after every image and paragraph add multiple line breaks.

**Answer:**"""
        
        return system_prompt

    def _azure_openai_stream(self, messages: List[Dict[str, str]]):
        """Yield text chunks from Azure OpenAI Chat Completions stream in real time."""
        client = AzureOpenAI(
            api_key=CONFIG.AZURE_OPENAI_API_KEY,
            azure_endpoint=CONFIG.AZURE_OPEN_API_ENDPOINT,
            api_version=CONFIG.AZURE_OPENAI_API_VERSION,
        )
        # Use the deployment name configured in Azure OpenAI
        deployment = CONFIG.AZURE_OPENAI_DEPLOYMENT
        stream = client.chat.completions.create(
            model=deployment,
            messages=messages,
            stream=True,
            temperature=0,
        )
        for event in stream:
            try:
                choice = getattr(event, "choices", [])[0]
                # delta.content is the incremental text
                delta = getattr(choice, "delta", None)
                if delta and getattr(delta, "content", None):
                    yield delta.content
                else:
                    # sometimes full message is present
                    msg = getattr(choice, "message", None)
                    if msg and getattr(msg, "content", None):
                        yield msg.content
            except Exception:
                text = str(event)
                if text:
                    yield text

    async def chat_with_rag_streaming_azure(self, message: str, hierarchy_filters: List[str], index_name: str = None, limit: int = 3) -> AsyncGenerator[str, None]:
        """Streaming chat with RAG using Azure OpenAI real-time streaming with automatic index classification."""
        try:
            # 1. Classify query to determine the best index
            classified_index = self._classify_and_get_index(message, index_name)
            
            # 2. Perform RAG search using the classified index
            relevant_docs = self.azure_search.semantic_search_documents(
                keywords=message,
                class_filters=hierarchy_filters,
                index_name=classified_index,
                limit=limit
            )

            # 3. Create RAG-enhanced prompt with the classified index
            system_prompt = self._create_rag_prompt(message, relevant_docs, classified_index)

            # 4. Stream LLM response from Azure OpenAI
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]
            for chunk in self._azure_openai_stream(messages):
                if chunk:
                    yield chunk
        except Exception as e:
            yield f"**Error:** {str(e)}"
    
    def _call_gemini_api(self, messages: List[Dict[str, str]]) -> str:
        """Call Gemini API and return the response"""
        try:
            url = f"{self.gemini_base_url}/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
            
            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    # Gemini doesn't have system role, prepend to user message
                    continue
                elif msg["role"] == "user":
                    content = msg["content"]
                    # If there was a system message before, prepend it
                    if messages[0]["role"] == "system":
                        content = messages[0]["content"] + "\n\n" + content
                    contents.append({
                        "parts": [{"text": content}]
                    })
            
            payload = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": 3500,
                    "temperature": 0
                }
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean the response to remove invalid control characters
            cleaned_response = self._clean_text_response(text_response)
            return cleaned_response
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _clean_text_response(self, text: str) -> str:
        """Clean text response to remove invalid control characters"""
        if not text:
            return text
        
        # Remove or replace invalid control characters
        import re
        # Keep only printable characters, newlines, tabs, and carriage returns
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        cleaned = re.sub(r'\r\n', '\n', cleaned)  # Normalize line endings
        cleaned = re.sub(r'\r', '\n', cleaned)    # Convert remaining \r to \n
        
        return cleaned
    
    def _call_taxgenii_search_api(self, message: str) -> Dict[str, Any]:
        """Call Taxgenii search API to get search data"""
        try:
            url = "https://api.taxgenii.lodgeit.net.au/api/chat/get-search-data"
            
            payload = {
                "username": "user",
                "prompt": message,
                "learn": False,
                "dataSource": "azureS",
                "fastSearch": None,
                "helpGuides": True,
                "legalInfo": False,
                "model": "gpt-4o",
                "toolTip": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            raise Exception(f"Taxgenii search API error: {str(e)}")
    
    def _call_taxgenii_response_api(self, message: str) -> tuple[str, list]:
        """Call Taxgenii response API to get the final response and reference documents"""
        try:
            url = "https://api.taxgenii.lodgeit.net.au/api/chat/get-response-message"
            
            payload = {
                "username": "user",
                "prompt": message,
                "learn": False,
                "stream": True
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            # Extract reference documents from x-metainfo header
            reference_docs = []
            x_metainfo = response.headers.get('x-metainfo')
            if x_metainfo:
                try:
                    metainfo = json.loads(x_metainfo)
                    if 'urls' in metainfo:
                        for url_info in metainfo['urls']:
                            reference_docs.append({
                                "title": url_info.get('hierrachy', ''),  # Note: API has typo "hierrachy"
                                "url": url_info.get('url', ''),
                                "hierarchy": url_info.get('hierrachy', '')
                            })
                except json.JSONDecodeError:
                    print(f"Failed to parse x-metainfo header: {x_metainfo}")
            
            # Check if response is JSON or plain text
            content_type = response.headers.get('content-type', '')
            
            if 'application/json' in content_type:
                try:
                    result = response.json()
                    # Extract the response text from the API response
                    if isinstance(result, dict) and "response" in result:
                        return result["response"], reference_docs
                    elif isinstance(result, str):
                        return result, reference_docs
                    else:
                        return str(result), reference_docs
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw text
                    return response.text, reference_docs
            else:
                # If not JSON, return the raw text
                return response.text, reference_docs
                
        except Exception as e:
            raise Exception(f"Taxgenii response API error: {str(e)}")
    
    def _get_taxgenii_response(self, message: str) -> tuple[str, list]:
        """Get response from Taxgenii APIs for ato_complete_data2 index"""
        try:
            # Skip search API due to server error and go directly to response API
            # The response API works correctly and provides the needed information
            response, reference_docs = self._call_taxgenii_response_api(message)
            
            return response, reference_docs
            
        except Exception as e:
            return f"Error getting Taxgenii response: {str(e)}", []
    
    async def chat_with_rag(self, message: str, hierarchy_filters: List[str], index_name: str = None, limit: int = 3) -> Dict[str, Any]:
        """Non-streaming chat with RAG using automatic index classification"""
        try:
            # 1. Classify query to determine the best index
            classified_index = self._classify_and_get_index(message, index_name)
            
            # 2. Handle different indexes
            if classified_index == "ato_complete_data2":
                # Use Taxgenii APIs for ATO data
                llm_response, relevant_docs = self._get_taxgenii_response(message)
            elif classified_index == "logit-website":
                # Use website search for website content
                chunks = self.azure_search.search_website_chunks(message, top=5)
                parent_id = chunks[0].get("parent_id") if chunks else None
                edges = self.azure_search.fetch_website_edges(parent_id, top=20) if parent_id else []
                context = self.azure_search.build_website_context_markdown(chunks, edges, question=message)
                
                # Create system prompt for website content
                system_prompt = self._create_rag_prompt(message, [], classified_index)
                system_prompt = system_prompt.replace("**Context from knowledge base:**\n\n", f"**Context from knowledge base:**\n{context}\n")
                
                # Get LLM response
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ]
                llm_response = self._call_gemini_api(messages)
                relevant_docs = []  # Website search doesn't return standard document format
            else:
                # Use Azure Search for other indexes
                relevant_docs = self.azure_search.semantic_search_documents(
                    keywords=message,
                    class_filters=hierarchy_filters,
                    index_name=classified_index,
                    limit=limit
                )
                
                # 3. Create RAG-enhanced prompt with the classified index
                system_prompt = self._create_rag_prompt(message, relevant_docs, classified_index)
                
                # 4. Get LLM response using Gemini API
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
                
                llm_response = self._call_gemini_api(messages)
            
            return {
                "response": llm_response,
                "relevant_documents": relevant_docs,
                "query": message,
                "classified_index": classified_index
            }
            
        except Exception as e:
            raise Exception(f"Chat service error: {str(e)}")
    
    async def chat_with_rag_streaming(self, message: str, hierarchy_filters: List[str], index_name: str = None, limit: int = 3) -> AsyncGenerator[str, None]:
        """Streaming chat with RAG using automatic index classification"""
        try:
            # 1. Classify query to determine the best index
            classified_index = self._classify_and_get_index(message, index_name)
            
            # 2. Handle different indexes
            if classified_index == "ato_complete_data2":
                # Use Taxgenii APIs for ATO data
                llm_response, _ = self._get_taxgenii_response(message)
            elif classified_index == "logit-website":
                # Use website search for website content
                chunks = self.azure_search.search_website_chunks(message, top=5)
                parent_id = chunks[0].get("parent_id") if chunks else None
                edges = self.azure_search.fetch_website_edges(parent_id, top=20) if parent_id else []
                context = self.azure_search.build_website_context_markdown(chunks, edges, question=message)
                
                # Create system prompt for website content
                system_prompt = self._create_rag_prompt(message, [], classified_index)
                system_prompt = system_prompt.replace("**Context from knowledge base:**\n\n", f"**Context from knowledge base:**\n{context}\n")
                
                # Get LLM response
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ]
                llm_response = self._call_gemini_api(messages)
            else:
                # Use Azure Search for other indexes
                relevant_docs = self.azure_search.semantic_search_documents(
                    keywords=message,
                    class_filters=hierarchy_filters,
                    index_name=classified_index,
                    limit=limit
                )
                
                # 3. Create RAG-enhanced prompt with the classified index
                system_prompt = self._create_rag_prompt(message, relevant_docs, classified_index)
                
                # 4. Get LLM response using Gemini API
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
                
                llm_response = self._call_gemini_api(messages)
            
            # Yield the response in chunks to simulate streaming
            words = llm_response.split()
            for word in words:
                yield word + " "
                
        except Exception as e:
            yield f"**Error:** {str(e)}"
    
