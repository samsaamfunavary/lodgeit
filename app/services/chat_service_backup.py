from openai import AzureOpenAI
from app.services.azure_search import Azure_Search
from app.services.classifier_service import ClassifierService
from app.core.config import CONFIG
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import requests
import json
from typing import List, Dict, Any, AsyncGenerator
import re
import os

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
        if index_name == "lodgeit-help-guides":
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
""",

          
        }
            return system_prompts
        elif index_name == "lodgeit-pricing":
            system_prompts = {
            "pricing": """You are a LodgeiT Pricing assistant. Answer using ONLY the pricing context provided.

Formatting and behavior:
- Provide prices in AUD; mention GST where applicable.
- If comparing plans, provide a concise comparison and call out key differences.
- When a plan is asked about, include the plan name, price, included allowances, notable features, and overage/extra usage fees.
- If bundles or e-sign packages are relevant, list the available tiers and per-unit costs.
- If information is missing from the context, state that it is not available.

Must follow:
- Use clean markdown with sections, bullet lists, and tables when helpful.
- Do not include non-pricing topics; redirect such questions to the appropriate resource.
"""
        }
            return system_prompts
        elif index_name == "ato_complete_data2":
            system_prompts = {
            "taxgenii": """You are a Taxgenii assistant for ATO operational guidance. Answer using ONLY the provided ATO/practice context.

Formatting and behavior:
- Focus on ATO portals, agent workflows, lodgment programs, client-to-agent linking, deferrals, POI, RAM/myGovID, and compliance.
- When steps are relevant, provide clear, ordered step-by-step instructions.
- Reference named ATO areas or programs as they appear in the context. Include links when URLs are present.
- If the question is outside operational guidance (e.g., tax law interpretations), state it is out of scope and suggest where to look.

Must follow:
- Professional, succinct markdown with headings and lists.
- No speculation; do not provide financial or legal advice.
"""
        }
            return system_prompts
        elif index_name == "website":
            system_prompts = {
            "website": """You are a LodgeiT Product & Website assistant. Answer using ONLY the provided product/features/resources context.

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
            return system_prompts
        
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
                pricing_results = self._search_pricing_data(message, max_results=5)
                context = self._format_pricing_results(pricing_results)
            except Exception:
                context = ""
        elif index_name == "logit-website":
            try:
                chunks = self._search_website_chunks(message, top=5)
                parent_id = chunks[0].get("parent_id") if chunks else None
                edges = self._fetch_website_edges(parent_id, top=20) if parent_id else []
                context = self._build_website_context_markdown(chunks, edges, question=message)
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

    # =========================
    # Pricing (custom schema) helpers
    # =========================

    def _search_pricing_data(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for pricing information in Azure Search (pricing index has different fields).
        Expects fields: id, tab_name, hierarchy, plan (JSON string with nested structure).
        """
        try:
            client = SearchClient(self.azure_search.api_endpoint, "pricing", AzureKeyCredential(self.azure_search.api_key))
            results = client.search(
                search_text=query,
                select=["id", "tab_name", "hierarchy", "plan"],
                top=max_results
            )

            search_results: List[Dict[str, Any]] = []
            for result in results:
                plan_raw = result.get('plan', '{}')
                try:
                    plan_data = json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
                except Exception:
                    plan_data = {}

                doc_info: Dict[str, Any] = {
                    "tab_name": result.get('tab_name', ''),
                    "hierarchy": result.get('hierarchy', ''),
                    "plans": []
                }

                if isinstance(plan_data, dict):
                    for category, category_plans in plan_data.items():
                        if isinstance(category_plans, dict):
                            for plan_name, plan_details in category_plans.items():
                                if isinstance(plan_details, dict) and 'title' in plan_details:
                                    plan_info = {
                                        "category": category,
                                        "plan_name": plan_details.get('title', plan_name),
                                        "price": plan_details.get('price', ''),
                                        "lodgments": plan_details.get('lodgments', ''),
                                        "users": plan_details.get('users', ''),
                                        "description": plan_details.get('description', ''),
                                        "features": plan_details.get('features', []),
                                        "income_tax_returns": plan_details.get('incomeTaxReturns', {}),
                                        "iitr_bas_returns": plan_details.get('iitrBasAndOthersReturns', {}),
                                        "business_reporting_forms": plan_details.get('businessReportingForms', {}),
                                        "financial_reports": plan_details.get('financialReports', {}),
                                        "financial_reports_pro": plan_details.get('financialReportsPro', {}),
                                        "legal_documents": plan_details.get('legalDocuments', {}),
                                        "e_signatures": plan_details.get('eSignatures', {}),
                                        "features_comparison": plan_data.get('featuresComparison', [])
                                    }
                                    doc_info["plans"].append(plan_info)

                search_results.append(doc_info)

            return search_results
        except Exception as e:
            print(f"Pricing search error: {e}")
            return []

    def _format_pricing_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format pricing search results for inclusion in the prompt context."""
        if not search_results:
            return "No pricing information found."

        formatted_text = "## Pricing Information Found:\n\n"
        for doc in search_results:
            formatted_text += f"### {doc.get('tab_name', '')}\n"
            formatted_text += f"**Category:** {doc.get('hierarchy', '')}\n\n"

            for plan in doc.get('plans', []):
                formatted_text += f"**Plan:** {plan.get('plan_name', '')}\n"
                formatted_text += f"**Price:** {plan.get('price', '')}\n"
                if plan.get('lodgments'):
                    formatted_text += f"**Lodgments:** {plan.get('lodgments')}\n"
                formatted_text += f"**Users:** {plan.get('users', '')}\n"
                if plan.get('description'):
                    formatted_text += f"**Description:** {plan.get('description')}\n"
                if plan.get('features'):
                    formatted_text += f"**Features:**\n"
                    for feature in plan.get('features', []):
                        formatted_text += f"  - {feature}\n"
                formatted_text += "\n"

                itr = plan.get('income_tax_returns') or {}
                if itr:
                    formatted_text += "**Income Tax Returns:**\n"
                    if 'details' in itr:
                        for detail in itr['details']:
                            formatted_text += f"  {detail}\n"
                    if 'cost' in itr:
                        if isinstance(itr['cost'], list):
                            for cost_item in itr['cost']:
                                formatted_text += f"  Cost: {cost_item}\n"
                        else:
                            formatted_text += f"  Cost: {itr['cost']}\n"
                    if 'packagePrices' in itr:
                        formatted_text += "  Package Prices:\n"
                        for pkg in itr['packagePrices']:
                            formatted_text += f"    - {pkg}\n"
                    formatted_text += "\n"

                iitr = plan.get('iitr_bas_returns') or {}
                if iitr:
                    formatted_text += "**IITR, BAS and Other Returns:**\n"
                    if 'details' in iitr:
                        for detail in iitr['details']:
                            formatted_text += f"  {detail}\n"
                    if 'cost' in iitr:
                        if isinstance(iitr['cost'], list):
                            for cost_item in iitr['cost']:
                                formatted_text += f"  Cost: {cost_item}\n"
                        else:
                            formatted_text += f"  Cost: {iitr['cost']}\n"
                    if 'packagePrices' in iitr:
                        formatted_text += "  Package Prices:\n"
                        for pkg in iitr['packagePrices']:
                            formatted_text += f"    - {pkg}\n"
                    formatted_text += "\n"

                brf = plan.get('business_reporting_forms') or {}
                if brf:
                    formatted_text += "**Business Reporting Forms:**\n"
                    if 'details' in brf:
                        for detail in brf['details']:
                            formatted_text += f"  {detail}\n"
                    if 'cost' in brf:
                        if isinstance(brf['cost'], list):
                            for cost_item in brf['cost']:
                                formatted_text += f"  Cost: {cost_item}\n"
                        else:
                            formatted_text += f"  Cost: {brf['cost']}\n"
                    if 'packagePrices' in brf:
                        formatted_text += "  Package Prices:\n"
                        for pkg in brf['packagePrices']:
                            formatted_text += f"    - {pkg}\n"
                    formatted_text += "\n"

                fr = plan.get('financial_reports') or {}
                if fr:
                    formatted_text += "**Financial Reports:**\n"
                    if 'description' in fr:
                        formatted_text += f"  {fr['description']}\n"
                    if 'cost' in fr:
                        formatted_text += f"  Cost: {fr['cost']}\n"
                    formatted_text += "\n"

                frp = plan.get('financial_reports_pro') or {}
                if frp:
                    formatted_text += "**Financial Reports Pro:**\n"
                    if 'cost' in frp:
                        if isinstance(frp['cost'], list):
                            for cost_item in frp['cost']:
                                formatted_text += f"  Cost: {cost_item}\n"
                        else:
                            formatted_text += f"  Cost: {frp['cost']}\n"
                    if 'packagePrices' in frp:
                        formatted_text += "  Package Prices:\n"
                        for pkg in frp['packagePrices']:
                            formatted_text += f"    - {pkg}\n"
                    formatted_text += "\n"

                ld = plan.get('legal_documents') or {}
                if ld:
                    formatted_text += "**Legal Documents:**\n"
                    if 'description' in ld:
                        formatted_text += f"  {ld['description']}\n"
                    if 'cost' in ld:
                        if isinstance(ld['cost'], list):
                            for cost_item in ld['cost']:
                                formatted_text += f"  Cost: {cost_item}\n"
                        else:
                            formatted_text += f"  Cost: {ld['cost']}\n"
                    if 'packagePrices' in ld:
                        formatted_text += "  Package Prices:\n"
                        for pkg in ld['packagePrices']:
                            formatted_text += f"    - {pkg}\n"
                    formatted_text += "\n"

                es = plan.get('e_signatures') or {}
                if es:
                    formatted_text += "**E-Signatures:**\n"
                    if 'description' in es:
                        formatted_text += f"  {es['description']}\n"
                    if 'cost' in es:
                        formatted_text += f"  Cost: {es['cost']}\n"
                    if 'packagePrices' in es:
                        formatted_text += "  Package Prices:\n"
                        for pkg in es['packagePrices']:
                            formatted_text += f"    - {pkg}\n"
                    formatted_text += "\n"

                formatted_text += "---\n\n"

        return formatted_text

    # =========================
    # Website graph-RAG helpers (chunks + edges)
    # =========================
    def _get_search_client(self, index_name: str) -> "SearchClient":
        return SearchClient(
            endpoint=self.azure_search.api_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self.azure_search.api_key),
        )

    def _search_website_chunks(self, query: str, top: int = 10) -> List[Dict[str, Any]]:
        chunk_index = os.getenv("CHUNK_INDEX_NAME", "lodgeit-chunks")
        client = self._get_search_client(chunk_index)
        results = client.search(
            search_text=query or "*",
            top=top,
            select=[
                "id",
                "parent_id",
                "title",
                "url",
                "hierarchy",
                "content",
                "chunk_index",
                "images",
            ],
        )
        return [r for r in results]

    def _fetch_website_edges(self, parent_id: str, top: int = 20) -> List[Dict[str, Any]]:
        if not parent_id:
            return []
        edge_index = os.getenv("EDGE_INDEX_NAME", "lodgeit-edges")
        client = self._get_search_client(edge_index)
        results = client.search(
            search_text="*",
            filter=f"parent_id eq '{parent_id}'",
            top=top,
            select=[
                "relation_type",
                "source_label",
                "target_label",
                "sentence",
                "confidence",
            ],
        )
        return [r for r in results]

    def _extract_markdown_assets(self, text: str) -> Dict[str, List[str]]:
        links: List[str] = []
        images_md: List[str] = []
        image_links: List[str] = []
        if not text:
            return {"links": links, "images_md": images_md, "image_links": image_links}
        for alt, url in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text):
            images_md.append(f"![{alt}]({url})")
            image_links.append(url)
        for label, url in re.findall(r'(?<!\!)\[([^\]]+)\]\(([^)]+)\)', text):
            links.append(f"[{label}]({url})")
        return {"links": links, "images_md": images_md, "image_links": image_links}

    def _extract_image_descriptions(self, text: str) -> List[str]:
        if not text:
            return []
        descs = re.findall(r'_image_description_in_text:\s*(.+?)(?:\n\s*\n|$)', text, flags=re.DOTALL)
        return [re.sub(r"\s+", " ", d).strip() for d in descs]

    def _select_relevant_images(self, question: str, image_urls: List[str], descriptions: List[str]) -> List[Dict[str, str]]:
        pairs: List[Dict[str, str]] = []
        for idx, url in enumerate(image_urls):
            desc = descriptions[idx] if idx < len(descriptions) else ""
            pairs.append({"url": url, "description": desc})
        if not pairs:
            return []
        if not question:
            return pairs[:6]
        qwords = {w.lower() for w in question.split() if len(w) > 3}
        ranked: List[tuple[int, Dict[str, str]]] = []
        for p in pairs:
            d = p.get("description", "").lower()
            score = sum(1 for w in qwords if w in d)
            ranked.append((score, p))
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = [p for s, p in ranked if s > 0][:6]
        if not selected:
            selected = [p for _, p in ranked[:3]]
        return selected

    def _build_website_context_markdown(self, chunks: List[Dict[str, Any]], edges: List[Dict[str, Any]], question: str = "") -> str:
        md: List[str] = []
        if chunks:
            md.append("## Retrieved Chunks\n")
            for i, ch in enumerate(chunks, start=1):
                title = ch.get("title", "")
                url = ch.get("url", "")
                hierarchy = ch.get("hierarchy", "")
                content = (ch.get("content", "") or "")[:800]
                full_content = ch.get("content", "") or ""
                assets = self._extract_markdown_assets(full_content)
                md_links = assets.get("links", [])
                md_images = assets.get("images_md", [])
                image_links = assets.get("image_links", [])
                image_urls_field = ch.get("images") or []
                merged_urls: List[str] = []
                seen = set()
                for u in image_links + image_urls_field:
                    if u and u not in seen:
                        seen.add(u)
                        merged_urls.append(u)
                image_descs = self._extract_image_descriptions(full_content)
                relevant = self._select_relevant_images(question, merged_urls, image_descs)
                md.append(f"### Chunk {i}: {title}\n")
                if hierarchy:
                    md.append(f"- Hierarchy: {hierarchy}\n")
                if url:
                    md.append(f"- URL: {url}\n")
                md.append("\n")
                md.append(content)
                md.append("\n\n---\n")
                if md_links:
                    md.append("**Links found in this chunk:**\n")
                    for link in md_links[:10]:
                        md.append(f"- {link}\n")
                if md_images:
                    md.append("\n**Images (from markdown in content):**\n")
                    for img in md_images[:6]:
                        md.append(f"{img}\n")
                if relevant:
                    md.append("\n**Relevant images (matched to question by description):**\n")
                    for r in relevant:
                        u = r.get("url", "")
                        d = r.get("description", "")
                        if u:
                            md.append(f"![related]({u})\n")
                        if d:
                            md.append(f"> {d}\n")
                md.append("\n\n")

        if edges:
            md.append("\n## Retrieved Relations\n")
            for e in edges[:20]:
                rel = e.get("relation_type", "RELATED_TO")
                s = e.get("source_label", "?")
                t = e.get("target_label", "?")
                sent = e.get("sentence", "")
                conf = e.get("confidence", 0)
                try:
                    conf_val = float(conf)
                except Exception:
                    conf_val = 0.0
                md.append(f"- {s} --[{rel}]--> {t} (conf {conf_val:.2f})\n")
                if sent:
                    md.append(f"  - Evidence: {sent}\n")

        return "\n".join(md).strip()

    def _collect_relevant_image_urls(self, chunks: List[Dict[str, Any]], question: str) -> List[str]:
        urls: List[str] = []
        for ch in chunks:
            full_content = ch.get("content", "") or ""
            assets = self._extract_markdown_assets(full_content)
            image_links = assets.get("image_links", [])
            image_urls_field = ch.get("images") or []
            merged: List[str] = []
            seen = set()
            for u in image_links + image_urls_field:
                if u and u not in seen:
                    seen.add(u)
                    merged.append(u)
            descs = self._extract_image_descriptions(full_content)
            relevant = self._select_relevant_images(question, merged, descs)
            urls.extend([r.get("url", "") for r in relevant if r.get("url")])
        dedup: List[str] = []
        seen2 = set()
        for u in urls:
            if u not in seen2:
                seen2.add(u)
                dedup.append(u)
        return dedup[:10]

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
            return result["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def chat_with_rag(self, message: str, hierarchy_filters: List[str], index_name: str = None, limit: int = 3) -> Dict[str, Any]:
        """Non-streaming chat with RAG using automatic index classification"""
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
            
            # 2. Perform RAG search using the classified index
            relevant_docs = self.azure_search.semantic_search_documents(
                keywords=message,
                class_filters=hierarchy_filters,
                index_name=classified_index,
                limit=limit
            )
            
            # 3. Create RAG-enhanced prompt with the classified index
            system_prompt = self._create_rag_prompt(message, relevant_docs, classified_index)
            
            # 4. Get LLM response using Gemini API (non-streaming for now)
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
