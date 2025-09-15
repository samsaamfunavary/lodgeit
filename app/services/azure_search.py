from app.core.config import CONFIG
import os
import json
import re
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from typing import List, Dict, Any

from openai import OpenAI
client = OpenAI(api_key=CONFIG.OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

class Azure_Search:
    def __init__(self):
        self.api_key = CONFIG.AZURE_KEY
        self.api_endpoint = CONFIG.AZURE_ENDPOINT
        
    def search_documents(self, keywords, class_filters, index_name, limit=3):
        filter_conditions = ""
        for class_filter in class_filters:
            ltan = class_filter + "addition"
            filter_conditions += f"hierarchy ge '{class_filter}' and hierarchy le '{ltan}' or "
        if filter_conditions:
            filter_conditions = filter_conditions[:-4]  # remove last " or "
        
        client = SearchClient(self.api_endpoint, index_name, AzureKeyCredential(self.api_key))       
        search_results = client.search(search_text=keywords, top=limit, filter=filter_conditions)
        relevant_documents = []
        for result in search_results:
            document = {
                "num": result.get("num", ""),
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "hierarchy": result.get("hierarchy", ""),
                "content": result.get("content", "")
            }
            relevant_documents.append(document)
        
        return relevant_documents

    def semantic_search_documents(self, keywords, class_filters, index_name, limit=3, semantic_configuration_name="default"):
        try:
            # Build filter string
            filter_conditions = ""
            for class_filter in class_filters:
                ltan = class_filter + "addition"
                filter_conditions += f"hierarchy ge '{class_filter}' and hierarchy le '{ltan}' or "
            if filter_conditions:
                filter_conditions = filter_conditions[:-4]  # remove last " or "
            
            # Init search client
            client = SearchClient(self.api_endpoint, index_name, AzureKeyCredential(self.api_key))
            
            # Use the provided semantic configuration name
            query_options = {
                "query_type": "semantic",
                "semantic_configuration_name": semantic_configuration_name,
                "top": limit,
            }
            
            # Perform semantic search with the correct configuration
            if filter_conditions:
                search_results = client.search(
                    search_text=keywords,
                    filter=filter_conditions,
                    **query_options
                )
            else:
                search_results = client.search(
                    search_text=keywords,
                    **query_options
                )
            
            relevant_documents = []
            for result in search_results:
                document = {
                    "score": result.get("@search.score", 0),   # semantic relevance score
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "hierarchy": result.get("hierarchy", ""),
                    "content": result.get("content", "")
                }
                relevant_documents.append(document)
            
            return relevant_documents
            
        except Exception as e:
            print(f"Search failed: {e}")
            # Return empty results instead of crashing
            return []
            
    def _get_search_client(self, index_name: str) -> SearchClient:
        """Get a search client for the specified index"""
        return SearchClient(
            endpoint=self.api_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self.api_key),
        )
    
    # =========================
    # Pricing Search Methods
    # =========================
    def search_pricing_data(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for pricing information in Azure Search (pricing index has different fields)"""
        try:
            client = self._get_search_client("lodgeit-pricing")
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
    
    def format_pricing_results(self, search_results: List[Dict[str, Any]]) -> str:
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

                # Income Tax Returns
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

                # IITR, BAS and Other Returns
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

                # Business Reporting Forms
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

                # Financial Reports
                fr = plan.get('financial_reports') or {}
                if fr:
                    formatted_text += "**Financial Reports:**\n"
                    if 'description' in fr:
                        formatted_text += f"  {fr['description']}\n"
                    if 'cost' in fr:
                        formatted_text += f"  Cost: {fr['cost']}\n"
                    formatted_text += "\n"

                # Financial Reports Pro
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

                # Legal Documents
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

                # E-Signatures
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
    # Website Graph-RAG Methods
    # =========================
    def search_website_chunks(self, query: str, top: int = 10) -> List[Dict[str, Any]]:
        """Search website chunks for graph-RAG"""
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

    def fetch_website_edges(self, parent_id: str, top: int = 20) -> List[Dict[str, Any]]:
        """Fetch edges for website graph-RAG"""
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
        """Extract markdown links and images from text"""
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
        """Extract image descriptions from text"""
        if not text:
            return []
        descs = re.findall(r'_image_description_in_text:\s*(.+?)(?:\n\s*\n|$)', text, flags=re.DOTALL)
        return [re.sub(r"\s+", " ", d).strip() for d in descs]

    def _select_relevant_images(self, question: str, image_urls: List[str], descriptions: List[str]) -> List[Dict[str, str]]:
        """Select relevant images based on question"""
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

    def build_website_context_markdown(self, chunks: List[Dict[str, Any]], edges: List[Dict[str, Any]], question: str = "") -> str:
        """Build markdown context for website graph-RAG"""
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