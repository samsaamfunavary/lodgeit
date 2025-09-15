#!/usr/bin/env python3
"""
Test script for the classifier integration with RAG system.
This script tests the automatic index classification and RAG responses.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.classifier_service import ClassifierService
from app.services.chat_service import ChatService

async def test_classifier_integration():
    """Test the complete classifier integration"""
    
    print("🧪 Testing Classifier Integration with RAG System")
    print("=" * 60)
    
    # Initialize services
    try:
        classifier = ClassifierService()
        chat_service = ChatService()
        print("✅ Services initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return
    
    # Test queries for different indexes
    test_queries = [
        {
            "query": "How to Connect QuickBooks Online Accountant to LodgeiT",
            "expected_index": "lodgeit-help-guides",
            "description": "Help guide query"
        },
        {
            "query": "What is the pricing for LodgeiT Nano plan for accountants?",
            "expected_index": "pricing", 
            "description": "Pricing query"
        },
        {
            "query": "How do I set up RAM for a new employee in ATO systems?",
            "expected_index": "Taxgenii",
            "description": "Tax professional query"
        },
        {
            "query": "What features does LodgeiT have for bookkeepers?",
            "expected_index": "logit-website",
            "description": "Product information query"
        },
        {
            "query": "Tell me about amending legislation and compliance",
            "expected_index": "Taxgenii",
            "description": "Tax law query"
        }
    ]
    
    print("\n🔍 Testing Query Classification")
    print("-" * 40)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected_index"]
        description = test_case["description"]
        
        print(f"\n{i}. {description}")
        print(f"   Query: '{query}'")
        
        try:
            # Test classification
            classified_index = classifier.classify_query(query)
            print(f"   Expected: {expected}")
            print(f"   Classified: {classified_index}")
            
            if classified_index == expected:
                print("   ✅ Classification correct")
            else:
                print("   ⚠️  Classification different (may still be valid)")
                
        except Exception as e:
            print(f"   ❌ Classification failed: {e}")
    
    print("\n🤖 Testing RAG Integration")
    print("-" * 40)
    
    # Test a few queries with RAG
    rag_test_queries = [
        "How much does LodgeiT cost for small accounting firms?",
        "How do I connect to the ATO from LodgeiT?",
        "What is LodgeiT and who is it for?"
    ]
    
    for i, query in enumerate(rag_test_queries, 1):
        print(f"\n{i}. Testing RAG with query: '{query}'")
        
        try:
            # Test RAG response
            response = await chat_service.chat_with_rag(
                message=query,
                hierarchy_filters=[],
                index_name=None,  # Let it auto-classify
                limit=2
            )
            
            print(f"   Classified Index: {response.get('classified_index', 'Unknown')}")
            print(f"   Response Length: {len(response.get('response', ''))} characters")
            print(f"   Documents Found: {len(response.get('relevant_documents', []))}")
            
            # Show first 200 characters of response
            response_preview = response.get('response', '')[:200]
            print(f"   Response Preview: {response_preview}...")
            
        except Exception as e:
            print(f"   ❌ RAG test failed: {e}")
    
    print("\n🎯 Testing Index Mapping")
    print("-" * 40)
    
    # Test index mapping
    index_mapping = classifier.get_index_mapping()
    print("Index Mapping:")
    for classifier_output, azure_index in index_mapping.items():
        print(f"  {classifier_output} -> {azure_index}")
    
    print("\n✅ Classifier Integration Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_classifier_integration())


