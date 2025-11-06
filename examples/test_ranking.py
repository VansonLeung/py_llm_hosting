"""Test ranking/rerank endpoint."""

import requests
import json
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test ranking endpoint')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)), 
                   help='Port number for the server (default: from PORT env var or 8000)')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}/v1"

def test_rerank():
    """Test document reranking."""
    print("\n" + "="*60)
    print("Test: Document Reranking")
    print("="*60)
    
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "The weather today is sunny and warm.",
        "Deep learning uses neural networks with multiple layers.",
        "I like to eat pizza for dinner.",
        "Supervised learning requires labeled training data."
    ]
    
    print(f"Query: {query}")
    print(f"Documents to rank: {len(documents)}")
    
    response = requests.post(
        f"{BASE_URL}/rerank",
        headers={"Content-Type": "application/json"},
        json={
            "model": "bge-reranker-v2-m3",
            "query": query,
            "documents": documents,
            "top_n": 3
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Successfully reranked documents")
        print(f"\nTop {len(data.get('results', []))} results:")
        
        for i, result in enumerate(data.get('results', []), 1):
            doc_idx = result.get('index', 0)
            score = result.get('relevance_score', 0)
            print(f"\n  {i}. Document {doc_idx} (Score: {score:.4f})")
            print(f"     {documents[doc_idx][:80]}...")
            
        return data
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"  {response.text}")
        
        if response.status_code == 400:
            print("\n  Note: Ranking endpoint requires a proxy server configuration.")
            print("  Add a rerank model server with:")
            print("  python main.py add-server \\")
            print("    --name 'Rerank Server' \\")
            print("    --model 'rerank-model' \\")
            print("    --endpoint 'http://your-rerank-server/v1/rerank'")
        
        return None

def test_rerank_different_queries():
    """Test reranking with different queries."""
    print("\n" + "="*60)
    print("Test: Multiple Query Reranking")
    print("="*60)
    
    test_cases = [
        {
            "query": "best programming languages",
            "docs": [
                "Python is great for data science and machine learning",
                "JavaScript is the language of the web",
                "C++ offers high performance for system programming",
                "The restaurant serves delicious Italian food"
            ]
        },
        {
            "query": "healthy eating tips",
            "docs": [
                "Exercise regularly for 30 minutes a day",
                "Eat plenty of fruits and vegetables",
                "Stay hydrated by drinking water",
                "Python is a programming language"
            ]
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test['query']}")
        
        response = requests.post(
            f"{BASE_URL}/rerank",
            json={
                "model": "bge-reranker-v2-m3",
                "query": test['query'],
                "documents": test['docs'],
                "top_n": 2
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Top 2 relevant documents:")
            for result in data.get('results', [])[:2]:
                doc_idx = result.get('index', 0)
                score = result.get('relevance_score', 0)
                print(f"    - Doc {doc_idx}: {test['docs'][doc_idx][:60]}... (Score: {score:.3f})")
        else:
            print(f"  ✗ Error: {response.status_code}")
            break

def main():
    """Run all ranking tests."""
    print("="*60)
    print("RANKING/RERANK ENDPOINT TESTING")
    print("="*60)
    print("\nEndpoint: /v1/rerank")
    
    try:
        # Test 1: Basic reranking
        test_rerank()
        
        # Test 2: Multiple queries (only if first test succeeded)
        test_rerank_different_queries()
        
        print("\n" + "="*60)
        print("RANKING TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
