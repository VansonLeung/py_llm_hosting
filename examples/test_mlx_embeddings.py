"""Test MLX embedding model."""

import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000/v1"

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_mlx_embedding_single():
    """Test single text embedding with MLX."""
    print("\n" + "="*60)
    print("Test: MLX Embedding (Single Text)")
    print("="*60)
    
    text = "Machine learning is a subset of artificial intelligence"
    
    print(f"Text: {text}")
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3-embed",
            "input": text
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        embedding = data["data"][0]["embedding"]
        print(f"\n✓ Generated embedding")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Usage: {data.get('usage', {})}")
        return embedding
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return None

def test_mlx_embedding_batch():
    """Test batch embedding with MLX."""
    print("\n" + "="*60)
    print("Test: MLX Embedding (Batch)")
    print("="*60)
    
    texts = [
        "Artificial intelligence and machine learning",
        "Deep learning uses neural networks",
        "The weather is sunny today"
    ]
    
    print(f"Processing {len(texts)} texts")
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3-embed",
            "input": texts
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        
        print(f"\n✓ Generated {len(embeddings)} embeddings")
        print(f"  Dimension: {len(embeddings[0])}")
        
        # Calculate similarities
        print("\n  Similarity Analysis:")
        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])
        sim_12 = cosine_similarity(embeddings[1], embeddings[2])
        
        print(f"    Text 0 vs Text 1 (AI/ML related): {sim_01:.4f}")
        print(f"    Text 0 vs Text 2 (AI vs Weather): {sim_02:.4f}")
        print(f"    Text 1 vs Text 2 (DL vs Weather): {sim_12:.4f}")
        
        if sim_01 > sim_02:
            print("    ✓ Related texts have higher similarity!")
        else:
            print("    ⚠ Warning: Expected higher similarity for related texts")
        
        return embeddings
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return None

def test_mlx_embedding_semantic_search():
    """Test semantic search use case with MLX embeddings."""
    print("\n" + "="*60)
    print("Test: MLX Semantic Search")
    print("="*60)
    
    query = "What is deep learning?"
    documents = [
        "Deep learning is a subset of machine learning using neural networks with multiple layers",
        "Python is a popular programming language for data science",
        "Neural networks are inspired by biological neurons in the brain",
        "The stock market closed higher today",
        "Machine learning algorithms can learn from data"
    ]
    
    print(f"Query: {query}")
    print(f"Documents: {len(documents)}")
    
    # Get query embedding
    query_response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3-embed",
            "input": query
        }
    )
    
    if query_response.status_code != 200:
        print(f"\n✗ Error getting query embedding: {query_response.status_code}")
        return
    
    query_embedding = query_response.json()["data"][0]["embedding"]
    
    # Get document embeddings
    docs_response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3-embed",
            "input": documents
        }
    )
    
    if docs_response.status_code != 200:
        print(f"\n✗ Error getting document embeddings: {docs_response.status_code}")
        return
    
    doc_embeddings = [item["embedding"] for item in docs_response.json()["data"]]
    
    # Calculate similarities and rank
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((i, sim, documents[i]))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\n✓ Semantic search results (ranked by relevance):")
    for rank, (idx, score, doc) in enumerate(similarities[:3], 1):
        print(f"\n  {rank}. Document {idx} (Score: {score:.4f})")
        print(f"     {doc[:80]}{'...' if len(doc) > 80 else ''}")

def test_mlx_embedding_normalization():
    """Test that MLX embeddings are normalized."""
    print("\n" + "="*60)
    print("Test: MLX Embedding Normalization")
    print("="*60)
    
    text = "Test normalization"
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3-embed",
            "input": text
        }
    )
    
    if response.status_code == 200:
        embedding = response.json()["data"][0]["embedding"]
        norm = np.linalg.norm(embedding)
        
        print(f"Text: {text}")
        print(f"Embedding norm: {norm:.6f}")
        
        if abs(norm - 1.0) < 0.01:
            print("✓ Embedding is normalized (norm ≈ 1.0)")
        else:
            print(f"⚠ Warning: Embedding norm is {norm}, expected ~1.0")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")

def main():
    """Run all MLX embedding tests."""
    print("="*60)
    print("MLX EMBEDDING TESTING")
    print("="*60)
    print("\nModel: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
    print("Backend: MLX (Apple Silicon)")
    print("Endpoint: /v1/embeddings")
    
    try:
        # Test 1: Single text
        test_mlx_embedding_single()
        
        # Test 2: Batch processing
        test_mlx_embedding_batch()
        
        # Test 3: Semantic search
        test_mlx_embedding_semantic_search()
        
        # Test 4: Normalization
        test_mlx_embedding_normalization()
        
        print("\n" + "="*60)
        print("MLX EMBEDDING TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
