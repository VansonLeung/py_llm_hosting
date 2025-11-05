"""Test embeddings endpoint with self-hosted model."""

import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000/v1"

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_single_embedding():
    """Test single text embedding."""
    print("\n" + "="*60)
    print("Test 1: Single Text Embedding")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json={
            "model": "bge-micro-v2",
            "input": "Hello, how are you?"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        embedding = data['data'][0]['embedding']
        print(f"✓ Successfully generated embedding")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Model: {data.get('model', 'N/A')}")
        return embedding
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return None

def test_batch_embeddings():
    """Test batch text embeddings."""
    print("\n" + "="*60)
    print("Test 2: Batch Embeddings")
    print("="*60)
    
    texts = [
        "The cat sits on the mat",
        "A feline rests on a rug",
        "The dog runs in the park"
    ]
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json={
            "model": "bge-micro-v2",
            "input": texts
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Successfully generated {len(data['data'])} embeddings")
        
        embeddings = [item['embedding'] for item in data['data']]
        
        # Calculate similarities
        print("\nSimilarity matrix:")
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if j >= i:
                    sim = cosine_similarity(
                        np.array(embeddings[i]), 
                        np.array(embeddings[j])
                    )
                    print(f"  Text {i+1} vs Text {j+1}: {sim:.4f}")
        
        return embeddings
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return None

def test_semantic_similarity():
    """Test semantic similarity between related texts."""
    print("\n" + "="*60)
    print("Test 3: Semantic Similarity")
    print("="*60)
    
    pairs = [
        ("Python programming", "Coding in Python"),
        ("Machine learning", "Artificial intelligence"),
        ("Apple fruit", "Apple company")
    ]
    
    for text1, text2 in pairs:
        response1 = requests.post(
            f"{BASE_URL}/embeddings",
            json={"model": "bge-micro-v2", "input": text1}
        )
        response2 = requests.post(
            f"{BASE_URL}/embeddings",
            json={"model": "bge-micro-v2", "input": text2}
        )
        
        if response1.status_code == 200 and response2.status_code == 200:
            emb1 = np.array(response1.json()['data'][0]['embedding'])
            emb2 = np.array(response2.json()['data'][0]['embedding'])
            similarity = cosine_similarity(emb1, emb2)
            
            print(f"  '{text1}' <-> '{text2}'")
            print(f"  Similarity: {similarity:.4f}\n")
        else:
            print(f"  ✗ Error getting embeddings for pair")

def main():
    """Run all embedding tests."""
    print("="*60)
    print("EMBEDDINGS ENDPOINT TESTING")
    print("="*60)
    print("\nTesting with model: bge-micro-v2")
    print("Endpoint: /v1/embeddings")
    
    try:
        # Test 1: Single embedding
        test_single_embedding()
        
        # Test 2: Batch embeddings
        test_batch_embeddings()
        
        # Test 3: Semantic similarity
        test_semantic_similarity()
        
        print("\n" + "="*60)
        print("✓ ALL EMBEDDING TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
