"""
Comprehensive test suite for all endpoints.

Tests:
- Chat completions (streaming and non-streaming)
- Embeddings (single and batch)
- Status tracking
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/v1"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.YELLOW}ℹ {msg}{Colors.END}")

def test_chat_completion():
    """Test chat completion endpoint."""
    print_test("Chat Completion (Non-Streaming)")
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": "qwen2.5-0.5b",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 20,
            "temperature": 0.3
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        tokens = data['usage']['total_tokens']
        print_success(f"Response: {content}")
        print_info(f"Tokens used: {tokens}")
        return True
    else:
        print_error(f"Status {response.status_code}: {response.text}")
        return False

def test_chat_streaming():
    """Test streaming chat completion."""
    print_test("Chat Completion (Streaming)")
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": "qwen2.5-0.5b",
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "max_tokens": 30,
            "temperature": 0.5,
            "stream": True
        },
        stream=True
    )
    
    if response.status_code == 200:
        chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunk_data = json.loads(line[6:])
                    if chunk_data['choices'][0]['delta'].get('content'):
                        chunks.append(chunk_data['choices'][0]['delta']['content'])
        
        full_response = ''.join(chunks)
        print_success(f"Received {len(chunks)} chunks")
        print_info(f"Response: {full_response}")
        return True
    else:
        print_error(f"Status {response.status_code}")
        return False

def test_embeddings_single():
    """Test single text embedding."""
    print_test("Embeddings (Single Text)")
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        json={
            "model": "all-minilm",
            "input": "Machine learning is awesome"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        embedding = data['data'][0]['embedding']
        print_success(f"Generated embedding with dimension: {len(embedding)}")
        print_info(f"First 3 values: {embedding[:3]}")
        return True
    else:
        print_error(f"Status {response.status_code}: {response.text}")
        return False

def test_embeddings_batch():
    """Test batch embeddings."""
    print_test("Embeddings (Batch)")
    
    texts = [
        "Artificial intelligence",
        "Machine learning",
        "Deep neural networks"
    ]
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        json={
            "model": "all-minilm",
            "input": texts
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print_success(f"Generated {len(data['data'])} embeddings")
        
        # Calculate similarity between first two
        import numpy as np
        emb1 = np.array(data['data'][0]['embedding'])
        emb2 = np.array(data['data'][1]['embedding'])
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print_info(f"Similarity between '{texts[0]}' and '{texts[1]}': {similarity:.4f}")
        return True
    else:
        print_error(f"Status {response.status_code}: {response.text}")
        return False

def test_server_status():
    """Test if server status tracking works."""
    print_test("Server Status Tracking")
    
    # Make a request to load a model
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": "qwen2.5-0.5b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5
        }
    )
    
    if response.status_code == 200:
        print_success("Model loaded successfully")
        print_info("Server status should now be 'active'")
        print_info("Check with: python main.py list-servers")
        return True
    else:
        print_error(f"Failed to load model: {response.status_code}")
        return False

def main():
    """Run all tests."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("COMPREHENSIVE API ENDPOINT TESTING")
    print(f"{'='*60}{Colors.END}\n")
    
    print_info(f"Testing server at: {BASE_URL}")
    print_info(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Test chat endpoints
    results['Chat (Non-Streaming)'] = test_chat_completion()
    results['Chat (Streaming)'] = test_chat_streaming()
    
    # Test embeddings
    results['Embeddings (Single)'] = test_embeddings_single()
    results['Embeddings (Batch)'] = test_embeddings_batch()
    
    # Test status tracking
    results['Status Tracking'] = test_server_status()
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}{Colors.END}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{Colors.BLUE}Total: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{'='*60}")
        print("✓ ALL TESTS PASSED!")
        print(f"{'='*60}{Colors.END}\n")
    else:
        print(f"\n{Colors.YELLOW}{'='*60}")
        print(f"⚠ {total - passed} test(s) failed")
        print(f"{'='*60}{Colors.END}\n")

if __name__ == "__main__":
    main()
