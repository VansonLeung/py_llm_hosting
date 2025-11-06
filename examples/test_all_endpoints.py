"""
Comprehensive test suite for all endpoints.

Tests all available backends:
- llama-cpp: Chat (streaming/non-streaming), Embeddings
- MLX: Chat (streaming/non-streaming), Embeddings  
- sentence-transformers: Embeddings
- reranker: Document reranking
"""

import requests
import json
import time
import sys
import subprocess
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test all LLM endpoints')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)), 
                   help='Port number for the server (default: from PORT env var or 8000)')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}/v1"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    END = '\033[0m'

def print_section(name):
    print(f"\n{Colors.CYAN}{'='*70}")
    print(f"{name.center(70)}")
    print(f"{'='*70}{Colors.END}")

def print_test(name):
    print(f"\n{Colors.BLUE}{'─'*60}")
    print(f"TEST: {name}")
    print(f"{'─'*60}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.YELLOW}ℹ {msg}{Colors.END}")

def print_model(msg):
    print(f"{Colors.MAGENTA}🔧 {msg}{Colors.END}")

def print_model(msg):
    print(f"{Colors.MAGENTA}🔧 {msg}{Colors.END}")


# ============================================================================
# CHAT COMPLETION TESTS
# ============================================================================

def test_llama_chat(streaming=False):
    """Test llama-cpp chat completion."""
    mode = "Streaming" if streaming else "Non-Streaming"
    print_test(f"llama-cpp Chat ({mode})")
    print_model("Model: tinyllama (llama-cpp backend)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "tinyllama",
                "messages": [{"role": "user", "content": "What is 2+2? Answer in one short sentence."}],
                "max_tokens": 30,
                "temperature": 0.3,
                "stream": streaming
            },
            stream=streaming
        )
        
        if response.status_code == 200:
            if streaming:
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
            else:
                data = response.json()
                content = data['choices'][0]['message']['content']
                tokens = data['usage']['total_tokens']
                print_success(f"Response: {content}")
                print_info(f"Tokens used: {tokens}")
            return True
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception: {e}")
        return False


def test_mlx_chat(streaming=False):
    """Test MLX chat completion."""
    mode = "Streaming" if streaming else "Non-Streaming"
    print_test(f"MLX Chat ({mode})")
    print_model("Model: qwen2.5-0.5b (MLX backend)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "qwen2.5-0.5b",
                "messages": [{"role": "user", "content": "Name one programming language. Just the name."}],
                "max_tokens": 20,
                "temperature": 0.5,
                "stream": streaming
            },
            stream=streaming
        )
        
        if response.status_code == 200:
            if streaming:
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
            else:
                data = response.json()
                content = data['choices'][0]['message']['content']
                tokens = data['usage']['total_tokens']
                print_success(f"Response: {content}")
                print_info(f"Tokens used: {tokens}")
            return True
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception: {e}")
        return False


# ============================================================================
# EMBEDDINGS TESTS
# ============================================================================

def test_llama_embeddings():
    """Test llama-cpp embeddings."""
    print_test("llama-cpp Embeddings")
    print_model("Model: nomic-embed-v1.5 (llama-cpp backend)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/embeddings",
            json={
                "model": "nomic-embed-v1.5",
                "input": "Machine learning is fascinating"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding = data['data'][0]['embedding']
            print_success(f"Generated embedding with dimension: {len(embedding)}")
            print_info(f"First 5 values: {[f'{x:.4f}' for x in embedding[:5]]}")
            return True
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception: {e}")
        return False


def test_mlx_embeddings():
    """Test MLX embeddings."""
    print_test("MLX Embeddings")
    print_model("Model: qwen3-embed (MLX backend)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/embeddings",
            json={
                "model": "qwen3-embed",
                "input": "Artificial intelligence is transforming the world"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding = data['data'][0]['embedding']
            print_success(f"Generated embedding with dimension: {len(embedding)}")
            print_info(f"First 5 values: {[f'{x:.4f}' for x in embedding[:5]]}")
            return True
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception: {e}")
        return False


def test_sentence_transformers_embeddings():
    """Test sentence-transformers embeddings."""
    print_test("Sentence-Transformers Embeddings")
    print_model("Model: bge-micro-v2 (sentence-transformers backend)")
    
    try:
        # Test batch embeddings and similarity
        texts = [
            "Python programming",
            "Coding in Python",
            "Quantum physics"
        ]
        
        response = requests.post(
            f"{BASE_URL}/embeddings",
            json={
                "model": "bge-micro-v2",
                "input": texts
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Generated {len(data['data'])} embeddings")
            
            # Calculate similarity between first two (should be high)
            import numpy as np
            emb1 = np.array(data['data'][0]['embedding'])
            emb2 = np.array(data['data'][1]['embedding'])
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            print_info(f"Dimension: {len(emb1)}")
            print_info(f"Similarity between related texts: {similarity:.4f}")
            return True
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception: {e}")
        return False


# ============================================================================
# RERANKING TESTS
# ============================================================================

def test_reranker():
    """Test document reranking."""
    print_test("Document Reranking")
    print_model("Model: bge-reranker-v2-m3 (reranker backend)")
    
    try:
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of AI focused on algorithms.",
            "The weather today is sunny and warm.",
            "Deep learning uses neural networks with multiple layers.",
            "I like to eat pizza for dinner."
        ]
        
        response = requests.post(
            f"{BASE_URL}/rerank",
            json={
                "model": "bge-reranker-v2-m3",
                "query": query,
                "documents": documents,
                "top_n": 3
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Reranked {len(documents)} documents")
            print_info(f"Top 3 results:")
            for i, result in enumerate(data.get('results', []), 1):
                doc_idx = result.get('index', 0)
                score = result.get('relevance_score', 0)
                print(f"      {i}. Doc {doc_idx} (Score: {score:.4f}): {documents[doc_idx][:50]}...")
            return True
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception: {e}")
        return False

        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print(f"\n{Colors.CYAN}{'='*70}")
    print("COMPREHENSIVE API ENDPOINT TESTING".center(70))
    print(f"{'='*70}{Colors.END}\n")
    
    print_info(f"Testing server at: {BASE_URL}")
    print_info(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # ========== CHAT COMPLETIONS ==========
    print_section("CHAT COMPLETIONS")
    
    # llama-cpp backend
    results['llama-cpp Chat (Non-Streaming)'] = test_llama_chat(streaming=False)
    results['llama-cpp Chat (Streaming)'] = test_llama_chat(streaming=True)
    
    # MLX backend
    results['MLX Chat (Non-Streaming)'] = test_mlx_chat(streaming=False)
    results['MLX Chat (Streaming)'] = test_mlx_chat(streaming=True)
    
    # ========== EMBEDDINGS ==========
    print_section("EMBEDDINGS")
    
    # llama-cpp embeddings
    results['llama-cpp Embeddings'] = test_llama_embeddings()
    
    # MLX embeddings
    results['MLX Embeddings'] = test_mlx_embeddings()
    
    # sentence-transformers embeddings
    results['Sentence-Transformers Embeddings'] = test_sentence_transformers_embeddings()
    
    # ========== RERANKING ==========
    print_section("RERANKING")
    
    results['Document Reranking'] = test_reranker()
    
    # ========== SUMMARY ==========
    print(f"\n{Colors.CYAN}{'='*70}")
    print("TEST SUMMARY".center(70))
    print(f"{'='*70}{Colors.END}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    # Group results by category
    chat_results = {k: v for k, v in results.items() if 'Chat' in k}
    embed_results = {k: v for k, v in results.items() if 'Embeddings' in k}
    rerank_results = {k: v for k, v in results.items() if 'Reranking' in k}
    
    print(f"{Colors.BLUE}Chat Completions:{Colors.END}")
    for test_name, result in chat_results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{Colors.BLUE}Embeddings:{Colors.END}")
    for test_name, result in embed_results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{Colors.BLUE}Reranking:{Colors.END}")
    for test_name, result in rerank_results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.END}")
    print(f"{Colors.BLUE}Total: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{'='*70}")
        print("✓ ALL TESTS PASSED!".center(70))
        print(f"{'='*70}{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{'='*70}")
        print(f"⚠ {total - passed} test(s) failed".center(70))
        print(f"{'='*70}{Colors.END}\n")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
