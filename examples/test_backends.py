#!/usr/bin/env python3
"""
Test script comparing llama-cpp and MLX backends.
"""
import requests
import json
import time
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test LLM backends')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)), 
                   help='Port number for the server (default: from PORT env var or 8000)')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}/v1"


def test_backend(model: str, backend_name: str, prompt: str, stream: bool = False):
    """Test a specific backend."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "stream": stream
    }
    
    print(f"\n{'='*70}")
    print(f"Backend: {backend_name}")
    print(f"Model: {model}")
    print(f"Streaming: {stream}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        if stream:
            print("Response: ", end="", flush=True)
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if not line.strip() or line.strip() == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print()
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            usage = result["usage"]
            
            print(f"Response: {content}")
            print(f"\nTokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")
        
        elapsed = time.time() - start_time
        print(f"Time: {elapsed:.2f}s")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run comparison tests."""
    print("="*70)
    print("Backend Comparison Test: llama-cpp vs MLX")
    print("="*70)
    
    tests = [
        # Test 1: llama-cpp non-streaming
        ("tinyllama", "llama-cpp (GGUF)", "What is the capital of Japan?", False),
        
        # Test 2: llama-cpp streaming
        ("tinyllama", "llama-cpp (GGUF)", "Count from 1 to 5", True),
        
        # Test 3: MLX non-streaming
        ("qwen2.5-0.5b", "MLX (Apple Silicon)", "What is 10 * 10?", False),
        
        # Test 4: MLX streaming
        ("qwen2.5-0.5b", "MLX (Apple Silicon)", "Write a short poem about AI", True),
    ]
    
    results = []
    for model, backend, prompt, stream in tests:
        success = test_backend(model, backend, prompt, stream)
        results.append((backend, "streaming" if stream else "non-streaming", success))
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    for backend, mode, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {backend} ({mode})")
    
    print("\n" + "="*70)
    print("Comparison Notes:")
    print("="*70)
    print("llama-cpp:")
    print("  - Supports GGUF quantized models")
    print("  - Efficient CPU/GPU inference")
    print("  - True token-by-token streaming")
    print("  - Supports embeddings")
    print("\nMLX:")
    print("  - Apple Silicon (M-series) optimized")
    print("  - Metal acceleration")
    print("  - Word-based streaming (simulated)")
    print("  - Best performance on Mac")
    print("="*70)


if __name__ == "__main__":
    main()
