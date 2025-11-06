#!/usr/bin/env python3
"""
Streaming test script for vLLM backend.

Demonstrates and tests streaming capabilities of the vLLM backend,
including real-time token generation and performance metrics.
"""
import requests
import json
import sys
import os
import time
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test vLLM streaming')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)), 
                   help='Port number for the server (default: from PORT env var or 8000)')
parser.add_argument('--model', type=str, default='gpt2',
                   help='Model name to test (default: gpt2)')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}/v1"


def stream_chat(prompt: str, model: str = None, max_tokens: int = 150):
    """Stream a chat completion and print tokens as they arrive."""
    model = model or args.model
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    print(f"\n{'='*70}")
    print(f"Model: {model}")
    print(f"Backend: vLLM (streaming)")
    print(f"Max Tokens: {max_tokens}")
    print(f"{'='*70}")
    print(f"User: {prompt}")
    print("Assistant: ", end="", flush=True)
    
    try:
        start_time = time.time()
        first_token_time = None
        chunk_count = 0
        total_content = ""
        
        response = requests.post(
            url,
            json=payload,
            stream=True,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                # Skip empty lines and [DONE] marker
                if not line.strip() or line.strip() == "data: [DONE]":
                    continue
                
                # Remove "data: " prefix
                if line.startswith("data: "):
                    line = line[6:]
                
                try:
                    chunk = json.loads(line)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            
                            print(content, end="", flush=True)
                            total_content += content
                            chunk_count += 1
                        
                        # Check for finish
                        finish_reason = chunk["choices"][0].get("finish_reason")
                        if finish_reason:
                            total_time = time.time() - start_time
                            ttft = first_token_time - start_time if first_token_time else 0
                            
                            print(f"\n\n{'='*70}")
                            print(f"Streaming Metrics:")
                            print(f"  Total chunks: {chunk_count}")
                            print(f"  Time to first token (TTFT): {ttft:.3f}s")
                            print(f"  Total time: {total_time:.3f}s")
                            print(f"  Generation time: {total_time - ttft:.3f}s")
                            if chunk_count > 1:
                                print(f"  Avg time per chunk: {(total_time - ttft) / (chunk_count - 1):.3f}s")
                            print(f"  Finish reason: {finish_reason}")
                            print(f"{'='*70}")
                
                except json.JSONDecodeError:
                    continue
        
        return True, total_content
        
    except requests.exceptions.RequestException as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}", file=sys.stderr)
        return False, None


def non_stream_chat(prompt: str, model: str = None, max_tokens: int = 150):
    """Send a non-streaming chat completion for comparison."""
    model = model or args.model
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    print(f"\n{'='*70}")
    print(f"Model: {model}")
    print(f"Backend: vLLM (non-streaming)")
    print(f"Max Tokens: {max_tokens}")
    print(f"{'='*70}")
    print(f"User: {prompt}")
    
    try:
        start_time = time.time()
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        total_time = time.time() - start_time
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        
        print(f"Assistant: {content}")
        
        print(f"\n{'='*70}")
        print(f"Non-Streaming Metrics:")
        print(f"  Prompt tokens: {usage['prompt_tokens']}")
        print(f"  Completion tokens: {usage['completion_tokens']}")
        print(f"  Total tokens: {usage['total_tokens']}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Tokens/sec: {usage['completion_tokens'] / total_time:.2f}")
        print(f"{'='*70}")
        
        return True, content
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}", file=sys.stderr)
        return False, None


def compare_streaming_vs_non_streaming(prompt: str):
    """Compare streaming vs non-streaming performance."""
    print("\n" + "="*70)
    print("COMPARISON: Streaming vs Non-Streaming")
    print("="*70)
    
    # Test streaming
    print("\n--- Streaming Mode ---")
    stream_success, stream_content = stream_chat(prompt, max_tokens=100)
    
    time.sleep(1)
    
    # Test non-streaming
    print("\n--- Non-Streaming Mode ---")
    non_stream_success, non_stream_content = non_stream_chat(prompt, max_tokens=100)
    
    print("\n" + "="*70)
    print("Comparison Summary:")
    print("="*70)
    print("Streaming:")
    print("  ✓ Provides immediate feedback")
    print("  ✓ Better user experience for long responses")
    print("  ✓ Lower perceived latency (TTFT)")
    print("\nNon-Streaming:")
    print("  ✓ Simpler client implementation")
    print("  ✓ All metadata available immediately")
    print("  ✓ Better for short responses")
    print("="*70)
    
    return stream_success and non_stream_success


def test_long_response_streaming():
    """Test streaming with a longer response."""
    prompt = "Write a detailed paragraph explaining how neural networks learn through backpropagation."
    
    print("\n" + "="*70)
    print("TEST: Long Response Streaming")
    print("="*70)
    
    success, _ = stream_chat(prompt, max_tokens=200)
    return success


def test_rapid_fire_streaming():
    """Test multiple rapid streaming requests."""
    print("\n" + "="*70)
    print("TEST: Rapid Fire Streaming")
    print("="*70)
    
    prompts = [
        "Count from 1 to 5.",
        "What is 2 + 2?",
        "Name a color.",
    ]
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Request {i}/{len(prompts)} ---")
        success, _ = stream_chat(prompt, max_tokens=30)
        results.append(success)
        time.sleep(0.5)  # Brief pause between requests
    
    return all(results)


def test_streaming_with_system_message():
    """Test streaming with system message."""
    url = f"{BASE_URL}/chat/completions"
    
    print("\n" + "="*70)
    print("TEST: Streaming with System Message")
    print("="*70)
    
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
            {"role": "user", "content": "Tell me about the weather."}
        ],
        "temperature": 0.8,
        "max_tokens": 100,
        "stream": True
    }
    
    print("System: You are a helpful assistant that speaks like a pirate.")
    print("User: Tell me about the weather.")
    print("Assistant: ", end="", flush=True)
    
    try:
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
        
        print("\n")
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        return False


def main():
    """Main test function."""
    print("="*70)
    print("vLLM Streaming Test Suite")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Base URL: {BASE_URL}")
    
    results = []
    
    # Test 1: Basic streaming
    print("\n" + "="*70)
    print("TEST 1: Basic Streaming")
    print("="*70)
    success, _ = stream_chat("Write a haiku about coding")
    results.append(("Basic Streaming", success))
    
    time.sleep(1)
    
    # Test 2: Compare streaming vs non-streaming
    success = compare_streaming_vs_non_streaming("What is machine learning?")
    results.append(("Streaming vs Non-Streaming", success))
    
    time.sleep(1)
    
    # Test 3: Long response streaming
    success = test_long_response_streaming()
    results.append(("Long Response Streaming", success))
    
    time.sleep(1)
    
    # Test 4: Rapid fire streaming
    success = test_rapid_fire_streaming()
    results.append(("Rapid Fire Streaming", success))
    
    time.sleep(1)
    
    # Test 5: Streaming with system message
    success = test_streaming_with_system_message()
    results.append(("System Message Streaming", success))
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    print("\n" + "="*70)
    print("vLLM Streaming Benefits:")
    print("="*70)
    print("✓ Real-time token generation")
    print("✓ Lower perceived latency")
    print("✓ Better user experience")
    print("✓ Progressive content delivery")
    print("✓ Efficient resource utilization")
    print("="*70)


if __name__ == "__main__":
    main()
