#!/usr/bin/env python3
"""
Test script for vLLM backend.

Tests both streaming and non-streaming chat completions using vLLM backend.
vLLM is optimized for high-performance GPU inference with throughput optimization.
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
parser = argparse.ArgumentParser(description='Test vLLM backend')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)), 
                   help='Port number for the server (default: from PORT env var or 8000)')
parser.add_argument('--model', type=str, default='qwen2.5-0.5b-vllm',
                   help='Model name to test (default: qwen2.5-0.5b-vllm)')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}/v1"


def test_non_streaming(model: str, prompt: str):
    """Test non-streaming chat completion."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    
    print(f"\n{'='*70}")
    print(f"Test: Non-Streaming Chat Completion")
    print(f"Model: {model}")
    print(f"Backend: vLLM")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        finish_reason = result["choices"][0].get("finish_reason", "unknown")
        
        elapsed = time.time() - start_time
        
        print(f"\nResponse: {content}")
        print(f"\nMetadata:")
        print(f"  Finish Reason: {finish_reason}")
        print(f"  Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Tokens/sec: {usage['completion_tokens'] / elapsed:.2f}")
        
        return True, result
        
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return False, None


def test_streaming(model: str, prompt: str):
    """Test streaming chat completion."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True
    }
    
    print(f"\n{'='*70}")
    print(f"Test: Streaming Chat Completion")
    print(f"Model: {model}")
    print(f"Backend: vLLM")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    
    start_time = time.time()
    full_response = ""
    chunk_count = 0
    
    try:
        print("\nResponse: ", end="", flush=True)
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        first_token_time = None
        
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
                            full_response += content
                            chunk_count += 1
                        
                        finish_reason = chunk["choices"][0].get("finish_reason")
                        if finish_reason:
                            elapsed = time.time() - start_time
                            ttft = first_token_time - start_time if first_token_time else 0
                            
                            print(f"\n\nMetadata:")
                            print(f"  Finish Reason: {finish_reason}")
                            print(f"  Chunks: {chunk_count}")
                            print(f"  Total Time: {elapsed:.2f}s")
                            print(f"  Time to First Token (TTFT): {ttft:.3f}s")
                            if chunk_count > 0:
                                print(f"  Avg Time per Chunk: {elapsed / chunk_count:.3f}s")
                            
                except json.JSONDecodeError as e:
                    print(f"\n[Warning: Failed to decode chunk: {e}]")
                    continue
        
        print()
        return True, full_response
        
    except Exception as e:
        print(f"\n\nError: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return False, None


def test_multi_turn_conversation(model: str):
    """Test multi-turn conversation."""
    url = f"{BASE_URL}/chat/completions"
    
    print(f"\n{'='*70}")
    print(f"Test: Multi-Turn Conversation")
    print(f"Model: {model}")
    print(f"Backend: vLLM")
    print(f"{'='*70}")
    
    messages = [
        {"role": "user", "content": "What is 5 + 7?"}
    ]
    
    try:
        # First turn
        print("\nTurn 1:")
        print(f"User: {messages[0]['content']}")
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": False
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        assistant_response = result["choices"][0]["message"]["content"]
        print(f"Assistant: {assistant_response}")
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
        
        # Second turn
        messages.append({"role": "user", "content": "What is double that number?"})
        
        print(f"\nTurn 2:")
        print(f"User: {messages[-1]['content']}")
        
        payload["messages"] = messages
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        assistant_response = result["choices"][0]["message"]["content"]
        print(f"Assistant: {assistant_response}")
        
        print(f"\n✓ Multi-turn conversation successful")
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        return False


def test_temperature_variations(model: str, prompt: str):
    """Test different temperature settings."""
    url = f"{BASE_URL}/chat/completions"
    
    print(f"\n{'='*70}")
    print(f"Test: Temperature Variations")
    print(f"Model: {model}")
    print(f"Backend: vLLM")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    
    temperatures = [0.0, 0.5, 1.0]
    
    for temp in temperatures:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": temp,
                "stream": False
            }
            
            print(f"\nTemperature: {temp}")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            
        except Exception as e:
            print(f"Error at temperature {temp}: {e}")
            return False
    
    print(f"\n✓ Temperature variations test successful")
    return True


def main():
    """Run all vLLM tests."""
    print("="*70)
    print("vLLM Backend Test Suite")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Base URL: {BASE_URL}")
    
    results = []
    
    # Test 1: Non-streaming chat completion
    success, _ = test_non_streaming(
        args.model,
        "Write a haiku about artificial intelligence"
    )
    results.append(("Non-Streaming Chat", success))
    
    time.sleep(1)
    
    # Test 2: Streaming chat completion
    success, _ = test_streaming(
        args.model,
        "Tell me a short story about a robot learning to paint"
    )
    results.append(("Streaming Chat", success))
    
    time.sleep(1)
    
    # Test 3: Multi-turn conversation
    success = test_multi_turn_conversation(args.model)
    results.append(("Multi-Turn Conversation", success))
    
    time.sleep(1)
    
    # Test 4: Temperature variations
    success = test_temperature_variations(
        args.model,
        "What is the meaning of life?"
    )
    results.append(("Temperature Variations", success))
    
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
    print("vLLM Backend Features:")
    print("="*70)
    print("✓ High-performance GPU inference")
    print("✓ Optimized for throughput")
    print("✓ PagedAttention for efficient memory management")
    print("✓ Continuous batching")
    print("✓ Fast token generation")
    print("✓ Multi-GPU support (tensor parallelism)")
    print("✓ HuggingFace model compatibility")
    print("="*70)


if __name__ == "__main__":
    main()
