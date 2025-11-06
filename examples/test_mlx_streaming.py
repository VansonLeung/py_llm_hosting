#!/usr/bin/env python3
"""
Example script demonstrating streaming chat completions.
"""
import requests
import json
import sys
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test MLX streaming')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)), 
                   help='Port number for the server (default: from PORT env var or 8000)')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}/v1"


def stream_chat(prompt: str, model: str = "qwen2.5-0.5b"):
    """Stream a chat completion and print tokens as they arrive."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True
    }
    
    print(f"User: {prompt}")
    print("Assistant: ", end="", flush=True)
    
    try:
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
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue
        
        print()  # New line after completion
        
    except requests.exceptions.RequestException as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        return False
    
    return True


def non_stream_chat(prompt: str, model: str = "tinyllama"):
    """Send a non-streaming chat completion."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
    
    print(f"User: {prompt}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        
        print(f"Assistant: {content}")
        print(f"\n[Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total]")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return False
    
    return True


def main():
    """Main test function."""
    print("=" * 60)
    print("Streaming Chat Completion Test")
    print("=" * 60)
    
    # Test 1: Streaming
    print("\n--- Test 1: Streaming Mode ---")
    stream_chat("Tell me a short joke about programming")
    
    print("\n" + "=" * 60)
    
    # Test 2: Non-streaming
    print("\n--- Test 2: Non-Streaming Mode ---")
    non_stream_chat("What is 5 + 7?")
    
    print("\n" + "=" * 60)
    
    # Test 3: Another streaming example
    print("\n--- Test 3: Streaming with longer response ---")
    stream_chat("Write a haiku about AI")
    
    print("\n" + "=" * 60)
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
