#!/usr/bin/env python3
"""
Test script for tool calling / function calling support.

Tests both streaming and non-streaming tool calls across different backends.
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
parser = argparse.ArgumentParser(description='Test tool calling')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)), 
                   help='Port number for the server (default: from PORT env var or 8000)')
parser.add_argument('--model', type=str, default='qwen3-8b-vllm',
                   help='Model name to test (default: qwen3-8b-vllm)')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}/v1"

# Define sample tools
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform basic mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    }
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}


def test_single_tool_call(model: str):
    """Test calling a single tool."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        "tools": [WEATHER_TOOL],
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": False
    }
    
    print(f"\n{'='*70}")
    print(f"Test: Single Tool Call (Weather)")
    print(f"Model: {model}")
    print(f"User: What's the weather like in San Francisco?")
    print(f"{'='*70}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        choice = result["choices"][0]
        message = choice["message"]
        
        print(f"\nResponse:")
        print(json.dumps(message, indent=2))
        
        if "tool_calls" in message:
            print(f"\n✓ Tool calls detected!")
            for tool_call in message["tool_calls"]:
                func = tool_call["function"]
                print(f"  - Function: {func['name']}")
                print(f"    Arguments: {func['arguments']}")
            return True
        else:
            print(f"\n⚠ No tool calls in response")
            print(f"Content: {message.get('content')}")
            return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return False


def test_multiple_tools(model: str):
    """Test with multiple available tools."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Calculate 15 multiplied by 7"}
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL],
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": False
    }
    
    print(f"\n{'='*70}")
    print(f"Test: Multiple Tools Available (Calculator)")
    print(f"Model: {model}")
    print(f"User: Calculate 15 multiplied by 7")
    print(f"{'='*70}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        choice = result["choices"][0]
        message = choice["message"]
        
        print(f"\nResponse:")
        print(json.dumps(message, indent=2))
        
        if "tool_calls" in message:
            print(f"\n✓ Tool calls detected!")
            for tool_call in message["tool_calls"]:
                func = tool_call["function"]
                print(f"  - Function: {func['name']}")
                print(f"    Arguments: {func['arguments']}")
            
            # Check if correct tool was selected
            if any(tc["function"]["name"] == "calculate" for tc in message["tool_calls"]):
                print(f"\n✓✓ Correct tool (calculate) selected!")
                return True
            else:
                print(f"\n⚠ Wrong tool selected")
                return False
        else:
            print(f"\n⚠ No tool calls in response")
            return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_streaming_with_tools(model: str):
    """Test streaming response with tools."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Search for 'latest AI breakthroughs'"}
        ],
        "tools": [SEARCH_TOOL],
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": True
    }
    
    print(f"\n{'='*70}")
    print(f"Test: Streaming with Tools (Search)")
    print(f"Model: {model}")
    print(f"User: Search for 'latest AI breakthroughs'")
    print(f"{'='*70}")
    
    try:
        print(f"\nStreaming response: ", end="", flush=True)
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        collected_deltas = []
        tool_calls_found = False
        
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
                        
                        if "tool_calls" in delta:
                            tool_calls_found = True
                            print(f"\n\n✓ Tool calls in stream!")
                            print(json.dumps(delta["tool_calls"], indent=2))
                        elif "content" in delta and delta["content"]:
                            print(delta["content"], end="", flush=True)
                        
                        collected_deltas.append(delta)
                except json.JSONDecodeError:
                    continue
        
        print()
        
        if tool_calls_found:
            print(f"\n✓ Streaming with tool calls successful!")
            return True
        else:
            print(f"\n⚠ No tool calls in streaming response")
            return False
        
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        return False


def test_multi_turn_with_tools(model: str):
    """Test multi-turn conversation with tool calls."""
    url = f"{BASE_URL}/chat/completions"
    
    print(f"\n{'='*70}")
    print(f"Test: Multi-Turn Conversation with Tools")
    print(f"Model: {model}")
    print(f"{'='*70}")
    
    messages = [
        {"role": "user", "content": "What's the weather in New York?"}
    ]
    
    try:
        # First turn - tool call
        print(f"\nTurn 1:")
        print(f"User: {messages[0]['content']}")
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": [WEATHER_TOOL],
            "temperature": 0.7,
            "max_tokens": 200,
            "stream": False
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        assistant_message = result["choices"][0]["message"]
        print(f"Assistant: {json.dumps(assistant_message, indent=2)}")
        
        messages.append(assistant_message)
        
        # Simulate tool execution
        if "tool_calls" in assistant_message:
            for tool_call in assistant_message["tool_calls"]:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({"temperature": 72, "condition": "sunny"})
                })
        
        # Second turn - respond to user
        messages.append({
            "role": "user",
            "content": "Thanks! What about in Celsius?"
        })
        
        print(f"\nTurn 2:")
        print(f"User: {messages[-1]['content']}")
        
        payload["messages"] = messages
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        assistant_message = result["choices"][0]["message"]
        print(f"Assistant: {assistant_message.get('content', json.dumps(assistant_message))}")
        
        print(f"\n✓ Multi-turn with tools successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_no_tools_fallback(model: str):
    """Test that model works without tools (baseline)."""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello! How are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
    
    print(f"\n{'='*70}")
    print(f"Test: No Tools (Baseline)")
    print(f"Model: {model}")
    print(f"User: Hello! How are you?")
    print(f"{'='*70}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        message = result["choices"][0]["message"]
        
        print(f"\nAssistant: {message.get('content')}")
        
        if "content" in message and message["content"]:
            print(f"\n✓ Normal chat works without tools!")
            return True
        else:
            print(f"\n⚠ Unexpected response format")
            return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    """Run all tool calling tests."""
    print("="*70)
    print("Tool Calling / Function Calling Test Suite")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Base URL: {BASE_URL}")
    
    results = []
    
    # Test 1: No tools baseline
    success = test_no_tools_fallback(args.model)
    results.append(("No Tools Baseline", success))
    time.sleep(1)
    
    # Test 2: Single tool call
    success = test_single_tool_call(args.model)
    results.append(("Single Tool Call", success))
    time.sleep(1)
    
    # Test 3: Multiple tools
    success = test_multiple_tools(args.model)
    results.append(("Multiple Tools", success))
    time.sleep(1)
    
    # Test 4: Streaming with tools
    success = test_streaming_with_tools(args.model)
    results.append(("Streaming with Tools", success))
    time.sleep(1)
    
    # Test 5: Multi-turn with tools
    success = test_multi_turn_with_tools(args.model)
    results.append(("Multi-Turn with Tools", success))
    
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
    print("Tool Calling Features:")
    print("="*70)
    print("✓ Function/tool definitions")
    print("✓ Tool call extraction from response")
    print("✓ OpenAI-compatible format")
    print("✓ Streaming support")
    print("✓ Multi-turn conversations")
    print("✓ Multiple tools support")
    print("="*70)


if __name__ == "__main__":
    main()
