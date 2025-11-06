#!/usr/bin/env python3
"""
Example: Using Tool/Function Calling with LlamaCpp Backend

This example demonstrates how to use OpenAI-style tool/function calling
with the LlamaCpp backend.
"""

import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.backends.llamacpp_backend import LlamaCppBackend


async def main():
    print("=" * 60)
    print("LlamaCpp Backend - Tool/Function Calling Example")
    print("=" * 60)
    
    # Initialize backend
    backend = LlamaCppBackend(
        model_path="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF:meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf",
        n_ctx=2048,
        n_gpu_layers=0
    )
    
    # Define tools in OpenAI format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
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
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    # Example 1: Chat completion with tools
    print("\n1. Chat completion with tools:")
    print("-" * 60)
    
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
    
    print(f"User: {messages[0]['content']}")
    print("\nTools provided:")
    for tool in tools:
        func = tool["function"]
        print(f"  - {func['name']}: {func['description']}")
    
    # Note: This example shows the API structure. 
    # Actual execution requires a loaded model.
    print("\n⚠️  Note: This is a structure example. To run with a real model:")
    print("   1. Ensure the model is downloaded or accessible")
    print("   2. Load the model: await backend.load_model()")
    print("   3. Call generate_chat with tools parameter")
    
    # Example response structure
    print("\nExpected response structure with tool calls:")
    example_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "model-name",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "San Francisco, CA"})
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    }
    print(json.dumps(example_response, indent=2))
    
    # Example 2: Processing tool results
    print("\n2. Processing tool results:")
    print("-" * 60)
    print("After receiving tool calls, you would:")
    print("  1. Execute the requested function(s)")
    print("  2. Add tool results to the conversation")
    print("  3. Continue the chat")
    
    # Example conversation flow
    conversation_example = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco, CA"}'
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"temperature": 18, "unit": "celsius", "condition": "sunny"}'
        },
        # Model would then generate a natural language response based on the tool result
    ]
    
    print("\nConversation flow:")
    for i, msg in enumerate(conversation_example, 1):
        role = msg.get("role", "")
        if role == "user":
            print(f"{i}. User: {msg['content']}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                print(f"{i}. Assistant: [Calls tool: {msg['tool_calls'][0]['function']['name']}]")
            else:
                print(f"{i}. Assistant: {msg['content']}")
        elif role == "tool":
            print(f"{i}. Tool Result: {msg['content']}")
    
    # Example 3: Tool detection formats
    print("\n3. Supported tool call formats:")
    print("-" * 60)
    print("The backend can detect tool calls in multiple formats:")
    
    formats = [
        {
            "name": "JSON format",
            "example": '{"tool_calls": [{"name": "get_weather", "arguments": {"location": "SF"}}]}'
        },
        {
            "name": "XML format",
            "example": '<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>'
        }
    ]
    
    for fmt in formats:
        print(f"\n  {fmt['name']}:")
        print(f"    {fmt['example']}")
    
    print("\n" + "=" * 60)
    print("For full integration, see the API examples in examples/")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
