## ✅ Tool Calling / Function Calling Implementation

**Date:** 2025-11-06  
**Status:** ✅ Complete

### Overview

Added comprehensive **tool calling (function calling) support** to all chat endpoints, enabling models to invoke external tools and functions during conversations. This implementation follows the OpenAI function calling specification.

---

### 🎯 Changes Made

#### 1. **API Layer** (`src/api/chat.py`)

**Enhanced Request Model:**
- Added `tools` parameter to `ChatCompletionRequest`
- Tools are properly passed through to backends
- Handles tool calls in both streaming and non-streaming responses

**Streaming Support:**
- Detects tool calls in streamed deltas
- Yields structured tool call data
- Maintains OpenAI-compatible SSE format

**Non-Streaming Support:**
- Passes tools to backend `generate_chat()` method
- Returns tool calls in message object
- Sets `finish_reason` to `"tool_calls"` when tools are invoked

#### 2. **vLLM Backend** (`src/backends/vllm_backend.py`)

**Added Methods:**
- `_format_prompt_with_tools()` - Formats tools into system prompt
- `_extract_tool_calls()` - Parses tool calls from model output

**Features:**
- Tries tokenizer's native tool support first
- Falls back to manual prompt formatting
- Extracts tool calls from JSON format: `{"tool_calls": [...]}`
- Extracts tool calls from XML format: `<tool_call>...</tool_call>`
- Converts to OpenAI-compatible format
- Supports both streaming and non-streaming

**Tool Call Format:**
```python
{
    "id": "call_<timestamp>_<index>",
    "type": "function",
    "function": {
        "name": "tool_name",
        "arguments": "{...}"  # JSON string
    }
}
```

#### 3. **MLX Backend** (`src/backends/mlx_backend.py`)

**Added Methods:**
- `_format_prompt_with_tools()` - Same as vLLM
- `_extract_tool_calls()` - Same parsing logic

**Features:**
- Native chat template support with tools (if available)
- Manual prompt formatting fallback
- JSON and XML tool call extraction
- Streaming and non-streaming support

---

### 📋 Tool Definition Format

Tools are defined in OpenAI format:

```python
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
}
```

---

### 🚀 Usage Examples

#### **Non-Streaming with Tools**

```python
import requests

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-8b-vllm",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": tools,
        "stream": False
    }
)

result = response.json()
message = result["choices"][0]["message"]

if "tool_calls" in message:
    for tool_call in message["tool_calls"]:
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Args: {tool_call['function']['arguments']}")
```

#### **Streaming with Tools**

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-8b-vllm",
        "messages": [
            {"role": "user", "content": "Calculate 15 * 7"}
        ],
        "tools": [calculator_tool],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]
            
            if "tool_calls" in delta:
                print(f"Tool calls: {delta['tool_calls']}")
            elif "content" in delta:
                print(delta["content"], end="", flush=True)
```

#### **Multi-Turn with Tool Results**

```python
# Turn 1: User asks, model invokes tool
response1 = requests.post(url, json={
    "model": "qwen3-8b-vllm",
    "messages": [
        {"role": "user", "content": "What's the weather in NYC?"}
    ],
    "tools": [weather_tool]
})

message1 = response1.json()["choices"][0]["message"]

# Turn 2: Return tool result, model responds
if "tool_calls" in message1:
    tool_call = message1["tool_calls"][0]
    
    messages = [
        {"role": "user", "content": "What's the weather in NYC?"},
        message1,  # Assistant's tool call
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps({"temp": 72, "condition": "sunny"})
        }
    ]
    
    response2 = requests.post(url, json={
        "model": "qwen3-8b-vllm",
        "messages": messages,
        "tools": [weather_tool]
    })
```

---

### 🧪 Testing

**Test Script:** `examples/test_tool_calling.py`

```bash
# Run all tool calling tests
./examples/test_tool_calling.py --model qwen3-8b-vllm

# Test with different model
./examples/test_tool_calling.py --model qwen2.5-0.5b --port 8000
```

**Test Coverage:**
- ✅ No tools baseline (normal chat)
- ✅ Single tool call
- ✅ Multiple tools available
- ✅ Streaming with tools
- ✅ Multi-turn conversations with tool results
- ✅ Tool call extraction (JSON format)
- ✅ Tool call extraction (XML format)
- ✅ OpenAI format compatibility

---

### 📊 Response Format

#### **With Tool Calls:**

```json
{
    "id": "chatcmpl-12345",
    "object": "chat.completion",
    "created": 1699000000,
    "model": "qwen3-8b-vllm",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1699000000_0",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"San Francisco, CA\"}"
                }
            }]
        },
        "finish_reason": "tool_calls"
    }],
    "usage": {
        "prompt_tokens": 45,
        "completion_tokens": 23,
        "total_tokens": 68
    }
}
```

#### **Without Tool Calls (Normal):**

```json
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "The weather is sunny and 72°F in San Francisco."
        },
        "finish_reason": "stop"
    }]
}
```

---

### 🔧 Implementation Details

#### **Tool Call Extraction Patterns:**

1. **JSON Format:**
   ```json
   {"tool_calls": [{"name": "func", "arguments": {...}}]}
   ```

2. **XML Format:**
   ```xml
   <tool_call>{"name": "func", "arguments": {...}}</tool_call>
   ```

#### **Prompt Formatting:**

When tools are provided, they're added to the prompt:

```
You have access to the following tools:

- get_weather: Get the current weather for a location
  Parameters: {"type": "object", "properties": {...}}

To use a tool, respond with a JSON object in this format:
{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}
```

#### **Tokenizer Support:**

1. Try native tokenizer tool support:
   ```python
   tokenizer.apply_chat_template(messages, tools=tools)
   ```

2. Fallback to manual formatting if not supported

---

### 🎨 Supported Backends

| Backend | Tools Support | Streaming | Multi-Turn |
|---------|---------------|-----------|------------|
| **vLLM** | ✅ | ✅ | ✅ |
| **MLX** | ✅ | ✅ | ✅ |
| **llama.cpp** | ⚠️ Needs update | ⚠️ | ⚠️ |
| **Transformers** | ⚠️ Needs update | ⚠️ | ⚠️ |

---

### 📝 Best Practices

1. **Define Clear Tools:**
   - Descriptive names and descriptions
   - Well-defined parameters with types
   - Mark required parameters

2. **Handle Tool Results:**
   - Return results in JSON format
   - Include tool_call_id in response
   - Continue conversation naturally

3. **Error Handling:**
   - Check for `tool_calls` in response
   - Validate tool call arguments
   - Handle missing or invalid tools gracefully

4. **Model Selection:**
   - Use models trained for tool calling (Qwen2.5, GPT-4, etc.)
   - Test tool calling behavior with your specific model
   - Some models may need prompt tuning

---

### 🔍 Troubleshooting

**Issue: Model doesn't invoke tools**
- ✅ Use a tool-calling trained model (Qwen2.5-7B-Instruct, etc.)
- ✅ Make tool descriptions clear and relevant
- ✅ Try temperature 0.3-0.7 for more deterministic tool calls

**Issue: Tool call format not recognized**
- ✅ Check model output in logs
- ✅ Model may need fine-tuning for tool calling
- ✅ Try adding examples in system prompt

**Issue: Streaming doesn't show tool calls**
- ✅ Tool calls may appear at end of stream
- ✅ Check delta for `tool_calls` field
- ✅ Collect all deltas to reconstruct full response

---

### 📚 Additional Resources

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Qwen2.5 Tool Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Test Script](./examples/test_tool_calling.py)

---

**Status:** ✅ Production Ready  
**Tool Calling:** Fully Implemented  
**Backends:** vLLM ✅, MLX ✅  
**Format:** OpenAI Compatible ✅
