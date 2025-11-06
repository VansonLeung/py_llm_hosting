# LlamaCpp Backend - Tool/Function Calling Implementation

## Summary

Implemented OpenAI-style tool/function calling support in the LlamaCpp backend (`src/backends/llamacpp_backend.py`).

## Changes Made

### 1. Updated `generate_chat` Method Signature
- Added `tools` parameter: `Optional[List[Dict[str, Any]]] = None`
- Maintains backward compatibility (tools defaults to None)

### 2. Added Tool Support Logic
The implementation tries two approaches:

**Native Support (Primary):**
- Attempts to pass `tools` directly to `llama-cpp-python`'s `create_chat_completion`
- Works if the library version supports the tools parameter

**Fallback Support (Secondary):**
- If native support is not available, falls back to manual formatting
- Formats tools as a system message with instructions
- Extracts tool calls from model responses

### 3. Added Helper Methods

#### `_format_messages_with_tools(messages, tools)`
Formats messages with tool information:
- Converts tools to a descriptive system message
- Includes tool names, descriptions, and parameters
- Provides instructions for tool invocation format

#### `_extract_tool_calls(text)`
Extracts tool calls from generated text:
- Supports multiple formats:
  - JSON format: `{"tool_calls": [{"name": "...", "arguments": {...}}]}`
  - XML format: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
  - Direct function format: `function_name(args)`
- Converts to OpenAI-compatible format with unique IDs

### 4. Documentation
- Added comprehensive docstring explaining tool support
- Included usage examples in the module docstring
- Created example file: `examples/example_llamacpp_tools.py`

## Features

### Tool Call Detection
The backend automatically detects when a model attempts to call a tool and formats it in OpenAI-compatible format:

```python
{
    "id": "call_123_0",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"San Francisco, CA\"}"
    }
}
```

### Streaming Support
Tool support works with both streaming and non-streaming responses.

### Error Handling
- Gracefully falls back to manual formatting if native support unavailable
- Logs warnings when fallback is used
- Maintains all existing error handling patterns

## Usage Example

```python
from src.backends.llamacpp_backend import LlamaCppBackend

backend = LlamaCppBackend(model_path="model.gguf")
await backend.load_model()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

result = await backend.generate_chat(
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools
)

# Check for tool calls
if "tool_calls" in result["choices"][0].get("message", {}):
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    # Process tool calls...
```

## Backward Compatibility

✅ **Fully backward compatible**
- Tools parameter is optional (defaults to None)
- Existing code without tools continues to work unchanged
- No breaking changes to existing API

## Testing

Created comprehensive tests covering:
- Tool message formatting
- Tool call extraction (JSON format)
- Tool call extraction (XML format)
- Handling responses without tool calls
- Backward compatibility

All tests pass successfully.

## Files Modified

1. `src/backends/llamacpp_backend.py` - Main implementation
2. `examples/example_llamacpp_tools.py` - Usage example (new file)

## Integration with API

The tool support integrates seamlessly with the existing API layer:
- `src/api/chat.py` already supports tools parameter
- LlamaCpp backend now handles tools consistently with other backends (e.g., MLX)
- No API changes required

## Future Enhancements

Potential improvements:
1. Support for parallel tool calls
2. Tool choice parameter (force specific tool, auto, none)
3. Enhanced tool call validation
4. Tool call confidence scoring
