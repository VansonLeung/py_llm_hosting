from typing import Dict, Any, List

def format_chat_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Format raw LLM response to OpenAI chat completion format."""
    # Assume raw response is already in OpenAI format or adapt
    # For now, return as is, but ensure object is "chat.completion"
    formatted = raw_response.copy()
    formatted["object"] = "chat.completion"
    return formatted

def format_embeddings_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Format raw LLM response to OpenAI embeddings format."""
    formatted = raw_response.copy()
    formatted["object"] = "list"
    return formatted

def format_rerank_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Format raw LLM response to OpenAI rerank format."""
    formatted = {
        "object": "list",
        "model": raw_response.get("model", ""),
        "results": raw_response.get("results", [])
    }
    return formatted