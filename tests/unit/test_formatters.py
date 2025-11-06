import pytest
from src.libs.formatters import format_chat_response, format_embeddings_response, format_rerank_response

def test_format_chat_response():
    """Test formatting chat completion response."""
    raw_response = {
        "id": "test-id",
        "object": "text_completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "text": "Hello!",
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

    formatted = format_chat_response(raw_response)
    assert formatted["object"] == "chat.completion"
    assert formatted["choices"][0]["message"]["content"] == "Hello!"

def test_format_embeddings_response():
    """Test formatting embeddings response."""
    raw_response = {
        "object": "list",
        "data": [{"embedding": [0.1, 0.2], "index": 0}],
        "model": "test-model",
        "usage": {"prompt_tokens": 5, "total_tokens": 5}
    }

    formatted = format_embeddings_response(raw_response)
    assert formatted["object"] == "list"
    assert len(formatted["data"]) == 1

def test_format_rerank_response():
    """Test formatting rerank response."""
    raw_response = {
        "results": [{"index": 0, "relevance_score": 0.9}]
    }

    formatted = format_rerank_response(raw_response)
    assert formatted["object"] == "list"
    assert formatted["results"][0]["relevance_score"] == 0.9