import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api import app

client = TestClient(app)

@patch('src.services.proxy.proxy_request')
def test_api_proxy_integration(mock_proxy):
    """Test API request proxying integration."""
    mock_proxy.return_value = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello!"
    mock_proxy.assert_called_once()