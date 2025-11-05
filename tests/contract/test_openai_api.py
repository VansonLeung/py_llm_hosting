import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_chat_completions_endpoint():
    """Test chat completions endpoint exists and accepts requests."""
    response = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    # Since no server configured, expect error
    assert response.status_code == 400  # Or whatever the error is

def test_embeddings_endpoint():
    """Test embeddings endpoint exists."""
    response = client.post("/v1/embeddings", json={
        "model": "test-model",
        "input": "Hello world"
    })
    assert response.status_code == 400

def test_rerank_endpoint():
    """Test rerank endpoint exists."""
    response = client.post("/v1/rerank", json={
        "model": "test-model",
        "query": "What is AI?",
        "documents": ["AI is artificial intelligence"]
    })
    assert response.status_code == 400