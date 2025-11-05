import pytest
from pydantic import ValidationError
from src.models.server import LLMServer

def test_valid_server_creation():
    """Test creating a valid server."""
    server = LLMServer(
        name="Valid Server",
        endpoint_url="https://api.example.com/v1/chat/completions",
        model_name="gpt-3.5-turbo"
    )
    assert server.name == "Valid Server"
    assert server.endpoint_url == "https://api.example.com/v1/chat/completions"
    assert server.model_name == "gpt-3.5-turbo"
    assert server.status == "inactive"

def test_invalid_url():
    """Test invalid URL validation."""
    with pytest.raises(ValidationError):
        LLMServer(
            name="Test",
            endpoint_url="not-a-url",
            model_name="model"
        )

def test_name_too_long():
    """Test name length validation."""
    long_name = "a" * 51
    with pytest.raises(ValidationError):
        LLMServer(
            name=long_name,
            endpoint_url="https://api.example.com/v1/chat/completions",
            model_name="model"
        )

def test_empty_model_name():
    """Test empty model name validation."""
    with pytest.raises(ValidationError):
        LLMServer(
            name="Test",
            endpoint_url="https://api.example.com/v1/chat/completions",
            model_name=""
        )