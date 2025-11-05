import pytest
from src.lib.persistence import Persistence
from src.models.server import LLMServer

def test_add_server():
    """Test adding a server to persistence."""
    persistence = Persistence(data_file="test_servers.json")

    # Create a test server
    server = LLMServer(
        name="Test Server",
        endpoint_url="http://localhost:1234/v1/chat/completions",
        model_name="test-model"
    )

    # Add server
    persistence.add_server(server)

    # Verify server was added
    servers = persistence.get_servers()
    assert len(servers) == 1
    assert servers[0].name == "Test Server"
    assert servers[0].endpoint_url == "http://localhost:1234/v1/chat/completions"
    assert servers[0].model_name == "test-model"

    # Cleanup
    import os
    if os.path.exists("test_servers.json"):
        os.remove("test_servers.json")

def test_add_duplicate_server_name():
    """Test that adding a server with duplicate name raises error."""
    persistence = Persistence(data_file="test_servers.json")

    server1 = LLMServer(
        name="Test Server",
        endpoint_url="http://localhost:1234/v1/chat/completions",
        model_name="test-model"
    )
    server2 = LLMServer(
        name="Test Server",  # Same name
        endpoint_url="http://localhost:5678/v1/chat/completions",
        model_name="another-model"
    )

    persistence.add_server(server1)

    with pytest.raises(ValueError, match="already exists"):
        persistence.add_server(server2)

    # Cleanup
    import os
    if os.path.exists("test_servers.json"):
        os.remove("test_servers.json")