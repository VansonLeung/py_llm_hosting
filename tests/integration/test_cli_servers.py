import subprocess
import json
import os
import pytest

def test_cli_add_server():
    """Test adding a server via CLI."""
    # Use a test data file
    test_file = "test_cli_servers.json"

    # Run CLI command to add server
    result = subprocess.run([
        "python", "-m", "src.cli.commands",
        "add-server",
        "--name", "CLI Test Server",
        "--endpoint", "http://localhost:1234/v1/chat/completions",
        "--model", "cli-model",
        "--data-file", test_file
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "Server added successfully" in result.stdout

    # Verify server was added
    assert os.path.exists(test_file)
    with open(test_file, 'r') as f:
        data = json.load(f)
    assert len(data["servers"]) == 1
    assert data["servers"][0]["name"] == "CLI Test Server"

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

def test_cli_list_servers():
    """Test listing servers via CLI."""
    test_file = "test_cli_servers.json"

    # First add a server
    subprocess.run([
        "python", "-m", "src.cli.commands",
        "add-server",
        "--name", "List Test Server",
        "--endpoint", "http://localhost:1234/v1/chat/completions",
        "--model", "list-model",
        "--data-file", test_file
    ], capture_output=True)

    # List servers
    result = subprocess.run([
        "python", "-m", "src.cli.commands",
        "list-servers",
        "--data-file", test_file
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "List Test Server" in result.stdout

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)