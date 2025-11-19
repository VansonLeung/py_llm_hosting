from fastapi.testclient import TestClient
import pytest

from src.api import app
from src.models.server import LLMServer, ServerMode, ServerStatus


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_servers():
    return [
        LLMServer(
            name="local-qwen",
            model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit",
            mode=ServerMode.SELF_HOSTED,
            status=ServerStatus.ACTIVE,
            model_path="/models/qwen",
            backend_type="mlx_vlm",
        ),
        LLMServer(
            name="proxy-gpt",
            model_name="gpt-4o-mini",
            mode=ServerMode.PROXY,
            status=ServerStatus.INACTIVE,
            endpoint_url="https://api.openai.com/v1/chat/completions",
        ),
    ]


def _patch_persistence(monkeypatch, servers):
    class _FakePersistence:
        def __init__(self, *args, **kwargs):
            self._servers = servers

        def get_servers(self):
            return self._servers

    monkeypatch.setattr("src.api.models.Persistence", _FakePersistence)


def test_list_models_returns_servers(client, sample_servers, monkeypatch):
    _patch_persistence(monkeypatch, sample_servers)

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == len(sample_servers)
    first_model = payload["data"][0]
    assert first_model["id"] == sample_servers[0].model_name
    assert first_model["metadata"]["mode"] == sample_servers[0].mode.value


def test_retrieve_model_by_name(client, sample_servers, monkeypatch):
    _patch_persistence(monkeypatch, sample_servers)

    target = sample_servers[1]
    response = client.get(f"/v1/models/{target.model_name}")

    assert response.status_code == 200
    model_payload = response.json()
    assert model_payload["id"] == target.model_name
    assert model_payload["metadata"]["status"] == target.status.value


def test_retrieve_model_not_found(client, monkeypatch):
    _patch_persistence(monkeypatch, [])

    response = client.get("/v1/models/unknown-model")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
