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
            name="alpha",
            model_name="gpt-alpha",
            mode=ServerMode.PROXY,
            status=ServerStatus.ACTIVE,
            endpoint_url="https://example.com",
        ),
        LLMServer(
            name="beta",
            model_name="beta-model",
            mode=ServerMode.SELF_HOSTED,
            status=ServerStatus.INACTIVE,
            model_path="/models/beta",
            backend_type="mlx",
        ),
    ]


def _patch_persistence(monkeypatch, servers):
    class _FakePersistence:
        def __init__(self, *args, **kwargs):
            self._servers = list(servers)

        def get_servers(self):
            return self._servers

        def set_servers(self, items):
            self._servers = list(items)

        def add_server(self, server):
            self._servers.append(server)

        def update_server(self, server):
            for idx, current in enumerate(self._servers):
                if current.id == server.id:
                    self._servers[idx] = server
                    break

        def remove_server(self, server_id):
            self._servers = [srv for srv in self._servers if srv.id != server_id]

    monkeypatch.setattr("src.web.admin.Persistence", _FakePersistence)


def test_admin_list_servers(client, sample_servers, monkeypatch):
    _patch_persistence(monkeypatch, sample_servers)

    response = client.get("/admin/api/servers")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == len(sample_servers)
    assert data[0]["name"] == sample_servers[0].name


def test_admin_replace_servers_raw(client, sample_servers, monkeypatch):
    _patch_persistence(monkeypatch, sample_servers)

    payload = [server.model_dump() for server in sample_servers[:1]]
    response = client.put("/admin/api/servers/raw", json=payload)

    assert response.status_code == 200
    assert response.json()["count"] == 1
```}