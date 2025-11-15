import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from src.backends.mlx_vlm_backend import MLXVLMBackend


def create_test_image_bytes(color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (4, 4), color=color)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_from_file_path(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(create_test_image_bytes())

    backend = MLXVLMBackend(model_path="dummy")
    image = backend._load_image_from_source(str(image_path))

    assert image.size == (4, 4)
    assert image.mode == "RGB"


def test_load_image_from_data_uri():
    image_bytes = create_test_image_bytes(color=(0, 255, 0))
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    backend = MLXVLMBackend(model_path="dummy")
    image = backend._load_image_from_source(data_uri)

    assert image.size == (4, 4)
    assert image.mode == "RGB"


def test_load_image_from_http_url(monkeypatch):
    image_bytes = create_test_image_bytes(color=(0, 0, 255))

    class DummyResponse:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        return DummyResponse(image_bytes)

    from src.backends import mlx_vlm_backend

    monkeypatch.setattr(mlx_vlm_backend.httpx, "get", fake_get)

    backend = MLXVLMBackend(model_path="dummy")
    image = backend._load_image_from_source("https://example.com/image.png")

    assert image.size == (4, 4)
    assert image.mode == "RGB"


def test_load_image_invalid_path(tmp_path):
    missing_path = tmp_path / "missing.png"
    backend = MLXVLMBackend(model_path="dummy")

    with pytest.raises(ValueError):
        backend._load_image_from_source(str(missing_path))
