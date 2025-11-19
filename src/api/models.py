from datetime import timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from src.libs.logging import logger
from src.libs.persistence import Persistence
from src.models.server import LLMServer, ServerMode

router = APIRouter()


def _server_to_model_payload(server: LLMServer) -> Dict[str, Any]:
    """Convert an LLMServer into an OpenAI-compatible model payload."""
    created_dt = server.created_at
    if created_dt.tzinfo is None:
        created_dt = created_dt.replace(tzinfo=timezone.utc)
    created_ts = int(created_dt.timestamp())

    owned_by = server.backend_type or (
        "self-hosted" if server.mode == ServerMode.SELF_HOSTED else "proxy"
    )

    metadata = {
        "name": server.name,
        "mode": server.mode.value,
        "status": server.status.value,
        "backend_type": server.backend_type,
        "endpoint_url": server.endpoint_url,
    }

    return {
        "id": server.model_name,
        "object": "model",
        "created": created_ts,
        "owned_by": owned_by,
        "root": server.model_name,
        "parent": None,
        "permission": [],
        "metadata": metadata,
    }


def _get_servers() -> list[LLMServer]:
    persistence = Persistence()
    servers = persistence.get_servers()
    logger.debug(f"Loaded {len(servers)} servers for /v1/models")
    return servers


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models in OpenAI-compatible format."""
    servers = _get_servers()
    data = [_server_to_model_payload(server) for server in servers]
    return {
        "object": "list",
        "data": data,
    }


@router.get("/models/{model_id}")
async def retrieve_model(model_id: str) -> Dict[str, Any]:
    """Retrieve metadata for a single model by ID (model name or server ID)."""
    servers = _get_servers()
    server = next(
        (
            s
            for s in servers
            if s.model_name == model_id or s.id == model_id
        ),
        None,
    )

    if server is None:
        logger.error(f"Model '{model_id}' not found")
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return _server_to_model_payload(server)
