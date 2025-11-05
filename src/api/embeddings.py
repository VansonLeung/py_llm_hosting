from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional
from src.services.proxy import proxy_request
from src.lib.formatters import format_embeddings_response
from src.models.server import ServerMode
from src.models.backend import ModelCapability
import time

router = APIRouter()

class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None


async def handle_self_hosted_embeddings(server, request: EmbeddingsRequest):
    """Handle embeddings using self-hosted model."""
    from src.services.model_manager import model_manager
    
    # Load model if not already loaded
    backend = model_manager.get_backend(server.id)
    if backend is None:
        backend = await model_manager.load_model(server)
    
    # Check if backend supports embeddings
    if not backend.supports_capability(ModelCapability.EMBEDDINGS):
        raise HTTPException(
            status_code=400,
            detail=f"Backend {server.backend_type} does not support embeddings"
        )
    
    # Convert input to list
    texts = [request.input] if isinstance(request.input, str) else request.input
    
    # Generate embeddings (backends are async)
    embeddings = []
    for text in texts:
        embedding_dict = await backend.embed([text])
        # Extract embedding array from response
        if isinstance(embedding_dict, dict) and "data" in embedding_dict:
            embedding = embedding_dict["data"][0]
        elif isinstance(embedding_dict, list):
            embedding = embedding_dict
        else:
            embedding = embedding_dict.get("embedding", [])
        embeddings.append(embedding)
    
    # Format response in OpenAI format
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb,
                "index": idx
            }
            for idx, emb in enumerate(embeddings)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": 0,  # TODO: implement token counting
            "total_tokens": 0
        }
    }


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """Create embeddings."""
    try:
        # Find server for the model
        from src.lib.persistence import Persistence
        persistence = Persistence()
        servers = persistence.get_servers()
        server = next((s for s in servers if s.model_name == request.model), None)
        if not server:
            raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found")

        # Route based on server mode
        if server.mode == ServerMode.SELF_HOSTED:
            return await handle_self_hosted_embeddings(server, request)
        else:
            # Proxy mode
            raw_response = await proxy_request(server.endpoint_url, request.model_dump())
            formatted_response = format_embeddings_response(raw_response)
            return formatted_response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))