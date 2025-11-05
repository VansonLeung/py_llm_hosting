from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.services.proxy import proxy_request
from src.lib.formatters import format_rerank_response
from src.models.server import ServerMode
from src.models.backend import ModelCapability

router = APIRouter()

class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = 10


async def handle_self_hosted_rerank(server, request: RerankRequest):
    """Handle reranking using self-hosted model."""
    from src.services.model_manager import model_manager
    
    # Load model if not already loaded
    backend = model_manager.get_backend(server.id)
    if backend is None:
        backend = await model_manager.load_model(server)
    
    # Check if backend supports reranking
    if not backend.supports_capability(ModelCapability.RERANK):
        raise HTTPException(
            status_code=400,
            detail=f"Backend {server.backend_type} does not support reranking"
        )
    
    # Perform reranking
    result = await backend.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n
    )
    
    # Return in compatible format
    return {
        "model": request.model,
        "results": result["results"],
        "usage": result.get("usage", {
            "total_tokens": 0
        })
    }


@router.post("/rerank")
async def create_rerank(request: RerankRequest):
    """Rerank documents."""
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
            return await handle_self_hosted_rerank(server, request)
        else:
            # Proxy mode
            raw_response = await proxy_request(server.endpoint_url, request.model_dump())
            formatted_response = format_rerank_response(raw_response)
            return formatted_response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))