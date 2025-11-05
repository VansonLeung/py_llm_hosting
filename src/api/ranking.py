from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.services.proxy import proxy_request
from src.lib.formatters import format_rerank_response

router = APIRouter()

class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = 10

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

        # Proxy the request
        raw_response = await proxy_request(server.endpoint_url, request.model_dump())
        formatted_response = format_rerank_response(raw_response)
        return formatted_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))