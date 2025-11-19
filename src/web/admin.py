from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.libs.persistence import Persistence
from src.models.server import LLMServer
from src.web.log_stream import log_stream_manager

router = APIRouter(prefix="/admin", tags=["admin"])
api_router = APIRouter(prefix="/api", tags=["admin-api"])

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@router.get("", response_class=HTMLResponse)
async def admin_home(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@api_router.get("/servers", response_model=List[LLMServer])
def list_servers():
    persistence = Persistence()
    return persistence.get_servers()


@api_router.post("/servers", response_model=LLMServer)
def create_server(server: LLMServer):
    persistence = Persistence()
    persistence.add_server(server)
    return server


@api_router.put("/servers/{server_id}", response_model=LLMServer)
def update_server(server_id: str, server: LLMServer):
    if server.id != server_id:
        raise HTTPException(status_code=400, detail="Server ID mismatch")
    persistence = Persistence()
    persistence.update_server(server)
    return server


@api_router.delete("/servers/{server_id}")
def delete_server(server_id: str):
    persistence = Persistence()
    persistence.remove_server(server_id)
    return {"status": "deleted"}


@api_router.get("/servers/raw")
def get_servers_raw():
    persistence = Persistence()
    servers = persistence.get_servers()
    return [server.model_dump() for server in servers]


@api_router.put("/servers/raw")
def replace_servers_raw(payload: List[LLMServer]):
    persistence = Persistence()
    persistence.set_servers(payload)
    return {"status": "updated", "count": len(payload)}


@router.get("/logs/stream")
async def stream_logs(request: Request, contains: str | None = None):
    queue = await log_stream_manager.subscribe(contains)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                entry = await queue.get()
                yield f"data: {json.dumps(entry)}\n\n"
        finally:
            await log_stream_manager.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


router.include_router(api_router)
