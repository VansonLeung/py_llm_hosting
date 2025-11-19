import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import chat, embeddings, ranking, models
from src.web import admin

app = FastAPI(
    title="LLM Endpoint Hosting API",
    description="OpenAI-compatible API for hosted LLM endpoints",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to initialize model_manager with persistence
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    from src.libs.logging import logger
    from src.libs.persistence import Persistence
    from src.services.model_manager import model_manager
    from src.web.log_stream import log_stream_handler, log_stream_manager
    
    persistence = Persistence()
    model_manager.set_persistence(persistence)

    loop = asyncio.get_event_loop()
    log_stream_manager.set_loop(loop)
    if log_stream_handler not in logger.handlers:
        logger.addHandler(log_stream_handler)

app.include_router(chat.router, prefix="/v1")
app.include_router(embeddings.router, prefix="/v1")
app.include_router(ranking.router, prefix="/v1")
app.include_router(models.router, prefix="/v1")
app.include_router(admin.router)
