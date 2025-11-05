from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    from src.services.model_manager import model_manager
    from src.lib.persistence import Persistence
    
    persistence = Persistence()
    model_manager.set_persistence(persistence)

# Import and include routers
from . import chat, embeddings, ranking

app.include_router(chat.router, prefix="/v1")
app.include_router(embeddings.router, prefix="/v1")
app.include_router(ranking.router, prefix="/v1")
