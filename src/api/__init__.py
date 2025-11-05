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

# Import and include routers
from . import chat, embeddings, ranking

app.include_router(chat.router, prefix="/v1")
app.include_router(embeddings.router, prefix="/v1")
app.include_router(ranking.router, prefix="/v1")
