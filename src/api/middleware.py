from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from src.lib.config import settings
from src.lib.logging import logger

async def api_key_middleware(request: Request, call_next):
    """Middleware to check API key authentication."""
    # Skip auth for certain paths if needed
    if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    # Check for API key
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Missing or invalid Authorization header")
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Missing or invalid API key", "type": "authentication_error"}}
        )

    api_key = auth_header[7:]  # Remove "Bearer "
    if settings.llm_api_key and api_key != settings.llm_api_key:
        logger.warning("Invalid API key provided")
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "authentication_error"}}
        )

    # Proceed
    response = await call_next(request)
    return response