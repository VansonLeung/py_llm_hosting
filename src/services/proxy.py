import httpx
from typing import Dict, Any
from src.libs.logging import logger

async def proxy_request(endpoint_url: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Proxy a request to the LLM server."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint_url,
                json=request_data,
                timeout=300.0  # 5 minutes for LLM responses
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error proxying to {endpoint_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error proxying to {endpoint_url}: {e}")
        raise