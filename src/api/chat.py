from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, AsyncIterator
from src.services.proxy import proxy_request
from src.libs.formatters import format_chat_response
from src.models.server import ServerMode
from src.models.backend import ModelCapability
import time
import uuid
import json

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]  # Allow string or list for multimodal
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    stream: bool = False


async def stream_chat_completion(server, request: ChatCompletionRequest) -> AsyncIterator[str]:
    """Stream chat completion chunks in SSE format."""
    from src.services.model_manager import model_manager
    
    # Load model if not already loaded
    backend = model_manager.get_backend(server.id)
    if backend is None:
        backend = await model_manager.load_model(server)
    
    # Convert messages to format expected by backend
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    # Generate streaming response
    response_dict = await backend.generate_chat(
        messages=messages,
        tools=request.tools,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=True
    )
    
    # If backend returns a stream/generator
    if hasattr(response_dict, '__aiter__'):
        async for token in response_dict:
            # Handle both string tokens and dict with tool_calls
            if isinstance(token, dict):
                delta = token
            else:
                delta = {"content": token}
            
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    
    # If backend returns dict with 'stream' generator
    elif isinstance(response_dict, dict) and "stream" in response_dict:
        stream_iter = response_dict["stream"]
        async for token in stream_iter:
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    
    # If backend doesn't support streaming, fall back to single response
    else:
        if isinstance(response_dict, dict) and "choices" in response_dict:
            # Extract content from OpenAI format
            content = response_dict["choices"][0]["message"]["content"]
        elif isinstance(response_dict, dict):
            content = response_dict.get("text", "")
        else:
            content = str(response_dict)
        
        # Send as single chunk
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send final chunk with finish_reason
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def handle_self_hosted_chat(server, request: ChatCompletionRequest):
    """Handle chat completion using self-hosted model."""
    from src.services.model_manager import model_manager
    from src.libs.logging import logger
    
    # Load model if not already loaded
    backend = model_manager.get_backend(server.id)
    if backend is None:
        try:
            logger.info(f"Loading model {server.model_name} (backend: {server.backend_type})")
            backend = await model_manager.load_model(server)
        except Exception as e:
            error_msg = f"Failed to load model '{server.model_name}' (id: {server.id}, backend: {server.backend_type}). Error: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Check if backend supports chat
    if not backend.supports_capability(ModelCapability.TEXT_GENERATION):
        error_msg = f"Backend {server.backend_type} does not support text generation"
        logger.error(error_msg)
        raise HTTPException(
            status_code=400, 
            detail=error_msg
        )
    
    # Check for multimodal content
    has_images = any(
        isinstance(msg.content, list) for msg in request.messages
    )
    
    if has_images and not backend.supports_capability(ModelCapability.VISION):
        error_msg = f"Backend {server.backend_type} does not support vision/multimodal input"
        logger.error(error_msg)
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(server, request),
            media_type="text/event-stream"
        )
    
    # Convert messages to format expected by backend
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    
    # Generate response (backends are async)
    response_dict = await backend.generate_chat(
        messages=messages,
        tools=request.tools,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=False
    )
    
    # Extract response - backends return OpenAI-compatible format
    if isinstance(response_dict, dict):
        # If backend returns full OpenAI response, use it directly
        if "choices" in response_dict:
            return response_dict
        # If backend returns just text
        response_text = response_dict.get("text", "")
    else:
        response_text = str(response_dict)
    
    # Format response in OpenAI format
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # TODO: implement token counting
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    from src.libs.logging import logger
    
    try:
        # Find server for the model
        from src.libs.persistence import Persistence
        persistence = Persistence()
        servers = persistence.get_servers()
        server = next((s for s in servers if s.model_name == request.model), None)
        if not server:
            error_msg = f"Model '{request.model}' not found. Available models: {[s.model_name for s in servers]}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        # Route based on server mode
        if server.mode == ServerMode.SELF_HOSTED:
            try:
                return await handle_self_hosted_chat(server, request)
            except RuntimeError as e:
                # Backend errors - provide detailed message
                error_msg = f"Backend error for model '{request.model}': {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=str(e))
            except ValueError as e:
                # Invalid parameters
                error_msg = f"Invalid request parameters for model '{request.model}': {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=str(e))
        else:
            # Proxy mode
            try:
                raw_response = await proxy_request(server.endpoint_url, request.model_dump())
                formatted_response = format_chat_response(raw_response)
                return formatted_response
            except Exception as e:
                error_msg = f"Proxy request failed for model '{request.model}' to endpoint '{server.endpoint_url}': {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=502, detail=f"Proxy request failed: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error in chat completion for model '{request.model}': {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)