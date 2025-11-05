"""
MLX-VLM backend implementation.

Apple Silicon optimized vision-language model inference using MLX.
Supports multimodal models that can process both text and images.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.lib.logging import logger
import asyncio


class MLXVLMBackend(ModelBackend):
    """MLX-VLM backend for vision-language models on Apple Silicon."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model = None
        self.processor = None
        self.max_tokens_default = kwargs.get("max_tokens", 512)

    async def load_model(self) -> None:
        """Load the vision-language model using MLX-VLM."""
        try:
            from mlx_vlm import load
            
            logger.info(f"Loading vision-language model from {self.model_path} with MLX-VLM...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: load(self.model_path)
            )
            
            self.model, self.processor = result
            
            self.loaded = True
            logger.info(f"Vision-language model loaded successfully from {self.model_path}")
            
        except ImportError:
            raise RuntimeError(
                "MLX-VLM not installed or not on Apple Silicon. "
                "Install with: pip install mlx-vlm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model:
            self.model = None
            self.processor = None
            self.loaded = False
            logger.info("Vision-language model unloaded")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from prompt (text-only, no image)."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        from mlx_vlm import generate as vlm_generate
        
        max_tokens = max_tokens or self.max_tokens_default
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: vlm_generate(
                self.model,
                self.processor,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )
        )
        
        return {
            "text": response,
            "usage": {
                "prompt_tokens": len(prompt.split()),  # Approximate
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }
        }

    async def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from prompt and image."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        from mlx_vlm import generate as vlm_generate
        from PIL import Image
        
        max_tokens = max_tokens or self.max_tokens_default
        
        # Load image
        image = Image.open(image_path)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: vlm_generate(
                self.model,
                self.processor,
                image,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )
        )
        
        return {
            "text": response,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }
        }

    async def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion (supports images in messages)."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        from mlx_vlm import generate as vlm_generate
        from PIL import Image
        import time
        
        # Extract text and images from messages
        text_parts = []
        image = None
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Check if content is a list (multimodal)
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(f"{msg['role']}: {part['text']}")
                    elif part.get("type") == "image_url":
                        # Load image from URL or path
                        image_url = part["image_url"]["url"]
                        if image_url.startswith("data:"):
                            # Base64 encoded image
                            import base64
                            import io
                            image_data = base64.b64decode(image_url.split(",")[1])
                            image = Image.open(io.BytesIO(image_data))
                        else:
                            # File path or URL
                            image = Image.open(image_url)
            else:
                text_parts.append(f"{msg['role']}: {content}")
        
        prompt = "\n".join(text_parts)
        max_tokens = max_tokens or self.max_tokens_default
        
        loop = asyncio.get_event_loop()
        if image:
            response = await loop.run_in_executor(
                None,
                lambda: vlm_generate(
                    self.model,
                    self.processor,
                    image,
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    verbose=False
                )
            )
        else:
            response = await loop.run_in_executor(
                None,
                lambda: vlm_generate(
                    self.model,
                    self.processor,
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    verbose=False
                )
            )
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }
        }

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings (not supported)."""
        raise NotImplementedError(
            "MLX-VLM backend does not support embeddings. "
            "Use a dedicated embedding model or llama.cpp backend."
        )

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        return capability in [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.VISION
        ]

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "mlx_vlm",
            "model_path": self.model_path,
            "loaded": self.loaded,
            "config": {
                "max_tokens": self.max_tokens_default
            },
            "capabilities": [
                ModelCapability.TEXT_GENERATION.value,
                ModelCapability.VISION.value
            ],
            "platform": "Apple Silicon (M-series)",
            "supports_images": True
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.MLX_VLM, MLXVLMBackend)