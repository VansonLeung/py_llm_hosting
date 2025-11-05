"""
MLX backend implementation.

Apple Silicon optimized inference using MLX framework.
Supports text generation models on Mac with Apple Silicon.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.lib.logging import logger
import asyncio


class MLXBackend(ModelBackend):
    """MLX backend for Apple Silicon optimized inference."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model = None
        self.tokenizer = None
        self.max_tokens_default = kwargs.get("max_tokens", 512)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)

    async def load_model(self) -> None:
        """Load the model using MLX."""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate
            
            logger.info(f"Loading model from {self.model_path} with MLX...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: load(
                    self.model_path,
                    tokenizer_config={
                        "trust_remote_code": self.trust_remote_code
                    }
                )
            )
            
            self.model, self.tokenizer = result
            
            self.loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except ImportError:
            raise RuntimeError(
                "MLX not installed or not on Apple Silicon. "
                "Install with: pip install mlx mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model:
            self.model = None
            self.tokenizer = None
            self.loaded = False
            logger.info("Model unloaded")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from prompt."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        from mlx_lm import generate
        
        max_tokens = max_tokens or self.max_tokens_default
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                verbose=False
            )
        )
        
        # Count tokens (approximate)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(response))
        
        return {
            "text": response,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
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
        """Generate chat completion."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        from mlx_lm import generate
        import time
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        max_tokens = max_tokens or self.max_tokens_default
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )
        )
        
        # Count tokens
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(response))
        
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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings (not directly supported by mlx-lm)."""
        raise NotImplementedError(
            "MLX backend does not support embeddings with mlx-lm. "
            "Consider using a dedicated embedding model or llama.cpp backend."
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        from mlx_lm import generate
        
        max_tokens = max_tokens or self.max_tokens_default
        
        # MLX supports streaming, but we'll implement a simple version
        # For true streaming, you'd need to use the stream parameter in generate
        result = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs
        )
        yield result["text"]

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        return capability == ModelCapability.TEXT_GENERATION

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "mlx",
            "model_path": self.model_path,
            "loaded": self.loaded,
            "config": {
                "max_tokens": self.max_tokens_default,
                "trust_remote_code": self.trust_remote_code
            },
            "capabilities": [
                ModelCapability.TEXT_GENERATION.value
            ],
            "platform": "Apple Silicon (M-series)"
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.MLX, MLXBackend)