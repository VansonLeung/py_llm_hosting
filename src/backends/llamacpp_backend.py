"""
Llama.cpp backend implementation.

Supports GGUF models with CPU and GPU acceleration.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.lib.logging import logger
import asyncio


class LlamaCppBackend(ModelBackend):
    """Llama.cpp backend using llama-cpp-python."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.llm = None
        self.n_ctx = kwargs.get("n_ctx", 2048)
        self.n_gpu_layers = kwargs.get("n_gpu_layers", 0)
        self.n_threads = kwargs.get("n_threads", None)
        self.verbose = kwargs.get("verbose", False)

    async def load_model(self) -> None:
        """Load the model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading model from {self.model_path} with llama.cpp...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.llm = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    n_threads=self.n_threads,
                    verbose=self.verbose,
                    embedding=True  # Enable embeddings
                )
            )
            
            self.loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.llm:
            self.llm = None
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

        max_tokens = max_tokens or 512
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stream=False,
                **kwargs
            )
        )
        
        return {
            "text": result["choices"][0]["text"],
            "usage": {
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"]
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

        max_tokens = max_tokens or 512
        
        # Handle streaming
        if stream:
            async def stream_wrapper():
                loop = asyncio.get_event_loop()
                # Run streaming in executor
                def stream_gen():
                    return self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                        **kwargs
                    )
                
                stream_iter = await loop.run_in_executor(None, stream_gen)
                for chunk in stream_iter:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    await asyncio.sleep(0)  # Allow other tasks to run
            
            return stream_wrapper()
        
        # Non-streaming
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **kwargs
            )
        )
        
        return {
            "id": result["id"],
            "object": "chat.completion",
            "created": result["created"],
            "model": self.model_path,
            "choices": result["choices"],
            "usage": result["usage"]
        }

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        embeddings = []
        
        for text in texts:
            embedding = await loop.run_in_executor(
                None,
                lambda: self.llm.embed(text)
            )
            embeddings.append(embedding)
        
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": emb,
                    "index": idx
                }
                for idx, emb in enumerate(embeddings)
            ],
            "model": self.model_path,
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts)
            }
        }

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

        max_tokens = max_tokens or 512
        
        # Create streaming generator in thread
        for chunk in self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        ):
            text = chunk["choices"][0]["text"]
            yield text
            await asyncio.sleep(0)  # Allow other tasks to run

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        return capability in [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.EMBEDDINGS
        ]

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "llamacpp",
            "model_path": self.model_path,
            "loaded": self.loaded,
            "config": {
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "n_threads": self.n_threads
            },
            "capabilities": [
                ModelCapability.TEXT_GENERATION.value,
                ModelCapability.EMBEDDINGS.value
            ]
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.LLAMACPP, LlamaCppBackend)