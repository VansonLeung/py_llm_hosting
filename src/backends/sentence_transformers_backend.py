"""
Sentence Transformers backend implementation.

Supports embedding models from sentence-transformers library.
Ideal for semantic search and similarity tasks.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.libs.logging import logger
import asyncio


class SentenceTransformersBackend(ModelBackend):
    """Sentence Transformers backend for embedding models."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model = None
        self.device = kwargs.get("device", None)  # Auto-detect if None
        self.normalize_embeddings = kwargs.get("normalize_embeddings", True)

    async def load_model(self) -> None:
        """Load the model using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading model from {self.model_path} with sentence-transformers...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def load_sync():
                return SentenceTransformer(
                    self.model_path,
                    device=self.device
                )
            
            self.model = await loop.run_in_executor(None, load_sync)
            
            self.loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Device: {self.model.device}")
            logger.info(f"Max sequence length: {self.model.max_seq_length}")
            
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model:
            self.model = None
            self.loaded = False
            logger.info("Model unloaded")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from prompt (not supported for embedding models)."""
        raise NotImplementedError(
            "Sentence Transformers backend only supports embeddings, not text generation."
        )

    async def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion (not supported for embedding models)."""
        raise NotImplementedError(
            "Sentence Transformers backend only supports embeddings, not chat."
        )

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings using sentence-transformers."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            import time
            
            logger.info(f"Generating embeddings for {len(texts)} texts with sentence-transformers")
            
            # Run embedding generation in executor
            loop = asyncio.get_event_loop()
            
            def embed_sync():
                # Generate embeddings
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True
                )
                return embeddings
            
            embeddings_array = await loop.run_in_executor(None, embed_sync)
            
            # Convert to list format
            embeddings_list = embeddings_array.tolist()
            
            # Format as OpenAI-compatible response
            data = []
            for i, embedding in enumerate(embeddings_list):
                data.append({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i
                })
            
            # Estimate token count (rough approximation)
            total_tokens = sum(len(text.split()) for text in texts) * 2
            
            return {
                "object": "list",
                "data": data,
                "model": self.model_path,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Sentence Transformers embedding generation failed: {str(e)}")

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generated text (not supported for embedding models)."""
        raise NotImplementedError(
            "Sentence Transformers backend only supports embeddings, not text generation."
        )
        yield  # Make it a generator

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        return capability == ModelCapability.EMBEDDINGS

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        info = {
            "backend": "sentence-transformers",
            "model_path": self.model_path,
            "loaded": self.loaded,
            "config": {
                "normalize_embeddings": self.normalize_embeddings,
                "device": self.device
            },
            "capabilities": [
                ModelCapability.EMBEDDINGS.value
            ]
        }
        
        if self.loaded and self.model:
            info["config"]["max_seq_length"] = self.model.max_seq_length
            info["config"]["embedding_dimension"] = self.model.get_sentence_embedding_dimension()
            info["config"]["device"] = str(self.model.device)
        
        return info


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.SENTENCE_TRANSFORMERS, SentenceTransformersBackend)
