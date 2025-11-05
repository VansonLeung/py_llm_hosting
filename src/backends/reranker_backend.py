"""
Reranker backend implementation.

Uses sentence-transformers cross-encoders for document reranking.
Optimized for BGE reranker models and similar architectures.
"""

from typing import Dict, Any, List, Optional, Tuple
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.lib.logging import logger
import asyncio


class RerankerBackend(ModelBackend):
    """Backend for reranking/cross-encoder models."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model = None
        self.max_length = kwargs.get("max_length", 512)
        self.device = kwargs.get("device", "cpu")

    async def load_model(self) -> None:
        """Load the reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading reranker model from {self.model_path}...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(
                    self.model_path,
                    max_length=self.max_length,
                    device=self.device
                )
            )
            
            self.loaded = True
            logger.info(f"Reranker model loaded successfully")
            
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model:
            self.model = None
            self.loaded = False
            logger.info("Reranker model unloaded")

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top results to return (default: all)
            
        Returns:
            Dict with reranked results and scores
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        if not documents:
            return {
                "results": [],
                "usage": {
                    "total_tokens": 0
                }
            }
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Run prediction in executor
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict(pairs)
        )
        
        # Convert to list if numpy array
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        
        # Create results with indices and scores
        results = [
            {
                "index": idx,
                "relevance_score": float(score),
                "document": doc
            }
            for idx, (score, doc) in enumerate(zip(scores, documents))
        ]
        
        # Sort by score descending
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Apply top_n limit if specified
        if top_n is not None and top_n > 0:
            results = results[:top_n]
        
        # Estimate token usage (rough approximation)
        total_text = query + " ".join(documents)
        estimated_tokens = len(total_text.split())
        
        return {
            "results": results,
            "usage": {
                "total_tokens": estimated_tokens
            }
        }

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Not supported for reranker models."""
        raise NotImplementedError("Reranker models do not support text generation")

    async def generate_chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Not supported for reranker models."""
        raise NotImplementedError("Reranker models do not support chat")

    async def embed(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Not supported for reranker models."""
        raise NotImplementedError("Reranker models do not support embeddings")

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        return capability == ModelCapability.RERANK

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "reranker",
            "model_path": self.model_path,
            "loaded": self.loaded,
            "config": {
                "max_length": self.max_length,
                "device": self.device
            },
            "capabilities": [ModelCapability.RERANK.value],
            "framework": "sentence-transformers"
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.RERANKER, RerankerBackend)
