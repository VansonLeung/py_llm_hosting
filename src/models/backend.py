"""
Base interface for model backends.

This module defines the abstract interface that all model hosting backends must implement.
Supports extensibility for different inference engines (llama.cpp, vLLM, MLX, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from enum import Enum


class ModelBackendType(str, Enum):
    """Supported model backend types."""
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    MLX = "mlx"
    MLX_VLM = "mlx_vlm"
    PROXY = "proxy"  # External server proxy (existing functionality)


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    RERANK = "rerank"


class ModelBackend(ABC):
    """Abstract base class for model hosting backends."""

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the model backend.
        
        Args:
            model_path: Path to model (local path or HuggingFace model ID)
            **kwargs: Backend-specific configuration
        """
        self.model_path = model_path
        self.config = kwargs
        self.loaded = False

    @abstractmethod
    async def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Backend-specific parameters
            
        Returns:
            Dict containing generated text and metadata
        """
        pass

    @abstractmethod
    async def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate chat completion from messages.
        
        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Backend-specific parameters
            
        Returns:
            Dict containing generated response and metadata
        """
        pass

    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            **kwargs: Backend-specific parameters
            
        Returns:
            Dict containing embeddings and metadata
        """
        pass

    @abstractmethod
    def supports_capability(self, capability: ModelCapability) -> bool:
        """
        Check if backend supports a capability.
        
        Args:
            capability: Capability to check
            
        Returns:
            True if supported, False otherwise
        """
        pass

    @abstractmethod
    async def get_info(self) -> Dict[str, Any]:
        """
        Get backend and model information.
        
        Returns:
            Dict containing backend info, model details, etc.
        """
        pass

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generated text (optional, for backends that support it).
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Backend-specific parameters
            
        Yields:
            Generated text chunks
        """
        # Default implementation for non-streaming backends
        result = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs
        )
        yield result.get("text", "")


class ModelBackendFactory:
    """Factory for creating model backend instances."""

    _backends: Dict[ModelBackendType, type] = {}

    @classmethod
    def register_backend(cls, backend_type: ModelBackendType, backend_class: type):
        """
        Register a backend implementation.
        
        Args:
            backend_type: Type of backend
            backend_class: Backend class implementing ModelBackend
        """
        cls._backends[backend_type] = backend_class

    @classmethod
    def create_backend(
        cls,
        backend_type: ModelBackendType,
        model_path: str,
        **kwargs
    ) -> ModelBackend:
        """
        Create a backend instance.
        
        Args:
            backend_type: Type of backend to create
            model_path: Path to model
            **kwargs: Backend-specific configuration
            
        Returns:
            ModelBackend instance
            
        Raises:
            ValueError: If backend type is not registered
        """
        if backend_type not in cls._backends:
            raise ValueError(f"Backend type '{backend_type}' not registered")
        
        backend_class = cls._backends[backend_type]
        return backend_class(model_path=model_path, **kwargs)

    @classmethod
    def list_backends(cls) -> List[ModelBackendType]:
        """List all registered backend types."""
        return list(cls._backends.keys())