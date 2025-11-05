"""
Base interface for LLM backends.

This module defines the abstract base class that all LLM backends must implement.
This design follows the Strategy pattern for maximum flexibility and extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from enum import Enum
from pydantic import BaseModel


class BackendType(str, Enum):
    """Supported backend types."""
    LLAMACPP = "llama-cpp"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    PROXY = "proxy"  # For external servers


class ModelConfig(BaseModel):
    """Configuration for loading a model."""
    model_path: str  # HuggingFace model ID or local path
    backend: BackendType
    n_ctx: Optional[int] = 2048  # Context window size
    n_gpu_layers: Optional[int] = 0  # GPU layers (llama.cpp)
    device: Optional[str] = "auto"  # Device for transformers
    load_in_8bit: Optional[bool] = False
    load_in_4bit: Optional[bool] = False
    max_memory: Optional[Dict[str, str]] = None
    trust_remote_code: Optional[bool] = False
    additional_params: Optional[Dict[str, Any]] = None


class GenerationParams(BaseModel):
    """Parameters for text generation."""
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repetition_penalty: Optional[float] = 1.1
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    All backend implementations must inherit from this class and implement
    all abstract methods. This ensures a consistent interface across different
    backend types.
    """
    
    def __init__(self, model_config: ModelConfig):
        """
        Initialize the backend with configuration.
        
        Args:
            model_config: Configuration for the model
        """
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    @abstractmethod
    async def load_model(self) -> None:
        """
        Load the model into memory.
        
        This should handle:
        - Downloading model files if needed
        - Loading model weights
        - Initializing tokenizer
        - Setting up GPU if available
        
        Raises:
            Exception: If model loading fails
        """
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """
        Unload the model from memory.
        
        This should:
        - Free GPU memory
        - Clear model references
        - Clean up resources
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        params: GenerationParams
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            params: Generation parameters
            
        Returns:
            Generated text
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        params: GenerationParams
    ) -> AsyncIterator[str]:
        """
        Generate text from a prompt with streaming.
        
        Args:
            prompt: Input text prompt
            params: Generation parameters
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        params: GenerationParams
    ) -> Dict[str, Any]:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of chat messages with 'role' and 'content'
            params: Generation parameters
            
        Returns:
            OpenAI-compatible chat completion response
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    async def get_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the backend is healthy and ready.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return self.model_config.backend
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_config.model_path,
            "backend": self.backend_type.value,
            "loaded": self.is_loaded,
            "config": self.model_config.model_dump()
        }