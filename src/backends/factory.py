"""
Factory for creating LLM backend instances.

This module provides a factory pattern for creating backend instances
based on configuration.
"""

from typing import Optional
from .base import LLMBackend, ModelConfig, BackendType
from .llamacpp_backend import LlamaCppBackend
from .transformers_backend import TransformersBackend
from ..libs.logging import logger


class BackendFactory:
    """Factory for creating LLM backend instances."""
    
    _backends = {
        BackendType.LLAMACPP: LlamaCppBackend,
        BackendType.TRANSFORMERS: TransformersBackend,
    }
    
    @classmethod
    def create_backend(cls, model_config: ModelConfig) -> LLMBackend:
        """
        Create a backend instance based on configuration.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Instantiated backend
            
        Raises:
            ValueError: If backend type is not supported
        """
        backend_class = cls._backends.get(model_config.backend)
        
        if not backend_class:
            raise ValueError(
                f"Unsupported backend: {model_config.backend}. "
                f"Supported backends: {list(cls._backends.keys())}"
            )
        
        logger.info(f"Creating backend: {model_config.backend}")
        return backend_class(model_config)
    
    @classmethod
    def register_backend(
        cls,
        backend_type: BackendType,
        backend_class: type
    ) -> None:
        """
        Register a new backend type.
        
        This allows external modules to add new backends.
        
        Args:
            backend_type: Backend type identifier
            backend_class: Backend class
        """
        if not issubclass(backend_class, LLMBackend):
            raise ValueError(
                f"Backend class must inherit from LLMBackend"
            )
        
        cls._backends[backend_type] = backend_class
        logger.info(f"Registered backend: {backend_type}")
    
    @classmethod
    def list_backends(cls) -> list:
        """
        List all registered backend types.
        
        Returns:
            List of backend types
        """
        return list(cls._backends.keys())