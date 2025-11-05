"""
Backend initialization module.

This module registers all available backends with the factory.
Import this module to ensure backends are registered before use.
"""

from src.models.backend import ModelBackendFactory, ModelBackendType


def register_all_backends():
    """Register all available backends with the factory."""
    
    # Register llama-cpp backend
    try:
        from src.backends.llamacpp_backend import LlamaCppBackend
        ModelBackendFactory.register_backend(ModelBackendType.LLAMACPP, LlamaCppBackend)
    except ImportError as e:
        pass  # Backend dependencies not installed
    
    # Register vLLM backend
    try:
        from src.backends.vllm_backend import VLLMBackend
        ModelBackendFactory.register_backend(ModelBackendType.VLLM, VLLMBackend)
    except ImportError:
        pass  # Backend dependencies not installed
    
    # Register MLX backend
    try:
        from src.backends.mlx_backend import MLXBackend
        ModelBackendFactory.register_backend(ModelBackendType.MLX, MLXBackend)
    except ImportError:
        pass  # Backend dependencies not installed
    
    # Register MLX-VLM backend
    try:
        from src.backends.mlx_vlm_backend import MLXVLMBackend
        ModelBackendFactory.register_backend(ModelBackendType.MLX_VLM, MLXVLMBackend)
    except ImportError:
        pass  # Backend dependencies not installed
    
    # Register Reranker backend
    try:
        from src.backends.reranker_backend import RerankerBackend
        ModelBackendFactory.register_backend(ModelBackendType.RERANKER, RerankerBackend)
    except ImportError:
        pass  # Backend dependencies not installed
    
    # Register Sentence Transformers backend
    try:
        from src.backends.sentence_transformers_backend import SentenceTransformersBackend
        ModelBackendFactory.register_backend(ModelBackendType.SENTENCE_TRANSFORMERS, SentenceTransformersBackend)
    except ImportError:
        pass  # Backend dependencies not installed


# Auto-register when module is imported
register_all_backends()
