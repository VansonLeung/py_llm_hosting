"""
Model Manager Service.

Manages loading, unloading, and accessing LLM models with different backends.
"""

from typing import Dict, Optional, Any
import asyncio
from ..models.backend import ModelBackend, ModelBackendFactory, ModelBackendType
from ..models.server import LLMServer, ServerMode, ServerStatus
from ..lib.logging import logger

# Import to register all backends
import src.backends


class ModelManager:
    """
    Manages multiple model instances.
    
    Provides centralized model management with lazy loading and caching.
    """
    
    def __init__(self, persistence=None):
        """Initialize model manager."""
        self._models: Dict[str, ModelBackend] = {}
        self._lock = asyncio.Lock()
        self._persistence = persistence
    
    async def load_model(self, server: LLMServer) -> ModelBackend:
        """
        Load a model for a server if not already loaded.
        
        Args:
            server: The LLMServer configuration
            
        Returns:
            The loaded backend instance
            
        Raises:
            ValueError: If server is not in self-hosted mode or missing required fields
            RuntimeError: If backend fails to load
        """
        if server.mode != ServerMode.SELF_HOSTED:
            raise ValueError(f"Server {server.name} is not in self-hosted mode")
        
        if not server.backend_type:
            raise ValueError(f"Server {server.name} missing backend_type")
        
        if not server.model_path:
            raise ValueError(f"Server {server.name} missing model_path")
        
        async with self._lock:
            # Check if already loaded
            if server.id in self._models:
                logger.warning(f"Model for server {server.name} already loaded")
                return self._models[server.id]
            
            logger.info(f"Loading model for server {server.name} with backend {server.backend_type}")
            
            try:
                # Convert backend_type string to enum
                backend_type_map = {
                    "llama-cpp": ModelBackendType.LLAMACPP,
                    "llamacpp": ModelBackendType.LLAMACPP,
                    "vllm": ModelBackendType.VLLM,
                    "mlx": ModelBackendType.MLX,
                    "mlx-vlm": ModelBackendType.MLX_VLM,
                    "reranker": ModelBackendType.RERANKER,
                    "sentence-transformers": ModelBackendType.SENTENCE_TRANSFORMERS,
                    "transformers": ModelBackendType.LLAMACPP,  # Use llamacpp as fallback
                }
                
                backend_type_enum = backend_type_map.get(server.backend_type.lower())
                if not backend_type_enum:
                    raise ValueError(f"Unknown backend type: {server.backend_type}")
                
                # Create backend instance
                backend = ModelBackendFactory.create_backend(
                    backend_type=backend_type_enum,
                    model_path=server.model_path,
                    **server.backend_config or {}
                )
                
                # Load the model (async call)
                await backend.load_model()
                
                # Store in cache
                self._models[server.id] = backend
                logger.info(f"Successfully loaded model for server {server.name}")
                
                # Update server status to ACTIVE
                if self._persistence:
                    try:
                        server.status = ServerStatus.ACTIVE
                        server.update_timestamp()
                        self._persistence.update_server(server)
                        logger.info(f"Updated server {server.name} status to ACTIVE")
                    except Exception as e:
                        logger.warning(f"Failed to update server status: {e}")
                
                return backend
                
            except Exception as e:
                logger.error(f"Failed to load model for server {server.name}: {e}")
                raise RuntimeError(f"Failed to load model: {e}") from e
    
    async def unload_model(self, server_id: str) -> None:
        """
        Unload a model and free resources.
        
        Args:
            server_id: Server identifier
            
        Raises:
            KeyError: If model is not loaded
        """
        async with self._lock:
            if server_id not in self._models:
                raise KeyError(f"Model {server_id} not found")
            
            logger.info(f"Unloading model: {server_id}")
            
            backend = self._models[server_id]
            try:
                # Async call
                await backend.unload_model()
            except Exception as e:
                logger.warning(f"Error unloading model for server {server_id}: {e}")
            finally:
                del self._models[server_id]
            
            logger.info(f"Model {server_id} unloaded successfully")
            
            # Update server status to INACTIVE
            if self._persistence:
                try:
                    servers = self._persistence.get_servers()
                    server = next((s for s in servers if s.id == server_id), None)
                    if server:
                        server.status = ServerStatus.INACTIVE
                        server.update_timestamp()
                        self._persistence.update_server(server)
                        logger.info(f"Updated server status to INACTIVE")
                except Exception as e:
                    logger.warning(f"Failed to update server status: {e}")
    
    def get_backend(self, server_id: str) -> Optional[ModelBackend]:
        """
        Get a loaded model backend.
        
        Args:
            server_id: Server identifier
            
        Returns:
            Model backend or None if not loaded
        """
        return self._models.get(server_id)
    
    def is_loaded(self, server_id: str) -> bool:
        """
        Check if a model is loaded.
        
        Args:
            server_id: Server identifier
            
        Returns:
            True if loaded, False otherwise
        """
        return server_id in self._models
    
    def list_loaded(self) -> Dict[str, Dict[str, Any]]:
        """
        List all loaded models with their info.
        
        Returns:
            Dict mapping server_id to backend info
        """
        result = {}
        for server_id, backend in self._models.items():
            result[server_id] = {
                "backend_type": type(backend).__name__,
                "capabilities": [cap.value for cap in backend.supported_capabilities]
            }
        return result
    
    async def unload_all(self) -> None:
        """Unload all models."""
        async with self._lock:
            server_ids = list(self._models.keys())
            for server_id in server_ids:
                try:
                    backend = self._models[server_id]
                    await backend.unload_model()
                except Exception as e:
                    logger.warning(f"Error unloading model {server_id}: {e}")
                finally:
                    if server_id in self._models:
                        del self._models[server_id]
            
            logger.info("All models unloaded")
    
    def set_persistence(self, persistence):
        """Set the persistence layer for status tracking."""
        self._persistence = persistence
        logger.info("Persistence layer configured for model_manager")


# Global model manager instance
model_manager = ModelManager()