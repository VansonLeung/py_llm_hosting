"""Model downloader for fetching models from HuggingFace Hub."""
import logging
from pathlib import Path
from typing import Optional
import os

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download models from HuggingFace Hub."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize downloader.
        
        Args:
            cache_dir: Directory to cache downloaded models. Defaults to ~/.cache/py_llm_hosting/models
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "py_llm_hosting" / "models"
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def download_model(
        self,
        model_id: str,
        filename: Optional[str] = None,
        force: bool = False
    ) -> Path:
        """Download a model from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
            filename: Specific file to download (for GGUF models). If None, downloads entire repo.
            force: Force re-download even if cached
            
        Returns:
            Path to the downloaded model
            
        Raises:
            ImportError: If huggingface_hub is not installed
            RuntimeError: If download fails
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading models. "
                "Install with: pip install huggingface-hub"
            )
        
        try:
            if filename:
                # Download specific file (e.g., GGUF)
                logger.info(f"Downloading {filename} from {model_id}")
                path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    cache_dir=str(self.cache_dir),
                    force_download=force
                )
                logger.info(f"Downloaded to {path}")
                return Path(path)
            else:
                # Download entire repo (for transformer models)
                logger.info(f"Downloading model repository {model_id}")
                path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(self.cache_dir),
                    force_download=force
                )
                logger.info(f"Downloaded to {path}")
                return Path(path)
                
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise RuntimeError(f"Failed to download model: {e}") from e
    
    def get_cached_path(self, model_id: str, filename: Optional[str] = None) -> Optional[Path]:
        """Get path to a cached model without downloading.
        
        Args:
            model_id: HuggingFace model ID
            filename: Specific file name (for GGUF models)
            
        Returns:
            Path to cached model or None if not cached
        """
        # This is a simplified check - actual HF cache structure is more complex
        # In practice, you'd use huggingface_hub's cache utilities
        if filename:
            # For GGUF files, check if file exists in cache
            # Note: This is simplified - real HF cache uses hashes
            potential_path = self.cache_dir / "models--" / model_id.replace("/", "--") / filename
            if potential_path.exists():
                return potential_path
        else:
            # For full repos
            potential_path = self.cache_dir / "models--" / model_id.replace("/", "--")
            if potential_path.exists():
                return potential_path
        
        return None
    
    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """Clear cached models.
        
        Args:
            model_id: Specific model to clear. If None, clears all cached models.
        """
        if model_id:
            # Clear specific model
            model_dir = self.cache_dir / "models--" / model_id.replace("/", "--")
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Cleared cache for {model_id}")
        else:
            # Clear all
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all model cache")


# Global instance
_downloader: Optional[ModelDownloader] = None


def get_downloader() -> ModelDownloader:
    """Get the global downloader instance."""
    global _downloader
    if _downloader is None:
        _downloader = ModelDownloader()
    return _downloader
