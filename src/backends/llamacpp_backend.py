"""
Llama.cpp backend implementation.

Supports GGUF models with CPU and GPU acceleration.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.libs.logging import logger
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
        self.embedding = kwargs.get("embedding", True)  # Enable embeddings by default
        self.resolved_model_path = None

    def _resolve_model_path(self, model_path: str) -> str:
        """
        Resolve model path from various formats:
        - Local file path: /path/to/model.gguf
        - HuggingFace model ID: owner/repo-name
        - HuggingFace with filename: owner/repo-name:filename.gguf
        
        Returns the actual file path to the model.
        """
        import os
        from pathlib import Path
        
        # If it's a local path and exists, use it directly
        if os.path.exists(model_path):
            logger.info(f"Using local model path: {model_path}")
            return model_path
        
        # Check if it's a HuggingFace model ID (contains /)
        if '/' in model_path and not os.path.exists(model_path):
            logger.info(f"Detected HuggingFace model ID: {model_path}")
            
            # Parse model_id and optional filename
            if ':' in model_path:
                model_id, filename = model_path.split(':', 1)
            else:
                model_id = model_path
                filename = None
            
            # Check HuggingFace cache first
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_cache_name = f"models--{model_id.replace('/', '--')}"
            model_cache_path = cache_dir / model_cache_name
            
            if model_cache_path.exists():
                logger.info(f"Found model in HuggingFace cache: {model_cache_path}")
                
                # Look for the model file in snapshots
                snapshots_dir = model_cache_path / "snapshots"
                if snapshots_dir.exists():
                    # Get the latest snapshot
                    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshots:
                        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                        
                        # If filename specified, look for it
                        if filename:
                            model_file = latest_snapshot / filename
                            if model_file.exists():
                                logger.info(f"Using cached model file: {model_file}")
                                return str(model_file)
                        else:
                            # Find any .gguf file
                            gguf_files = list(latest_snapshot.glob("*.gguf"))
                            if gguf_files:
                                model_file = gguf_files[0]
                                logger.info(f"Using cached model file: {model_file}")
                                return str(model_file)
            
            # Not in cache, try to download
            logger.info(f"Model not in cache, attempting to download from HuggingFace...")
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
                
                # If no filename specified, find the first .gguf file
                if not filename:
                    logger.info("No filename specified, searching for .gguf files...")
                    try:
                        files = list_repo_files(model_id)
                        gguf_files = [f for f in files if f.endswith('.gguf')]
                        if gguf_files:
                            filename = gguf_files[0]
                            logger.info(f"Found GGUF file: {filename}")
                        else:
                            raise RuntimeError(f"No .gguf files found in {model_id}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to list files in {model_id}: {e}")
                
                # Download the model
                logger.info(f"Downloading {filename} from {model_id}...")
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    cache_dir=str(cache_dir)
                )
                logger.info(f"Downloaded model to: {downloaded_path}")
                return downloaded_path
                
            except ImportError:
                raise RuntimeError(
                    "huggingface_hub not installed. "
                    "Install with: pip install huggingface-hub"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download model from HuggingFace: {e}")
        
        # If we get here, the path doesn't exist
        raise RuntimeError(f"Model path does not exist: {model_path}")

    async def load_model(self) -> None:
        """Load the model using llama-cpp-python."""
        try:
            from llama_cpp import Llama

            # Resolve model path (handles HuggingFace IDs, local paths, etc.)
            try:
                self.resolved_model_path = self._resolve_model_path(self.model_path)
            except RuntimeError as e:
                error_msg = f"Failed to resolve model path '{self.model_path}': {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(f"Loading model from {self.resolved_model_path} with llama.cpp...")

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.llm = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=self.resolved_model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    n_threads=self.n_threads,
                    verbose=self.verbose,
                    embedding=self.embedding  # Use configurable embedding setting
                )
            )

            self.loaded = True
            logger.info(f"Model loaded successfully from {self.resolved_model_path}")

        except ImportError as e:
            error_msg = f"llama-cpp-python not installed. Install with: pip install llama-cpp-python. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except FileNotFoundError as e:
            error_msg = f"Model file not found at {self.resolved_model_path or self.model_path}. Check the path and ensure the model exists. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except ValueError as e:
            error_msg = f"Invalid llama.cpp configuration for model {self.model_path}. Check n_ctx ({self.n_ctx}), n_gpu_layers ({self.n_gpu_layers}), n_threads ({self.n_threads}). Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                error_msg = f"Out of memory while loading model {self.model_path}. Try reducing n_ctx (current: {self.n_ctx}) or n_gpu_layers (current: {self.n_gpu_layers}). Error: {e}"
            else:
                error_msg = f"Runtime error loading model {self.model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading model {self.model_path} with llama.cpp backend. Model path: {self.model_path}, Config: n_ctx={self.n_ctx}, n_gpu_layers={self.n_gpu_layers}, n_threads={self.n_threads}. Error type: {type(e).__name__}, Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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
            error_msg = f"Model {self.model_path} is not loaded. Please start the server first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
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
        except ValueError as e:
            error_msg = f"Invalid generation parameters for model {self.model_path}. Check max_tokens ({max_tokens}), temperature ({temperature}), top_p ({top_p}). Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                error_msg = f"Out of memory during generation with model {self.model_path}. Try reducing max_tokens (current: {max_tokens}) or n_ctx (current: {self.n_ctx}). Error: {e}"
            else:
                error_msg = f"Runtime error during generation with model {self.model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during text generation with model {self.model_path}. Prompt length: {len(prompt)}, max_tokens: {max_tokens}. Error type: {type(e).__name__}, Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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
            error_msg = f"Model {self.model_path} is not loaded. Please start the server first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            max_tokens = max_tokens or 512

            # Handle streaming
            if stream:
                async def stream_wrapper():
                    try:
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
                    except Exception as e:
                        error_msg = f"Error during streaming chat generation with model {self.model_path}. Error type: {type(e).__name__}, Error: {e}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

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
        except ValueError as e:
            error_msg = f"Invalid chat generation parameters for model {self.model_path}. Check max_tokens ({max_tokens}), temperature ({temperature}). Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                error_msg = f"Out of memory during chat generation with model {self.model_path}. Try reducing max_tokens (current: {max_tokens}) or n_ctx (current: {self.n_ctx}). Error: {e}"
            else:
                error_msg = f"Runtime error during chat generation with model {self.model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during chat generation with model {self.model_path}. Messages: {len(messages)} message(s), max_tokens: {max_tokens}, temperature: {temperature}. Error type: {type(e).__name__}, Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings."""
        if not self.loaded:
            error_msg = f"Model {self.model_path} is not loaded. Please start the server first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
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
        except ValueError as e:
            error_msg = f"Invalid embedding parameters for model {self.model_path}. Check input texts format. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                error_msg = f"Out of memory during embedding generation with model {self.model_path}. Try processing fewer texts at once. Error: {e}"
            else:
                error_msg = f"Runtime error during embedding generation with model {self.model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during embedding generation with model {self.model_path}. Number of texts: {len(texts)}. Error type: {type(e).__name__}, Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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
                "n_threads": self.n_threads,
                "embedding": self.embedding
            },
            "capabilities": [
                ModelCapability.TEXT_GENERATION.value,
                ModelCapability.EMBEDDINGS.value
            ]
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.LLAMACPP, LlamaCppBackend)