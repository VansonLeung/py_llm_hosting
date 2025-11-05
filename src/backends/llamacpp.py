"""
Llama.cpp backend implementation.

This backend uses llama-cpp-python for efficient CPU/GPU inference with GGUF models.
"""

import asyncio
from typing import Dict, Any, List, AsyncIterator, Optional
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

from .base import LLMBackend, ModelConfig, GenerationParams, BackendType
from ..lib.logging import logger


class LlamaCppBackend(LLMBackend):
    """
    Backend implementation using llama-cpp-python.
    
    Supports GGUF format models with efficient CPU/GPU inference.
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize llama-cpp backend."""
        super().__init__(model_config)
        self.llama: Optional[Llama] = None
    
    async def load_model(self) -> None:
        """Load GGUF model using llama.cpp."""
        try:
            logger.info(f"Loading model with llama-cpp: {self.model_config.model_path}")
            
            # Check if it's a HuggingFace model or local path
            model_path = self.model_config.model_path
            
            if not os.path.exists(model_path):
                # Try to download from HuggingFace
                logger.info(f"Downloading model from HuggingFace: {model_path}")
                # Assume format: "repo_id/filename.gguf" or just "repo_id"
                if "/" in model_path and ".gguf" in model_path:
                    parts = model_path.rsplit("/", 1)
                    repo_id = parts[0]
                    filename = parts[1]
                else:
                    repo_id = model_path
                    # Try common GGUF filenames
                    filename = "ggml-model-q4_k_m.gguf"
                
                model_path = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename
                )
                logger.info(f"Model downloaded to: {model_path}")
            
            # Load model in a thread to avoid blocking
            self.llama = await asyncio.to_thread(
                Llama,
                model_path=model_path,
                n_ctx=self.model_config.n_ctx or 2048,
                n_gpu_layers=self.model_config.n_gpu_layers or 0,
                verbose=False
            )
            
            self._loaded = True
            logger.info("Model loaded successfully with llama-cpp")
            
        except Exception as e:
            logger.error(f"Failed to load model with llama-cpp: {e}")
            raise
    
    async def unload_model(self) -> None:
        """Unload model and free memory."""
        try:
            if self.llama:
                # llama-cpp-python handles cleanup in destructor
                del self.llama
                self.llama = None
                self._loaded = False
                logger.info("Model unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        params: GenerationParams
    ) -> str:
        """Generate text from prompt."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run generation in thread pool
            result = await asyncio.to_thread(
                self.llama,
                prompt,
                max_tokens=params.max_tokens or 512,
                temperature=params.temperature or 0.7,
                top_p=params.top_p or 0.9,
                top_k=params.top_k or 40,
                repeat_penalty=params.repetition_penalty or 1.1,
                stop=params.stop or []
            )
            
            return result["choices"][0]["text"]
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        params: GenerationParams
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Create generator
            stream = self.llama(
                prompt,
                max_tokens=params.max_tokens or 512,
                temperature=params.temperature or 0.7,
                top_p=params.top_p or 0.9,
                top_k=params.top_k or 40,
                repeat_penalty=params.repetition_penalty or 1.1,
                stop=params.stop or [],
                stream=True
            )
            
            # Yield chunks
            for chunk in stream:
                text = chunk["choices"][0]["text"]
                if text:
                    yield text
                    await asyncio.sleep(0)  # Allow other tasks to run
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        params: GenerationParams
    ) -> Dict[str, Any]:
        """Generate chat completion."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Format messages as prompt
            prompt = self._format_chat_prompt(messages)
            
            # Generate response
            if params.stream:
                # For streaming, return a different structure
                raise NotImplementedError("Streaming chat not yet implemented")
            else:
                text = await self.generate(prompt, params)
                
                # Format as OpenAI-compatible response
                import time
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_config.model_path,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text.strip()
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,  # TODO: Calculate actual tokens
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def get_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings (if model supports it)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            embeddings = []
            for text in texts:
                # llama-cpp-python can generate embeddings
                result = await asyncio.to_thread(
                    self.llama.embed,
                    text
                )
                embeddings.append(result)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        return self.is_loaded and self.llama is not None
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt.
        
        Uses a simple template. Can be customized for different models.
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)