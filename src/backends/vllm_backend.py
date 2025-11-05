"""
vLLM backend implementation.

High-performance inference engine optimized for throughput and GPU utilization.
Supports HuggingFace Transformers models.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.lib.logging import logger
import asyncio


class VLLMBackend(ModelBackend):
    """vLLM backend for high-performance inference."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.llm = None
        self.tokenizer = None
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.max_model_len = kwargs.get("max_model_len", None)
        self.dtype = kwargs.get("dtype", "auto")
        self.trust_remote_code = kwargs.get("trust_remote_code", False)

    async def load_model(self) -> None:
        """Load the model using vLLM."""
        try:
            from vllm import LLM
            from vllm import SamplingParams
            
            logger.info(f"Loading model from {self.model_path} with vLLM...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.llm = await loop.run_in_executor(
                None,
                lambda: LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    dtype=self.dtype,
                    trust_remote_code=self.trust_remote_code,
                    download_dir=self.config.get("download_dir")
                )
            )
            
            # Load tokenizer for chat template support
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            self.loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except ImportError as e:
            raise RuntimeError(
                f"vLLM not installed. Install with: pip install vllm transformers. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.llm:
            # vLLM doesn't have explicit unload, rely on garbage collection
            self.llm = None
            self.tokenizer = None
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
            raise RuntimeError("Model not loaded")

        from vllm import SamplingParams
        
        max_tokens = max_tokens or 512
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.llm.generate([prompt], sampling_params)
        )
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            }
        }

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
            raise RuntimeError("Model not loaded")

        from vllm import SamplingParams
        import time
        
        # Apply chat template
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        max_tokens = max_tokens or 512
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.llm.generate([prompt], sampling_params)
        )
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": output.outputs[0].finish_reason
            }],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            }
        }

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings (not supported by vLLM for generation models)."""
        raise NotImplementedError(
            "vLLM backend does not support embeddings for generation models. "
            "Use a dedicated embedding model or llama.cpp backend."
        )

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        return capability == ModelCapability.TEXT_GENERATION

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "vllm",
            "model_path": self.model_path,
            "loaded": self.loaded,
            "config": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "dtype": self.dtype
            },
            "capabilities": [
                ModelCapability.TEXT_GENERATION.value
            ]
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.VLLM, VLLMBackend)