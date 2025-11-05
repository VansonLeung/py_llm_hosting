"""
MLX backend implementation.

Apple Silicon optimized inference using MLX framework.
Supports text generation models on Mac with Apple Silicon.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.lib.logging import logger
import asyncio


class MLXBackend(ModelBackend):
    """MLX backend for Apple Silicon optimized inference."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model = None
        self.tokenizer = None
        self.max_tokens_default = kwargs.get("max_tokens", 512)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    async def load_model(self) -> None:
        """Load the model using MLX."""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate
            
            logger.info(f"Loading model from {self.model_path} with MLX...")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: load(
                    self.model_path,
                    tokenizer_config={
                        "trust_remote_code": self.trust_remote_code
                    }
                )
            )
            
            self.model, self.tokenizer = result
            
            self.loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except ImportError:
            raise RuntimeError(
                "MLX not installed or not on Apple Silicon. "
                "Install with: pip install mlx mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model:
            self.model = None
            self.tokenizer = None
            self.loaded = False
            logger.info("Model unloaded")

    def _generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repetition_penalty: float,
        **extra_kwargs  # Catch any extra kwargs
    ):
        """Internal generate method with proper sampler and logits processors."""
        from mlx_lm import generate as mlx_lm_generate, sample_utils
        
        if extra_kwargs:
            logger.warning(f"_generate received unexpected kwargs: {extra_kwargs}")
        
        logger.info(f"_generate called with: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}, rep_penalty={repetition_penalty}")
        
        sampler = sample_utils.make_sampler(temp=temperature, top_p=top_p)
        logits_processors = sample_utils.make_logits_processors(
            repetition_penalty=repetition_penalty
        )
        
        logger.info(f"Created sampler and logits_processors")
        
        # Only pass the parameters that mlx_lm_generate() expects
        result = mlx_lm_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            sampler=sampler,
            logits_processors=logits_processors,
            max_tokens=max_tokens
            # DO NOT pass temperature, top_p, etc. as they are handled by sampler
        )
        
        logger.info(f"Generated result: {result[:50] if len(result) > 50 else result}")
        
        return result

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from prompt."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        max_tokens = max_tokens or self.max_tokens_default
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        
        loop = asyncio.get_event_loop()
        
        # Use _generate with proper parameters
        response = await loop.run_in_executor(
            None,
            lambda: self._generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty
            )
        )
        
        # Count tokens (approximate)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(response))
        
        return {
            "text": response,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    async def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion."""
        logger.info(f"generate_chat called with: stream={stream}, temperature={temperature}, kwargs={kwargs}")
        
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        import time
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        max_tokens = max_tokens or self.max_tokens_default
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        
        # Handle streaming
        if stream:
            async def stream_wrapper():
                loop = asyncio.get_event_loop()
                
                # MLX generate function - run in executor
                def generate_sync():
                    return self._generate(
                        prompt=prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        repetition_penalty=repetition_penalty
                    )
                
                full_response = await loop.run_in_executor(None, generate_sync)
                
                # Simulate streaming by yielding words
                words = full_response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield " " + word
                    await asyncio.sleep(0.01)  # Small delay for visual effect
            
            return stream_wrapper()
        
        # Non-streaming
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty
            )
        )
        
        # Count tokens
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(response))
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings using MLX."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            import mlx.core as mx
            import time
            
            logger.info(f"Generating embeddings for {len(texts)} texts with MLX")
            
            # Check if model supports embeddings
            if not (hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens')):
                raise RuntimeError(
                    "MLX model does not have embed_tokens. "
                    "This model may not support embeddings."
                )
            
            # Run embedding generation in executor
            loop = asyncio.get_event_loop()
            
            def embed_sync():
                embeddings = []
                total_tokens = 0
                
                for text in texts:
                    # Tokenize the input
                    input_ids = self.tokenizer.encode(text)
                    total_tokens += len(input_ids)
                    
                    # Convert to MLX array
                    input_ids_mx = mx.array([input_ids])
                    
                    # Get token embeddings from the model
                    token_embeddings = self.model.model.embed_tokens(input_ids_mx)
                    
                    # Mean pooling across sequence dimension
                    # token_embeddings shape: (1, seq_len, hidden_dim)
                    mean_embedding = mx.mean(token_embeddings[0], axis=0)
                    
                    # Normalize the embedding
                    norm = mx.sqrt(mx.sum(mean_embedding * mean_embedding))
                    normalized_embedding = mean_embedding / norm
                    
                    # Convert to list
                    embedding_list = normalized_embedding.tolist()
                    embeddings.append(embedding_list)
                
                return embeddings, total_tokens
            
            embeddings, total_tokens = await loop.run_in_executor(None, embed_sync)
            
            # Format as OpenAI-compatible response
            data = []
            for i, embedding in enumerate(embeddings):
                data.append({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i
                })
            
            return {
                "object": "list",
                "data": data,
                "model": self.model_path,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"MLX embedding generation failed: {str(e)}")

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        max_tokens = max_tokens or self.max_tokens_default
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        
        # MLX supports streaming, but we'll implement a simple version
        # For true streaming, you'd need to use the stream parameter in generate
        result = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=False,
            **kwargs
        )
        yield result["text"]

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        # MLX supports both text generation and embeddings
        return capability in [ModelCapability.TEXT_GENERATION, ModelCapability.EMBEDDINGS]

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "mlx",
            "model_path": self.model_path,
            "loaded": self.loaded,
            "config": {
                "max_tokens": self.max_tokens_default,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "trust_remote_code": self.trust_remote_code
            },
            "capabilities": [
                ModelCapability.TEXT_GENERATION.value,
                ModelCapability.EMBEDDINGS.value
            ],
            "platform": "Apple Silicon (M-series)"
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.MLX, MLXBackend)