"""
Transformers backend implementation.

This backend uses HuggingFace Transformers for flexible model loading
and inference with support for various model architectures.
"""

import asyncio
from typing import Dict, Any, List, AsyncIterator, Optional
import torch

from .base import LLMBackend, ModelConfig, GenerationParams, BackendType
from ..lib.logging import logger


class TransformersBackend(LLMBackend):
    """
    Backend implementation using HuggingFace Transformers.
    
    Supports a wide variety of model architectures with CPU/GPU inference.
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize transformers backend."""
        super().__init__(model_config)
        self.pipeline = None
        self.device = None
    
    async def load_model(self) -> None:
        """Load model using Transformers."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
                BitsAndBytesConfig
            )
            
            logger.info(f"Loading model with Transformers: {self.model_config.model_path}")
            
            # Determine device
            if self.model_config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.model_config.device
            
            logger.info(f"Using device: {self.device}")
            
            # Setup quantization config if needed
            quantization_config = None
            if self.model_config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.model_config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                self.model_config.model_path,
                trust_remote_code=self.model_config.trust_remote_code or False
            )
            
            # Load model
            logger.info("Loading model...")
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_config.model_path,
                "trust_remote_code": self.model_config.trust_remote_code or False,
                "device_map": self.device if self.device != "cpu" else None,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if self.model_config.max_memory:
                model_kwargs["max_memory"] = self.model_config.max_memory
            
            self.model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained,
                **model_kwargs
            )
            
            # Move to device if CPU
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device != "cpu" else -1
            )
            
            self._loaded = True
            logger.info("Model loaded successfully with Transformers")
            
        except Exception as e:
            logger.error(f"Failed to load model with Transformers: {e}")
            raise
    
    async def unload_model(self) -> None:
        """Unload model and free memory."""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            # Prepare generation config
            gen_kwargs = {
                "max_new_tokens": params.max_tokens or 512,
                "temperature": params.temperature or 0.7,
                "top_p": params.top_p or 0.9,
                "top_k": params.top_k or 40,
                "repetition_penalty": params.repetition_penalty or 1.1,
                "do_sample": True,
            }
            
            if params.stop:
                gen_kwargs["stop_sequence"] = params.stop
            
            # Generate in thread pool
            result = await asyncio.to_thread(
                self.pipeline,
                prompt,
                **gen_kwargs
            )
            
            # Extract generated text
            generated_text = result[0]["generated_text"]
            
            # Remove prompt from output if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
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
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            # Setup streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Prepare generation config
            gen_kwargs = {
                "max_new_tokens": params.max_tokens or 512,
                "temperature": params.temperature or 0.7,
                "top_p": params.top_p or 0.9,
                "top_k": params.top_k or 40,
                "repetition_penalty": params.repetition_penalty or 1.1,
                "do_sample": True,
                "streamer": streamer,
            }
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Start generation in thread
            generation_kwargs = {**inputs, **gen_kwargs}
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield generated tokens
            for text in streamer:
                if text:
                    yield text
                    await asyncio.sleep(0)
            
            thread.join()
            
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
            # Try to use chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback to simple formatting
                prompt = self._format_chat_prompt(messages)
            
            # Generate response
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
        """Generate embeddings."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # For embedding generation, we'd need a different model
            # This is a simplified version using last hidden state
            embeddings = []
            
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    # Use mean of last hidden state as embedding
                    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
                    embeddings.append(embedding.cpu().tolist())
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        return self.is_loaded and self.model is not None
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt."""
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