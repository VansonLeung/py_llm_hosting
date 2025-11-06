"""
MLX backend implementation.

Apple Silicon optimized inference using MLX framework.
Supports text generation models on Mac with Apple Silicon.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.libs.logging import logger
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
        self.is_thinking_model = kwargs.get("is_thinking_model", False)

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

        except ImportError as e:
            error_msg = f"MLX not installed or not on Apple Silicon. Install with: pip install mlx mlx-lm. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except FileNotFoundError as e:
            error_msg = f"Model not found at path: {self.model_path}. The model will be downloaded automatically on first use. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except ValueError as e:
            error_msg = f"Invalid MLX configuration for model {self.model_path}. Check trust_remote_code ({self.trust_remote_code}). Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            if "apple silicon" in str(e).lower() or "m1" in str(e).lower() or "m2" in str(e).lower() or "m3" in str(e).lower():
                error_msg = f"MLX requires Apple Silicon (M1/M2/M3) Mac. This appears to be running on non-Apple Silicon hardware. Error: {e}"
            elif "out of memory" in str(e).lower() or "memory" in str(e).lower():
                error_msg = f"Out of memory while loading model {self.model_path}. Try using a smaller model or reducing max_tokens. Error: {e}"
            else:
                error_msg = f"Runtime error loading model {self.model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading model {self.model_path} with MLX backend. Model path: {self.model_path}, Config: trust_remote_code={self.trust_remote_code}. Error type: {type(e).__name__}, Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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
        
        logger.info(f"Generated result: {f"{result[:50]}..." if len(result) > 50 else result}")
        
        return result

    async def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion with optional tools support."""
        logger.info(f"generate_chat called with: stream={stream}, temperature={temperature}, tools={tools}, kwargs={kwargs}")

        if not self.loaded:
            error_msg = f"Model {self.model_path} is not loaded. Please start the server first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            import time
            import json

            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # Try to apply with tools if supported
                try:
                    if tools:
                        logger.info("Applying chat template with tools")
                        prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tools=tools,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        logger.info("Applying chat template without tools")
                        prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                except TypeError:
                    # Fallback if tools not supported
                    logger.warning("Tokenizer doesn't support tools parameter")
                    prompt = self._format_prompt_with_tools(messages, tools)
            else:
                # Fallback: manual formatting
                logger.warning("Tokenizer has no apply_chat_template method")
                prompt = self._format_prompt_with_tools(messages, tools)

            max_tokens = max_tokens or self.max_tokens_default
            temperature = temperature if temperature is not None else self.temperature
            top_p = top_p if top_p is not None else self.top_p
            repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty

            # Handle streaming
            if stream:
                async def stream_wrapper():
                    try:
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

                        # Check for tool calls
                        tool_calls = self._extract_tool_calls(full_response) if tools else None
                        
                        logger.info(f"Streaming generation completed, response length: {len(full_response)}")
                        logger.info(f"Tool calls extracted: {tool_calls}")

                        if tool_calls:
                            # Yield tool calls as structured data
                            yield {
                                "tool_calls": tool_calls,
                                "content": None
                            }
                        else:
                            # Simulate streaming by yielding words
                            words = full_response.split()
                            for i, word in enumerate(words):
                                if i == 0:
                                    yield word
                                else:
                                    yield " " + word
                                await asyncio.sleep(0.01)  # Small delay for visual effect
                    except Exception as e:
                        error_msg = f"Error during streaming chat generation with model {self.model_path}. Error type: {type(e).__name__}, Error: {e}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

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

            # Parse tool calls if present
            tool_calls = self._extract_tool_calls(response) if tools else None

            # Extract thinking content if this is a thinking model
            thinking_content = None
            final_response = response
            if self.is_thinking_model:
                logger.info("Processing chat response as thinking model")
                thinking_content, final_response = self._extract_thinking(response)
                if thinking_content:
                    logger.info(f"Extracted thinking content from chat ({len(thinking_content)} chars)")

            # Build message content
            message_content = {
                "role": "assistant",
                "content": final_response if not tool_calls else None
            }

            if tool_calls:
                message_content["tool_calls"] = tool_calls

            result = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_path,
                "choices": [{
                    "index": 0,
                    "message": message_content,
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
            # Add thinking content if present
            if thinking_content:
                result["choices"][0]["message"]["thinking"] = thinking_content
            
            return result
        except ValueError as e:
            error_msg = f"Invalid chat generation parameters for model {self.model_path}. Check max_tokens ({max_tokens}), temperature ({temperature}), top_p ({top_p}), repetition_penalty ({repetition_penalty}). Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                error_msg = f"Out of memory during chat generation with model {self.model_path}. Try reducing max_tokens (current: {max_tokens}). Error: {e}"
            else:
                error_msg = f"Runtime error during chat generation with model {self.model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during chat generation with model {self.model_path}. Messages: {len(messages)} message(s), max_tokens: {max_tokens}, temperature: {temperature}. Error type: {type(e).__name__}, Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _format_prompt_with_tools(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format prompt with tools information."""
        import json
        
        if not tools:
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        # Format tools as a system message
        tools_description = "You have access to the following tools:\n\n"
        for tool in tools:
            func = tool.get("function", {})
            tools_description += f"- {func.get('name')}: {func.get('description')}\n"
            if func.get('parameters'):
                tools_description += f"  Parameters: {json.dumps(func.get('parameters'))}\n"
        
        tools_description += "\nTo use a tool, respond with a JSON object in this format:\n"
        tools_description += '{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}\n\n'
        
        # Insert tools description as system message
        formatted_messages = [{"role": "system", "content": tools_description}] + messages
        
        # Apply chat template or simple formatting
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                pass
        
        return "\n".join([f"{m['role']}: {m['content']}" for m in formatted_messages])
    
    def _extract_tool_calls(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from generated text."""
        import json
        import re
        import time
        
        # Try to find JSON with tool_calls
        json_pattern = r'\{["\']tool_calls["\']\s*:\s*\[.*?\]\s*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            try:
                data = json.loads(match.group(0))
                tool_calls = data.get("tool_calls", [])
                
                # Convert to OpenAI format
                formatted_calls = []
                for i, call in enumerate(tool_calls):
                    formatted_calls.append({
                        "id": f"call_{int(time.time())}_{i}",
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.dumps(call.get("arguments", {}))
                        }
                    })
                
                return formatted_calls if formatted_calls else None
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: XML-style tags
        xml_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(xml_pattern, text, re.DOTALL)
        
        if matches:
            formatted_calls = []
            for i, match_text in enumerate(matches):
                try:
                    call_data = json.loads(match_text)
                    formatted_calls.append({
                        "id": f"call_{int(time.time())}_{i}",
                        "type": "function",
                        "function": {
                            "name": call_data.get("name"),
                            "arguments": json.dumps(call_data.get("arguments", {}))
                        }
                    })
                except json.JSONDecodeError:
                    continue
            
            return formatted_calls if formatted_calls else None
        
        return None
    
    def _extract_thinking(self, text: str) -> tuple[Optional[str], str]:
        """
        Extract thinking/reasoning content from generated text.
        
        Returns:
            tuple: (thinking_content, final_response)
        """
        import re
        
        # Pattern 1: <think>...</think> tags
        think_pattern = r'<think>(.*?)</think>'
        match = re.search(think_pattern, text, re.DOTALL)
        
        if match:
            thinking = match.group(1).strip()
            # Remove the thinking section from the response
            final_response = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
            return thinking, final_response
        
        # Pattern 2: <reasoning>...</reasoning> tags
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        match = re.search(reasoning_pattern, text, re.DOTALL)
        
        if match:
            thinking = match.group(1).strip()
            final_response = re.sub(reasoning_pattern, '', text, flags=re.DOTALL).strip()
            return thinking, final_response
        
        # Pattern 3: Markdown-style thinking blocks
        md_thinking_pattern = r'```thinking\n(.*?)\n```'
        match = re.search(md_thinking_pattern, text, re.DOTALL)
        
        if match:
            thinking = match.group(1).strip()
            final_response = re.sub(md_thinking_pattern, '', text, flags=re.DOTALL).strip()
            return thinking, final_response
        
        # No thinking found
        return None, text

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings using MLX."""
        if not self.loaded:
            error_msg = f"Model {self.model_path} is not loaded. Please start the server first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            import mlx.core as mx
            import time

            logger.info(f"Generating embeddings for {len(texts)} texts with MLX")

            # Check if model supports embeddings
            if not (hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens')):
                error_msg = f"MLX model {self.model_path} does not support embeddings. The model may not have embed_tokens layer."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

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
                "trust_remote_code": self.trust_remote_code,
                "is_thinking_model": self.is_thinking_model
            },
            "capabilities": [
                ModelCapability.TEXT_GENERATION.value,
                ModelCapability.EMBEDDINGS.value
            ],
            "platform": "Apple Silicon (M-series)"
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.MLX, MLXBackend)