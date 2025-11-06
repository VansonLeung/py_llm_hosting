"""
vLLM backend implementation.

High-performance inference engine optimized for throughput and GPU utilization.
Supports HuggingFace Transformers models.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.libs.logging import logger
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
            error_msg = f"vLLM backend requires vLLM library. Install with: pip install vllm transformers. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except FileNotFoundError as e:
            error_msg = f"Model not found at path: {self.model_path}. The model will be downloaded automatically on first use. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except ValueError as e:
            error_msg = f"Invalid vLLM configuration for model {self.model_path}. Check tensor_parallel_size, gpu_memory_utilization, max_model_len, and quantization settings. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                error_msg = f"GPU/CUDA error while loading model {self.model_path}. Ensure CUDA is installed and GPU is available. Error: {e}"
            elif "memory" in str(e).lower():
                error_msg = f"Out of memory while loading model {self.model_path}. Try reducing gpu_memory_utilization (current: {self.gpu_memory_utilization}) or max_model_len (current: {self.max_model_len}). Error: {e}"
            else:
                error_msg = f"Runtime error loading model {self.model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading model {self.model_path} with vLLM backend. Model path: {self.model_path}, Config: tensor_parallel={self.tensor_parallel_size}, gpu_util={self.gpu_memory_utilization}, max_len={self.max_model_len}. Error type: {type(e).__name__}, Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from prompt with optional tools support."""
        if not self.loaded:
            error_msg = f"Model {self.model_path} is not loaded. Please start the server first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            from vllm import SamplingParams
            
            max_tokens = max_tokens or 512
            
            # If tools are provided, format the prompt with tool information
            if tools:
                logger.debug(f"Tools provided for vLLM generation ({len(tools)} tools), formatting prompt")
                # Convert tools to a simple prompt format for basic generation
                tools_description = "You have access to the following tools:\n\n"
                for i, tool in enumerate(tools):
                    func = tool.get("function", {})
                    tool_name = func.get('name', f'tool_{i}')
                    tool_desc = func.get('description', 'No description')
                    logger.debug(f"Adding tool {i+1}: {tool_name} - {tool_desc}")
                    tools_description += f"- {tool_name}: {tool_desc}\n"
                    if func.get('parameters'):
                        import json
                        tools_description += f"  Parameters: {json.dumps(func.get('parameters'))}\n"
                
                tools_description += "\nTo use a tool, respond with a JSON object in this format:\n"
                tools_description += '{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}\n\n'
                
                # Prepend tools description to prompt
                enhanced_prompt = tools_description + prompt
                logger.debug(f"Enhanced vLLM prompt length: {len(enhanced_prompt)} (original: {len(prompt)})")
            else:
                enhanced_prompt = prompt
                logger.debug("No tools provided for vLLM generation")
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs
            )
            
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: self.llm.generate([enhanced_prompt], sampling_params)
            )
            
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            # If tools were provided, check for tool calls in the response
            tool_calls = None
            if tools:
                logger.debug(f"Checking for tool calls in vLLM response (length: {len(generated_text)})")
                tool_calls = self._extract_tool_calls(generated_text)
                if tool_calls:
                    logger.debug(f"Extracted {len(tool_calls)} tool calls from vLLM response")
                else:
                    logger.debug("No tool calls found in vLLM response")
            
            return {
                "text": generated_text,
                "tool_calls": tool_calls,
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
                }
            }
        except ValueError as e:
            error_msg = f"Invalid generation parameters for model {self.model_path}. Check max_tokens ({max_tokens}), temperature ({temperature}), top_p ({top_p}). Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                error_msg = f"Out of memory during generation with model {self.model_path}. Try reducing max_tokens (current: {max_tokens}) or batch size. Error: {e}"
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
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs
    ):
        """Generate chat completion (streaming or non-streaming) with optional tool calling support."""
        if not self.loaded:
            error_msg = f"Model {self.model_path} is not loaded. Please start the server first using: python main.py server start <server-id>"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            from vllm import SamplingParams
            import time
            import json
            import re
            
            # Apply chat template with tools support
            try:
                if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                    # Try to apply chat template with tools
                    try:
                        if tools:
                            # Some models support tools in chat template
                            prompt = self.tokenizer.apply_chat_template(
                                messages,
                                tools=tools,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        else:
                            prompt = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                    except TypeError:
                        # If tokenizer doesn't support tools parameter, fall back
                        logger.warning("Tokenizer doesn't support tools parameter, applying manual tool formatting")
                        prompt = self._format_prompt_with_tools(messages, tools)
                else:
                    # Fallback: manual formatting
                    prompt = self._format_prompt_with_tools(messages, tools)
            except Exception as e:
                error_msg = f"Error formatting chat prompt for model {self.model_path}. Check message format: {messages}. Error: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            max_tokens = max_tokens or 512
            
            try:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                error_msg = f"Invalid sampling parameters for model {self.model_path}. temperature={temperature}, max_tokens={max_tokens}, kwargs={kwargs}. Error: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Handle streaming
            if stream:
                async def stream_wrapper():
                    try:
                        loop = asyncio.get_event_loop()
                        
                        # vLLM's LLM.generate doesn't support streaming directly
                        # We need to generate the full response and simulate streaming
                        # For production use, consider using vLLM's AsyncLLMEngine instead
                        outputs = await loop.run_in_executor(
                            None,
                            lambda: self.llm.generate([prompt], sampling_params)
                        )
                        
                        output = outputs[0]
                        generated_text = output.outputs[0].text
                        
                        # Parse tool calls if present
                        tool_calls = self._extract_tool_calls(generated_text) if tools else None
                        
                        if tool_calls:
                            # Yield tool calls as structured data
                            yield {
                                "tool_calls": tool_calls,
                                "content": None
                            }
                        else:
                            # Simulate streaming by yielding words/tokens
                            # In a real implementation with AsyncLLMEngine, this would be true streaming
                            words = generated_text.split()
                            for i, word in enumerate(words):
                                if i == 0:
                                    yield word
                                else:
                                    yield " " + word
                                await asyncio.sleep(0.01)  # Small delay for visual effect
                    except Exception as e:
                        error_msg = f"Error during streaming generation with model {self.model_path}. Error type: {type(e).__name__}, Error: {e}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                
                return stream_wrapper()
            
            # Non-streaming
            try:
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self.llm.generate([prompt], sampling_params)
                )
                
                output = outputs[0]
                generated_text = output.outputs[0].text
                
                # Parse tool calls if tools were provided
                tool_calls = self._extract_tool_calls(generated_text) if tools else None
                
                # Build response
                message_content = {
                    "role": "assistant",
                    "content": generated_text if not tool_calls else None
                }
                
                if tool_calls:
                    message_content["tool_calls"] = tool_calls
                
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_path,
                    "choices": [{
                        "index": 0,
                        "message": message_content,
                        "finish_reason": "tool_calls" if tool_calls else output.outputs[0].finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": len(output.outputs[0].token_ids),
                        "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
                    }
                }
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    error_msg = f"Out of memory during chat generation with model {self.model_path}. Try reducing max_tokens (current: {max_tokens}) or gpu_memory_utilization (current: {self.gpu_memory_utilization}). Error: {e}"
                else:
                    error_msg = f"Runtime error during chat generation with model {self.model_path}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error during chat generation with model {self.model_path}. Messages: {len(messages)} message(s), max_tokens: {max_tokens}, temperature: {temperature}. Error type: {type(e).__name__}, Error: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            # Catch any exceptions from the outer try block (prompt formatting, etc.)
            if not isinstance(e, RuntimeError):
                error_msg = f"Error in generate_chat for model {self.model_path}: {type(e).__name__}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                raise
    
    def _format_prompt_with_tools(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format prompt with tools information."""
        import json
        
        if not tools:
            # Simple concatenation without tools
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
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                pass
        
        # Fallback formatting
        return "\n".join([f"{m['role']}: {m['content']}" for m in formatted_messages])
    
    def _extract_tool_calls(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from generated text."""
        import json
        import re
        import time
        
        # Try to find JSON with tool_calls
        # Look for patterns like {"tool_calls": [...]} or <tool_call>...</tool_call>
        
        # Pattern 1: JSON format
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