"""
MLX-VLM backend implementation.

Apple Silicon optimized vision-language model inference using MLX.
Supports multimodal models that can process both text and images.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from pathlib import Path
import base64
import io
import asyncio
import httpx
from src.models.backend import ModelBackend, ModelCapability, ModelBackendType, ModelBackendFactory
from src.libs.logging import logger


class MLXVLMBackend(ModelBackend):
    """MLX-VLM backend for vision-language models on Apple Silicon."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model = None
        self.processor = None
        self.max_tokens_default = kwargs.get("max_tokens", 512)

    def _load_image_from_source(self, image_source: str):
        """Load an image from local path, HTTP(S) URL, or data URI."""
        if not isinstance(image_source, str) or not image_source.strip():
            error_msg = (
                f"Invalid image reference: expected non-empty string, got {repr(image_source)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            from PIL import Image
        except ImportError as e:
            error_msg = (
                f"Pillow not installed. Install with: pip install pillow. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        normalized_source = image_source.strip()

        # Resolve image bytes from supported schemes
        image_bytes: bytes
        if normalized_source.startswith("data:"):
            if "," not in normalized_source:
                error_msg = (
                    f"Invalid data URI format for image: {normalized_source[:32]}..."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            _, data_part = normalized_source.split(",", 1)
            try:
                image_bytes = base64.b64decode(data_part)
            except (base64.binascii.Error, ValueError) as e:
                error_msg = (
                    f"Failed to decode base64 image data. Error: {e}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif normalized_source.startswith("http://") or normalized_source.startswith("https://"):
            try:
                response = httpx.get(normalized_source, timeout=30.0)
                response.raise_for_status()
                image_bytes = response.content
            except httpx.HTTPError as e:
                error_msg = (
                    f"Failed to download image from URL: {normalized_source}. "
                    f"Ensure the URL is reachable. Error: {e}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            path = Path(normalized_source).expanduser()
            if not path.exists():
                error_msg = (
                    f"Image file not found: {path}. Provide an absolute path, URL, or data URI."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            try:
                image_bytes = path.read_bytes()
            except OSError as e:
                error_msg = (
                    f"Failed to read image file: {path}. Error: {e}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # Ensure image is fully decoded
            return image
        except Exception as e:
            error_msg = (
                f"Failed to decode image from source: {normalized_source}. Error: {e}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def load_model(self) -> None:
        """Load the MLX-VLM model."""
        if self.loaded:
            return

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as e:
            error_msg = (
                f"MLX-VLM not installed. Install with: pip install mlx-vlm. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            logger.info(f"Loading MLX-VLM model from: {self.model_path}")
            self.model, self.processor = load(self.model_path)
            self.loaded = True
            logger.info(f"Successfully loaded MLX-VLM model: {self.model_path}")
        except FileNotFoundError as e:
            error_msg = (
                f"Model path not found: {self.model_path}. "
                f"Please ensure the model path exists and is accessible. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except ValueError as e:
            error_msg = (
                f"Invalid model configuration for path: {self.model_path}. "
                f"Please verify the model is a valid MLX-VLM model. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            error_msg = (
                f"Failed to load MLX-VLM model: {self.model_path}. "
                f"This may be due to insufficient memory or incompatible model format. "
                f"Try using a smaller model or check Apple Silicon compatibility. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error loading MLX-VLM model: {self.model_path}. "
                f"Error type: {type(e).__name__}, Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model:
            self.model = None
            self.processor = None
            self.loaded = False
            logger.info("Vision-language model unloaded")

    async def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from prompt and image reference (path/URL/data URI)."""
        if not self.loaded:
            error_msg = (
                "Model not loaded. Call load_model() first. "
                f"Model path: {self.model_path}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            from mlx_vlm import generate as vlm_generate
        except ImportError as e:
            error_msg = (
                f"Required libraries not installed. Install with: pip install mlx-vlm pillow. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Validate parameters
        if not isinstance(prompt, str) or not prompt.strip():
            error_msg = (
                f"Invalid prompt: must be non-empty string. "
                f"Got: {type(prompt).__name__} with value: {repr(prompt[:100])}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(image_path, str) or not image_path.strip():
            error_msg = (
                f"Invalid image_path: must be non-empty string. "
                f"Got: {type(image_path).__name__} with value: {repr(image_path)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            error_msg = (
                f"Invalid max_tokens: must be positive integer or None. "
                f"Got: {type(max_tokens).__name__} with value: {max_tokens}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(temperature, (int, float)) or temperature < 0:
            error_msg = (
                f"Invalid temperature: must be non-negative number. "
                f"Got: {type(temperature).__name__} with value: {temperature}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        max_tokens = max_tokens or self.max_tokens_default

        # Load and validate image
        image = self._load_image_from_source(image_path)

        try:
            logger.debug(f"Generating text with image. Prompt length: {len(prompt)}, image: {image_path}, max_tokens: {max_tokens}")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: vlm_generate(
                    self.model,
                    self.processor,
                    image,
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    verbose=False
                )
            )

            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }

            logger.debug(f"Generation with image completed. Usage: {usage}")
            return {
                "text": response,
                "usage": usage
            }

        except RuntimeError as e:
            error_msg = (
                f"Generation with image failed due to runtime error. "
                f"This may be due to insufficient memory or model incompatibility. "
                f"Prompt length: {len(prompt)}, image: {image_path}, max_tokens: {max_tokens}, temperature: {temperature}. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error during generation with image. "
                f"Error type: {type(e).__name__}, Error: {e}. "
                f"Model: {self.model_path}, Prompt length: {len(prompt)}, Image: {image_path}"
            )
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
        """Generate chat completion (supports images in messages)."""
        if not self.loaded:
            error_msg = (
                "Model not loaded. Call load_model() first. "
                f"Model path: {self.model_path}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            from mlx_vlm import generate as vlm_generate
            import time
        except ImportError as e:
            error_msg = (
                f"Required libraries not installed. Install with: pip install mlx-vlm pillow. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Validate parameters
        if not isinstance(messages, list) or not messages:
            error_msg = (
                f"Invalid messages: must be non-empty list of message dictionaries. "
                f"Got: {type(messages).__name__} with length: {len(messages) if isinstance(messages, list) else 'N/A'}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            error_msg = (
                f"Invalid max_tokens: must be positive integer or None. "
                f"Got: {type(max_tokens).__name__} with value: {max_tokens}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(temperature, (int, float)) or temperature < 0:
            error_msg = (
                f"Invalid temperature: must be non-negative number. "
                f"Got: {type(temperature).__name__} with value: {temperature}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if stream:
            error_msg = (
                f"Streaming not supported by MLX-VLM backend. "
                f"Set stream=False for chat completions."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Extract text and images from messages
        text_parts = []
        image = None

        try:
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    error_msg = (
                        f"Invalid message at index {i}: must be dictionary. "
                        f"Got: {type(msg).__name__}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                role = msg.get("role")
                if role not in ["user", "assistant", "system"]:
                    error_msg = (
                        f"Invalid role in message {i}: must be 'user', 'assistant', or 'system'. "
                        f"Got: {repr(role)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                content = msg.get("content", "")

                # Check if content is a list (multimodal)
                if isinstance(content, list):
                    for j, part in enumerate(content):
                        if not isinstance(part, dict):
                            error_msg = (
                                f"Invalid content part at message {i}, part {j}: must be dictionary. "
                                f"Got: {type(part).__name__}"
                            )
                            logger.error(error_msg)
                            raise ValueError(error_msg)

                        part_type = part.get("type")
                        if part_type == "text":
                            text = part.get("text", "")
                            if not isinstance(text, str):
                                error_msg = (
                                    f"Invalid text in message {i}, part {j}: must be string. "
                                    f"Got: {type(text).__name__}"
                                )
                                logger.error(error_msg)
                                raise ValueError(error_msg)
                            text_parts.append(f"{role}: {text}")
                        elif part_type == "image_url":
                            if image is not None:
                                error_msg = (
                                    f"Multiple images not supported. Found second image in message {i}, part {j}."
                                )
                                logger.error(error_msg)
                                raise ValueError(error_msg)

                            image_url = part.get("image_url", {}).get("url")
                            if not isinstance(image_url, str) or not image_url.strip():
                                error_msg = (
                                    f"Invalid image_url in message {i}, part {j}: must be non-empty string. "
                                    f"Got: {repr(image_url)}"
                                )
                                logger.error(error_msg)
                                raise ValueError(error_msg)

                            image = self._load_image_from_source(image_url)
                        else:
                            error_msg = (
                                f"Unsupported content type in message {i}, part {j}: {part_type}. "
                                f"Supported types: 'text', 'image_url'"
                            )
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                else:
                    if not isinstance(content, str):
                        error_msg = (
                            f"Invalid content in message {i}: must be string or list. "
                            f"Got: {type(content).__name__}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    text_parts.append(f"{role}: {content}")

            if not text_parts:
                error_msg = (
                    f"No text content found in messages. "
                    f"At least one message must contain text."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            if "error_msg" not in locals():
                error_msg = (
                    f"Unexpected error processing messages. "
                    f"Error type: {type(e).__name__}, Error: {e}"
                )
                logger.error(error_msg)
            raise ValueError(error_msg)

        prompt = "\n".join(text_parts)
        max_tokens = max_tokens or self.max_tokens_default

        try:
            logger.debug(f"Generating chat completion. Prompt length: {len(prompt)}, has_image: {image is not None}, max_tokens: {max_tokens}")
            loop = asyncio.get_event_loop()
            if image:
                response = await loop.run_in_executor(
                    None,
                    lambda: vlm_generate(
                        self.model,
                        self.processor,
                        image,
                        prompt,
                        max_tokens=max_tokens,
                        temp=temperature,
                        verbose=False
                    )
                )
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: vlm_generate(
                        self.model,
                        self.processor,
                        prompt,
                        max_tokens=max_tokens,
                        temp=temperature,
                        verbose=False
                    )
                )

            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }

            logger.debug(f"Chat completion generated. Usage: {usage}")
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
                "usage": usage
            }

        except RuntimeError as e:
            error_msg = (
                f"Chat completion failed due to runtime error. "
                f"This may be due to insufficient memory or model incompatibility. "
                f"Prompt length: {len(prompt)}, has_image: {image is not None}, max_tokens: {max_tokens}, temperature: {temperature}. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error during chat completion. "
                f"Error type: {type(e).__name__}, Error: {e}. "
                f"Model: {self.model_path}, Prompt length: {len(prompt)}, Has image: {image is not None}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings (not supported)."""
        raise NotImplementedError(
            "MLX-VLM backend does not support embeddings. "
            "Use a dedicated embedding model or llama.cpp backend."
        )

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        return capability in [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.VISION
        ]

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "mlx_vlm",
            "model_path": self.model_path,
            "loaded": self.loaded,
            "config": {
                "max_tokens": self.max_tokens_default
            },
            "capabilities": [
                ModelCapability.TEXT_GENERATION.value,
                ModelCapability.VISION.value
            ],
            "platform": "Apple Silicon (M-series)",
            "supports_images": True
        }


# Register the backend
ModelBackendFactory.register_backend(ModelBackendType.MLX_VLM, MLXVLMBackend)