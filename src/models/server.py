from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
import uuid

class ServerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

class ServerMode(str, Enum):
    """Server operation mode."""
    PROXY = "proxy"  # Proxy to external server
    SELF_HOSTED = "self-hosted"  # Host model locally

class LLMServer(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=50)
    model_name: str
    status: ServerStatus = ServerStatus.INACTIVE
    mode: ServerMode = ServerMode.PROXY
    
    # Proxy mode fields
    endpoint_url: Optional[str] = None
    
    # Self-hosted mode fields
    model_path: Optional[str] = None  # HuggingFace model ID or local path
    backend_type: Optional[str] = None  # "llama-cpp", "transformers", "vllm"
    backend_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Common fields
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator('endpoint_url')
    @classmethod
    def validate_endpoint_url(cls, v, info):
        # Only validate if mode is proxy
        if info.data.get('mode') == ServerMode.PROXY and v:
            from urllib.parse import urlparse
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError('Invalid URL format')
        return v

    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v, info):
        # Require model_path if mode is self-hosted
        if info.data.get('mode') == ServerMode.SELF_HOSTED and not v:
            raise ValueError('model_path required for self-hosted mode')
        return v

    @field_validator('name')
    @classmethod
    def validate_name_unique(cls, v):
        # Note: uniqueness check done at persistence level
        return v

    def update_timestamp(self):
        self.updated_at = datetime.utcnow()