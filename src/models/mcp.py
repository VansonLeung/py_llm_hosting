from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import uuid

class MCPState(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"

class MCPTool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class MCPMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class MCPContext(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tools: List[MCPTool] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    state: MCPState = MCPState.ACTIVE
    messages: List[MCPMessage] = Field(default_factory=list)