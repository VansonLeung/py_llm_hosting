from typing import Any

from pydantic import BaseModel, Field


class ToolBatchRequest(BaseModel):
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)


class ToolBatchResponse(BaseModel):
    tool_messages: list[dict[str, Any]] = Field(default_factory=list)
    execution_log: list[dict[str, Any]] = Field(default_factory=list)
