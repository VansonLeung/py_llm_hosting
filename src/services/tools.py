import copy
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from src.libs.config import settings
from src.libs.logging import logger

ToolExecutor = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]

MAX_TOOL_ITERATIONS = 3
TOOL_SERVICE_PATH = "/api/v1/tool-executions"
TOOL_SERVICE_TIMEOUT = 30.0


class ToolExecutionError(RuntimeError):
    """Raised when a tool execution fails."""


async def handle_tool_calls(
    *,
    base_messages: list[dict[str, Any]],
    initial_response: dict[str, Any],
    tools: list[dict[str, Any]],
    call_llm: ToolExecutor,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> dict[str, Any]:
    """Execute requested tool calls and loop until the LLM returns content."""

    if not tools:
        return initial_response

    response = initial_response
    messages = copy.deepcopy(base_messages)
    execution_log: list[dict[str, Any]] = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        tool_calls = _extract_tool_calls(response)
        if not tool_calls:
            break

        assistant_message = (response.get("choices") or [{}])[0].get("message", {})
        messages.append(
            {
                "role": assistant_message.get("role", "assistant"),
                "content": assistant_message.get("content"),
                "tool_calls": tool_calls,
            }
        )

        tool_messages, batch_log = await _execute_tool_batch_remote(tool_calls, tools)
        execution_log.extend(batch_log)
        messages.extend(tool_messages)

        response = await call_llm(messages)

    if execution_log:
        response["tool_execution"] = execution_log

    return response


def _extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    if not response:
        return []
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls")
    return tool_calls or []


async def _execute_tool_batch_remote(
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not tool_calls:
        return [], []

    url = _build_tool_service_endpoint()
    payload = {
        "tool_calls": tool_calls,
        "tools": tools,
    }

    async with httpx.AsyncClient(timeout=TOOL_SERVICE_TIMEOUT) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            status_code = exc.response.status_code
            logger.error("Tool service error %s: %s", status_code, detail)
            message = f"Tool service error {status_code}: {detail}"
            raise ToolExecutionError(message) from exc
        except httpx.HTTPError as exc:
            logger.error("Tool service request failed: %s", exc)
            message = "Tool service request failed"
            raise ToolExecutionError(message) from exc

    data = response.json()
    tool_messages = data.get("tool_messages") or []
    execution_log = data.get("execution_log") or []
    return tool_messages, execution_log


def _build_tool_service_endpoint() -> str:
    base_url = settings.tool_service_url or os.getenv("TOOL_SERVICE_URL")
    if not base_url:
        error_message = (
            "Tool service URL is not configured. "
            "Set LLM_TOOL_SERVICE_URL or TOOL_SERVICE_URL."
        )
        raise ToolExecutionError(error_message)
    base = base_url.rstrip("/")
    return f"{base}{TOOL_SERVICE_PATH}"


def validate_tools(tools: list[dict[str, Any]]) -> bool:
    """Validate tool definitions."""

    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        if not func.get("name") or not func.get("parameters"):
            return False

    return True
