import json
import logging
from typing import Any

import httpx

from .websearch import WebSearchError, perform_web_search

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_RESULTS = 5


class ToolExecutionError(RuntimeError):
    """Raised when tool execution fails."""


async def execute_tool_batch(
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tool_messages: list[dict[str, Any]] = []
    execution_log: list[dict[str, Any]] = []
    tool_map = _build_tool_map(tools)

    for tool_call in tool_calls or []:
        func = tool_call.get("function", {})
        name = (func.get("name") or "").strip()
        definition = tool_map.get(name.lower())

        if definition is None:
            exc: Exception = ToolExecutionError(f"Tool '{name}' is not registered")
        else:
            try:
                log_entry, tool_message = await _execute_tool(tool_call, definition)
                tool_messages.append(tool_message)
                execution_log.append(log_entry)
                continue
            except ToolExecutionError as tool_exc:
                exc = tool_exc

        error_payload = {
            "error": str(exc),
            "tool": name or "unknown",
        }
        log_entry = {
            "id": tool_call.get("id"),
            "type": "function",
            "name": name,
            "arguments": {},
            "error": str(exc),
        }
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": json.dumps(error_payload),
        }
        logger.warning("Tool execution failed: %s", exc)

        tool_messages.append(tool_message)
        execution_log.append(log_entry)

    return tool_messages, execution_log


def _build_tool_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tool_map: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = (func.get("name") or "").strip()
        if name:
            tool_map[name.lower()] = tool
    return tool_map


async def _execute_tool(
    tool_call: dict[str, Any],
    tool_definition: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    func = tool_call.get("function", {})
    name = func.get("name") or tool_definition.get("function", {}).get("name")
    arguments_raw = func.get("arguments") or "{}"

    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError("Tool arguments must be valid JSON") from exc

    handler_name = (name or "").lower()

    if handler_name in {"websearch", "web_search"}:
        return await _run_web_search(tool_call, name, arguments)

    raise ToolExecutionError(f"Tool '{name}' is not supported")


async def _run_web_search(
    tool_call: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolExecutionError("WebSearch tool requires a non-empty 'query' string")

    num_results = arguments.get("num_results", DEFAULT_SEARCH_RESULTS)

    try:
        search_payload = await perform_web_search(query, max_results=num_results)
    except (WebSearchError, httpx.HTTPError) as exc:
        raise ToolExecutionError(f"Web search failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"Unexpected web search error: {exc}") from exc

    content = {
        "query": search_payload.get("query"),
        "provider": search_payload.get("provider"),
        "results": search_payload.get("results", []),
    }

    log_entry = {
        "id": tool_call.get("id"),
        "type": "function",
        "name": tool_name,
        "arguments": arguments,
        "output": content,
    }

    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "content": json.dumps(content, ensure_ascii=False),
    }

    return log_entry, tool_message
