# Data Model: LLM Endpoint Hosting

**Date**: 2025-11-05
**Feature**: LLM Endpoint Hosting

## Entities

### LLM Server
**Purpose**: Represents a configured LLM server/endpoint that can be hosted.

**Fields**:
- `id`: string (UUID, unique identifier)
- `name`: string (human-readable name, required, 1-50 chars)
- `endpoint_url`: string (HTTP URL of the LLM server, required, valid URL)
- `model_name`: string (model identifier, e.g., "gpt-3.5-turbo", required)
- `status`: enum ("active", "inactive", "error") (current operational status)
- `config`: object (optional additional configuration, key-value pairs)
- `created_at`: datetime (ISO format)
- `updated_at`: datetime (ISO format)

**Relationships**:
- None (standalone entity)

**Validation Rules**:
- `endpoint_url` must be valid HTTP/HTTPS URL
- `name` must be unique across all servers
- `model_name` cannot be empty
- `config` must be valid JSON object if provided

**State Transitions**:
- "inactive" → "active": When server becomes reachable
- "active" → "error": When server fails health check
- "error" → "active": When server recovers

### API Request
**Purpose**: Represents an incoming OpenAI-compatible API request.

**Fields**:
- `id`: string (UUID, request identifier)
- `endpoint`: string (API endpoint, e.g., "/v1/chat/completions")
- `method`: string ("GET", "POST")
- `headers`: object (HTTP headers)
- `body`: object (request payload, validated per OpenAI schema)
- `auth_key`: string (optional API key)
- `timestamp`: datetime (request time)

**Relationships**:
- References LLM Server via model selection in body

**Validation Rules**:
- `body` must conform to OpenAI API schema for the endpoint
- `auth_key` validated if provided
- `endpoint` must be supported endpoint

### MCP Context
**Purpose**: Represents Model Context Protocol data for tool integration.

**Fields**:
- `session_id`: string (UUID, MCP session identifier)
- `tools`: array (list of available tools, each with name, description, parameters)
- `context`: object (current conversation context)
- `state`: enum ("active", "completed", "error")
- `messages`: array (MCP message history)

**Relationships**:
- Associated with API Request for tool calls

**Validation Rules**:
- `tools` must be valid MCP tool definitions
- `context` must be valid MCP context object
- `messages` must conform to MCP message format

**State Transitions**:
- "active" → "completed": When tool execution finishes
- "active" → "error": When tool fails
- "completed" → "active": New tool request

## Data Flow

1. User adds LLM Server via CLI → persisted to JSON
2. API receives request → validates against API Request model
3. If MCP involved → creates/updates MCP Context
4. Request proxied to appropriate LLM Server
5. Response formatted to OpenAI-compatible format

## Persistence

- All entities persisted in single JSON file: `servers.json`
- Structure: `{"servers": [LLM Server objects], "mcp_sessions": [MCP Context objects]}`
- Atomic writes to prevent corruption
- Backup on updates

## Validation

- Pydantic models used for runtime validation
- JSON schema validation for persistence
- Business rule validation in service layer