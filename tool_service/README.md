# Tool Service

This standalone FastAPI microservice executes assistant tool calls so the main `py_llm_hosting` API can stay lean. It currently implements the `websearch` tool (Google Custom Search with DuckDuckGo fallback) and returns responses that are compatible with the OpenAI tool-calling schema.

## Features
- Executes one or more tool calls in a single request and returns both the tool messages and an execution log.
- Google Custom Search support with DuckDuckGo fallback (same environment variables as the main repo).
- Health endpoint (`/health`) for readiness/liveness probes.

## Installation

```bash
cd tool_service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running locally

```bash
uvicorn tool_service.app.main:app --reload --port 9001
```

Set the following optional environment variables to configure providers:

- `LLM_GOOGLE_API_KEY` or `GOOGLE_API_KEY`
- `LLM_GOOGLE_CSE_ID` or `GOOGLE_CSE_ID`

## API

### `POST /api/v1/tool-executions`

Request body:

```json
{
  "tool_calls": [ { ... } ],
  "tools": [ { ... } ]
}
```

Response body:

```json
{
  "tool_messages": [ { ... } ],
  "execution_log": [ { ... } ]
}
```

### `GET /health`

Returns `{ "status": "ok" }` when the service is ready.
