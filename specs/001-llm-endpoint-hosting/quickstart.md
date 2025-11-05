# Quickstart: LLM Endpoint Hosting

**Date**: 2025-11-05
**Feature**: LLM Endpoint Hosting

## Prerequisites

- Python 3.11+
- Access to LLM servers (e.g., local LM Studio instance, remote API)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd py_llm_hosting
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set API key for authentication:
   ```bash
   export LLM_API_KEY=your-secret-key
   ```

## Basic Usage

### Start the Server

```bash
python -m src.cli start --port 8000
```

The API will be available at `http://localhost:8000`

### Add an LLM Server

```bash
python -m src.cli add-server \
  --name "My Local LLM" \
  --endpoint "http://localhost:1234/v1/chat/completions" \
  --model "local-model"
```

### List Servers

```bash
python -m src.cli list-servers
```

### Remove a Server

```bash
python -m src.cli remove-server --id <server-id>
```

## API Usage

### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }'
```

### With Tools

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "What is the weather?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            }
          }
        }
      }
    ]
  }'
```

### Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{
    "model": "local-model",
    "input": "Hello, world!"
  }'
```

### Ranking

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{
    "model": "local-model",
    "query": "What is AI?",
    "documents": [
      "Artificial Intelligence is a field of computer science",
      "Machine learning is a subset of AI",
      "Deep learning uses neural networks"
    ]
  }'
```

## Configuration

Servers are stored in `servers.json` in the working directory.

Example configuration:
```json
{
  "servers": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Local LM Studio",
      "endpoint_url": "http://localhost:1234/v1/chat/completions",
      "model_name": "llama-2-7b",
      "status": "active",
      "config": {},
      "created_at": "2025-11-05T10:00:00Z",
      "updated_at": "2025-11-05T10:00:00Z"
    }
  ]
}
```

## Troubleshooting

### Server Not Responding

- Check that the LLM server is running and accessible
- Verify the endpoint URL is correct
- Check server logs for connection errors

### Authentication Errors

- Ensure API key is set if authentication is required
- Check that the key is passed in the Authorization header

### Performance Issues

- Monitor concurrent requests (limited to 50)
- Check LLM server performance
- Review response times in logs

## Next Steps

- Run tests: `pytest`
- Generate API docs: Visit `http://localhost:8000/docs`
- Check logs for detailed information