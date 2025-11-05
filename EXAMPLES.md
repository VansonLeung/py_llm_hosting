# Example Usage Guide

## Quick Examples

### 1. Add a Local LLM Server (e.g., LM Studio)

```bash
python main.py add-server \
  --name "LM Studio" \
  --endpoint "http://localhost:1234/v1/chat/completions" \
  --model "llama-2-7b-chat"
```

### 2. Add an OpenAI-Compatible Server

```bash
python main.py add-server \
  --name "OpenAI" \
  --endpoint "https://api.openai.com/v1/chat/completions" \
  --model "gpt-3.5-turbo"
```

### 3. Start the Hosting API

```bash
python main.py start --port 8000
```

### 4. Test Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### 5. Test Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

### 6. Test Reranking

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of artificial intelligence",
      "Python is a programming language",
      "Neural networks are used in deep learning",
      "Machine learning algorithms learn from data"
    ],
    "top_n": 2
  }'
```

### 7. Remove a Server

```bash
# First, list servers to get the ID
python main.py list-servers

# Then remove by ID
python main.py remove-server --id "5cab8af7-1240-45b4-a58f-5481289f40e0"
```

## With API Key Authentication

### Set API Key

```bash
export LLM_API_KEY="my-secret-key-123"
```

### Make Authenticated Requests

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key-123" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Integration with Python Code

```python
import requests

# Add a server
response = requests.post('http://localhost:8000/v1/chat/completions',
    json={
        "model": "llama-2-7b-chat",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
)

print(response.json())
```

## Integration with OpenAI Python Client

```python
from openai import OpenAI

# Point OpenAI client to your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # or your LLM_API_KEY if set
)

# Use as normal
response = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ]
)

print(response.choices[0].message.content)
```

## Common Workflows

### Single Local LLM

1. Start your local LLM (e.g., LM Studio on port 1234)
2. Add it to the hosting service:
   ```bash
   python main.py add-server --name "Local" --endpoint "http://localhost:1234/v1/chat/completions" --model "local-model"
   ```
3. Start the hosting API:
   ```bash
   python main.py start
   ```
4. Now your local LLM is available at `http://localhost:8000`

### Multiple LLM Servers

1. Add multiple servers with different models:
   ```bash
   python main.py add-server --name "Fast Model" --endpoint "http://server1:1234/v1" --model "fast"
   python main.py add-server --name "Accurate Model" --endpoint "http://server2:5678/v1" --model "accurate"
   ```
2. Start hosting API
3. Choose which model to use per request by setting the `model` field

### Production Deployment

1. Set up environment variables:
   ```bash
   export LLM_API_KEY="your-production-key"
   export LLM_LOG_LEVEL="WARNING"
   export LLM_DATA_FILE="/etc/llm-hosting/servers.json"
   ```
2. Start server on production port:
   ```bash
   python main.py start --port 80 --host 0.0.0.0
   ```
3. Use process manager like systemd or supervisor to keep it running