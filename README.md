# LLM Endpoint Hosting

A Python console application for hosting LLM endpoints with OpenAI-compatible API.

## ✨ Features

- 🖥️ CLI for adding/removing LLM servers
- 🔌 OpenAI-compatible REST API (chat completions, embeddings, rerank)
- 💾 JSON persistence for server configurations
- 🔐 Optional API key authentication
- 🚀 Async request proxying
- 📝 Comprehensive logging

## 📦 Installation

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Setup

```bash
# Clone or navigate to the repository
cd py_llm_hosting

# Create virtual environment (if not exists)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### CLI Commands

#### 1. Add a Server

```bash
python main.py add-server \
  --name "My LLM" \
  --endpoint "http://localhost:1234/v1/chat/completions" \
  --model "llama-2"
```

#### 2. List Servers

```bash
python main.py list-servers
```

Output:
```
ID: 5cab8af7-1240-45b4-a58f-5481289f40e0
Name: Test Server
Endpoint: http://localhost:1234/v1/chat/completions
Model: test-model
Status: inactive
---
```

#### 3. Remove a Server

```bash
python main.py remove-server --id <server-id>
```

#### 4. Start API Server

```bash
python main.py start --port 8000 --host 0.0.0.0
```

Options:
- `--port`: Port to run on (default: 8000)
- `--host`: Host to bind to (default: 0.0.0.0)

### API Usage

#### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }'
```

#### Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "input": "Hello, world!"
  }'
```

#### Rerank

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "query": "What is AI?",
    "documents": [
      "Artificial Intelligence is a field of computer science",
      "Machine learning is a subset of AI"
    ]
  }'
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔧 Configuration

### Environment Variables

Set these environment variables to configure the application:

```bash
export LLM_API_KEY=your-secret-key  # Optional API key for authentication
export LLM_LOG_LEVEL=INFO           # Logging level (DEBUG, INFO, WARNING, ERROR)
export LLM_DATA_FILE=servers.json   # Path to data file
```

### API Key Authentication

If `LLM_API_KEY` is set, all API requests must include:

```bash
curl -H "Authorization: Bearer your-secret-key" ...
```

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Contract tests
pytest tests/contract/
```

## 🛠️ Development

### Code Formatting

```bash
# Format code with black
black .

# Lint with ruff
ruff check .
```

### Project Structure

```
py_llm_hosting/
├── src/
│   ├── api/              # FastAPI endpoints
│   │   ├── chat.py       # Chat completions
│   │   ├── embeddings.py # Embeddings
│   │   └── ranking.py    # Reranking
│   ├── cli/              # CLI commands
│   │   └── commands.py
│   ├── lib/              # Utilities
│   │   ├── config.py     # Configuration
│   │   ├── logging.py    # Logging setup
│   │   ├── persistence.py # JSON storage
│   │   └── formatters.py # Response formatting
│   ├── models/           # Data models
│   │   ├── server.py     # LLM Server model
│   │   └── mcp.py        # MCP Context model
│   └── services/         # Business logic
│       ├── proxy.py      # Request proxying
│       └── tools.py      # Tool handling
├── tests/
│   ├── contract/         # API contract tests
│   ├── integration/      # Integration tests
│   └── unit/             # Unit tests
├── main.py               # Entry point
└── requirements.txt      # Dependencies
```

## 📝 Data Storage

Server configurations are stored in `servers.json`:

```json
{
  "servers": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Local LM Studio",
      "endpoint_url": "http://localhost:1234/v1/chat/completions",
      "model_name": "llama-2-7b",
      "status": "inactive",
      "config": {},
      "created_at": "2025-11-05T10:00:00Z",
      "updated_at": "2025-11-05T10:00:00Z"
    }
  ]
}
```

## 🐛 Troubleshooting

### Server Not Found Error

If you get "Model 'xyz' not found", ensure you've added the server:

```bash
python main.py list-servers
```

### Connection Errors

Verify the LLM server is running and accessible:

```bash
curl http://localhost:1234/v1/chat/completions
```

### Import Errors

Ensure you're using the virtual environment:

```bash
source .venv/bin/activate
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code is formatted with `black`
- Tests pass with `pytest`
- Linting passes with `ruff check .`