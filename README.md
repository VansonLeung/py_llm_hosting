# 🚀 Self-Hosted LLM Endpoint Platform

A production-ready FastAPI-based platform for self-hosting Large Language Model endpoints with OpenAI-compatible APIs. Run your own LLMs locally with support for multiple backends, streaming, embeddings, and document reranking.

## ✨ Features

- **🔌 OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **💬 Chat Completions**: Streaming and non-streaming modes
- **📊 Embeddings**: Generate text embeddings for semantic search
- **🔍 Document Reranking**: Cross-encoder based relevance ranking
- **🖼️ Vision Chat**: MLX-VLM backend for multimodal prompts (text + image) on Apple Silicon
- **🎯 Multiple Backends**: Support for llama-cpp, MLX, Transformers, vLLM, and reranker models
- **⚡ Streaming Support**: Server-Sent Events (SSE) for real-time responses
- **�️ Apple Silicon Optimized**: Native MLX backend for M-series chips
- **📦 Easy Management**: CLI tools for server configuration and model management
- **� Status Tracking**: Real-time monitoring of model states
- **🌐 Proxy Mode**: Route requests to external API services

## 🛠️ Supported Backends

| Backend | Description | Best For |
|---------|-------------|----------|
| **llama-cpp** | GGUF quantized models | CPU/GPU inference, low memory |
| **MLX** | Apple Silicon optimized | M1/M2/M3 Macs, fastest on Apple Silicon |
| **MLX-VLM** | Apple Silicon multimodal (text + vision) | On-device image reasoning |
| **Transformers** | HuggingFace models | Wide model compatibility |
| **vLLM** | High-performance inference | GPU clusters, high throughput |
| **Reranker** | Cross-encoder models | Document reranking, semantic search |

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip
- Virtual environment (recommended)
- For MLX: Apple Silicon Mac (M1/M2/M3)
- For GPU acceleration: CUDA-compatible GPU (optional)

### Setup

```bash
# Clone or navigate to the repository
cd py_llm_hosting

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Optional Backend Installation

#### For Apple Silicon (MLX):
```bash
pip install mlx==0.20.0 mlx-lm==0.19.3
```

#### For Vision-Language Models (MLX-VLM):
```bash
pip install mlx-vlm pillow
```
*Requires Apple Silicon (M-series) hardware.*

#### For GPU Inference (vLLM):
```bash
# Install vLLM for high-performance GPU inference
pip install vllm

# For specific version
pip install vllm==0.6.3

# Note: Requires CUDA-compatible GPU and CUDA toolkit installed
# Supports AWQ quantization for 4-bit models
```

#### For Sentence Transformers (Embeddings & Reranking):
```bash
pip install sentence-transformers torch
```

## 🚀🚀🚀 Quickest Start

### 1. Copy the servers.json Example
```bash
cp servers.json.example servers.json
```

### 1.5. Configure Environment (Optional)
Create a `.env` file to customize the server port (defaults to 8000):

```bash
# Create .env file
echo "PORT=8000" > .env

# Or edit manually
nano .env
```

The `.env` file supports:
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)

**Note**: Test scripts in the `examples/` folder also respect the `PORT` environment variable and can be overridden with `--port` flag.

### 2. Start the Server

```bash
python main.py start
```

### 3. List Configured Servers

```bash
python main.py list-servers
```

Output:
```
ID: eb75e115-8421-408b-ac16-eaa97fed0727
Name: TinyLlama Chat
Model: tinyllama
Mode: self-hosted
Backend: llama-cpp
Status: active
---
```

### 4. Run test requests (see Examples in EXAMPLES.md)

```bash
python examples/test_all_endpoints.py
```

Output:
```
======================================================================
                  COMPREHENSIVE API ENDPOINT TESTING                  
======================================================================

ℹ Testing server at: http://localhost:8000/v1
ℹ Test started at: 2025-11-05 22:22:49


======================================================================
                           CHAT COMPLETIONS                           
======================================================================

────────────────────────────────────────────────────────────
TEST: llama-cpp Chat (Non-Streaming)
────────────────────────────────────────────────────────────
🔧 Model: tinyllama (llama-cpp backend)
✓ Response: 2 + 2 = 5
ℹ Tokens used: 36

────────────────────────────────────────────────────────────
TEST: llama-cpp Chat (Streaming)
────────────────────────────────────────────────────────────
🔧 Model: tinyllama (llama-cpp backend)
✓ Received 8 chunks
ℹ Response: 2 + 2 = 4.

────────────────────────────────────────────────────────────
TEST: MLX Chat (Non-Streaming)
────────────────────────────────────────────────────────────
🔧 Model: qwen2.5-0.5b (MLX backend)
✓ Response: C++
ℹ Tokens used: 40

────────────────────────────────────────────────────────────
TEST: MLX Chat (Streaming)
────────────────────────────────────────────────────────────
🔧 Model: qwen2.5-0.5b (MLX backend)
✓ Received 1 chunks
ℹ Response: Java

======================================================================
                              EMBEDDINGS                              
======================================================================

────────────────────────────────────────────────────────────
TEST: llama-cpp Embeddings
────────────────────────────────────────────────────────────
🔧 Model: nomic-embed-v1.5 (llama-cpp backend)
✓ Generated embedding with dimension: 768
ℹ First 5 values: ['0.5825', '1.1363', '-3.5205', '-0.1286', '0.9406']

────────────────────────────────────────────────────────────
TEST: MLX Embeddings
────────────────────────────────────────────────────────────
🔧 Model: qwen3-embed (MLX backend)
✓ Generated embedding with dimension: 1024
ℹ First 5 values: ['0.0115', '-0.0432', '-0.0479', '0.0679', '-0.0393']

────────────────────────────────────────────────────────────
TEST: Sentence-Transformers Embeddings
────────────────────────────────────────────────────────────
🔧 Model: bge-micro-v2 (sentence-transformers backend)
✓ Generated 3 embeddings
ℹ Dimension: 384
ℹ Similarity between related texts: 0.8544

======================================================================
                              RERANKING                               
======================================================================

────────────────────────────────────────────────────────────
TEST: Document Reranking
────────────────────────────────────────────────────────────
🔧 Model: bge-reranker-v2-m3 (reranker backend)
✓ Reranked 4 documents
ℹ Top 3 results:
      1. Doc 0 (Score: 0.9994): Machine learning is a subset of AI focused on algo...
      2. Doc 2 (Score: 0.0016): Deep learning uses neural networks with multiple l...
      3. Doc 3 (Score: 0.0000): I like to eat pizza for dinner....

======================================================================
                             TEST SUMMARY                             
======================================================================

Chat Completions:
  llama-cpp Chat (Non-Streaming): PASS
  llama-cpp Chat (Streaming): PASS
  MLX Chat (Non-Streaming): PASS
  MLX Chat (Streaming): PASS

Embeddings:
  llama-cpp Embeddings: PASS
  MLX Embeddings: PASS
  Sentence-Transformers Embeddings: PASS

Reranking:
  Document Reranking: PASS

──────────────────────────────────────────────────────────────────────
Total: 8/8 tests passed

======================================================================
                         ✓ ALL TESTS PASSED!                          
======================================================================

```


## 🚀 Quick Start

### 1. Download a Model

```bash
# Download a small chat model (GGUF format)
python main.py download-model \
  --model-id TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --filename tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### 2. Add Server Configuration

```bash
# Add a self-hosted llama-cpp server
python main.py add-server \
  --name "TinyLlama Chat" \
  --model "tinyllama" \
  --mode self-hosted \
  --backend llama-cpp \
  --model-path ~/.cache/py_llm_hosting/models/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/.../tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Or add an MLX server (for Apple Silicon)
python main.py add-server \
  --name "Qwen MLX" \
  --model "qwen2.5-0.5b" \
  --mode self-hosted \
  --backend mlx \
  --model-path "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Or add an MLX-VLM server for multimodal chat
python main.py add-server \
  --name "Chandra Vision" \
  --model "chandra-4bit" \
  --mode self-hosted \
  --backend mlx-vlm \
  --model-path "mlx-community/chandra-4bit"

# Or add a proxy server (external API)
python main.py add-server \
  --name "OpenAI GPT-4" \
  --model "gpt-4" \
  --mode proxy \
  --endpoint "https://api.openai.com/v1/chat/completions"
```

### 3. Start the Server

```bash
python main.py start
```

The API will be available at `http://localhost:8000`

### 4. List Configured Servers

```bash
python main.py list-servers
```

Output:
```
ID: eb75e115-8421-408b-ac16-eaa97fed0727
Name: TinyLlama Chat
Model: tinyllama
Mode: self-hosted
Backend: llama-cpp
Status: active
---
```

## 📚 Usage Examples

### Chat Completion (Non-Streaming)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "tinyllama",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Chat Completion (Streaming)

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "tinyllama",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5"}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data != '[DONE]':
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
```

### Vision Chat With Images (MLX-VLM)

```python
import base64
from pathlib import Path
import httpx

image_bytes = Path("samples/cat.png").read_bytes()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

response = httpx.post(
  "http://localhost:8000/v1/chat/completions",
  json={
    "model": "chandra-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image"},
          {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
          }
        ]
      }
    ]
  },
  timeout=120.0
)

print(response.json()["choices"][0]["message"]["content"])
```

> Tip: Run `python examples/test_mlx_vlm.py --image <path>` for a ready-made CLI tester.

### Generate Embeddings

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    headers={"Content-Type": "application/json"},
    json={
        "model": "nomic-embed",
        "input": "Machine learning is awesome!"
    }
)

embedding = response.json()["data"][0]["embedding"]
print(f"Embedding dimension: {len(embedding)}")
```

### Rerank Documents

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/rerank",
    headers={"Content-Type": "application/json"},
    json={
        "model": "bge-reranker-v2-m3",
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of artificial intelligence",
            "The weather is sunny today",
            "Deep learning uses neural networks"
        ],
        "top_n": 2
    }
)

results = response.json()["results"]
for result in results:
    print(f"Score: {result['relevance_score']:.4f} - {result['document']}")
```

## 🔧 CLI Commands

### Server Management

```bash
# List all configured servers
python main.py list-servers

# Add a self-hosted server
python main.py add-server \
  --name "My Model" \
  --model "model-name" \
  --mode self-hosted \
  --backend llama-cpp \
  --model-path /path/to/model

# Add a proxy server
python main.py add-server \
  --name "External API" \
  --model "gpt-4" \
  --mode proxy \
  --endpoint "https://api.example.com/v1/chat/completions"

# Remove a server
python main.py remove-server --name "My Model"

# Start the API server (uses PORT from .env or defaults to 8000)
python main.py start
```

### Model Management

```bash
# Download a model from HuggingFace
python main.py download-model --model-id <model-id> --filename <filename>

# List loaded models
python main.py list-loaded

# Unload a model
python main.py unload-model --name "model-name"
```

### Backend-Specific Options

#### llama-cpp
```bash
python main.py add-server \
  --backend llama-cpp \
  --gpu-layers 35  # Number of layers to offload to GPU
```

#### Transformers
```bash
python main.py add-server \
  --backend transformers \
  --load-in-4bit   # Use 4-bit quantization
  # or
  --load-in-8bit   # Use 8-bit quantization
```

#### vLLM
```bash
python main.py add-server \
  --backend vllm \
  --tensor-parallel 2  # Tensor parallelism size
```

## 🎯 Example Model Configurations

### Chat Models

#### TinyLlama (GGUF) - Good for Testing
```bash
# Download
python main.py download-model \
  --model-id TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --filename tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Add server
python main.py add-server \
  --name "TinyLlama" \
  --model "tinyllama" \
  --mode self-hosted \
  --backend llama-cpp \
  --model-path ~/.cache/py_llm_hosting/models/.../tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

#### Qwen 2.5 (MLX) - Apple Silicon Optimized
```bash
python main.py add-server \
  --name "Qwen MLX" \
  --model "qwen2.5-0.5b" \
  --mode self-hosted \
  --backend mlx \
  --model-path "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

#### Qwen 2.5 (vLLM) - High-Performance GPU Inference
```bash
# vLLM automatically downloads models from HuggingFace
python main.py add-server \
  --name "Qwen 2.5 vLLM" \
  --model "qwen2.5-0.5b-vllm" \
  --mode self-hosted \
  --backend vllm \
  --model-path "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
# Note: AWQ quantized models are recommended for reduced memory usage
```

### Embedding Models

#### Nomic Embed (GGUF)
```bash
# Download
python main.py download-model \
  --model-id nomic-ai/nomic-embed-text-v1.5-GGUF \
  --filename nomic-embed-text-v1.5.Q4_K_M.gguf

# Add server
python main.py add-server \
  --name "Nomic Embed" \
  --model "nomic-embed" \
  --mode self-hosted \
  --backend llama-cpp \
  --model-path ~/.cache/py_llm_hosting/models/.../nomic-embed-text-v1.5.Q4_K_M.gguf
```

### Reranker Models

#### BGE Reranker v2-m3
```bash
# Download
python main.py download-model \
  --model-id BAAI/bge-reranker-v2-m3

# Add server
python main.py add-server \
  --name "BGE Reranker" \
  --model "bge-reranker-v2-m3" \
  --mode self-hosted \
  --backend reranker \
  --model-path ~/.cache/py_llm_hosting/models/.../bge-reranker-v2-m3
```

## 🏗️ Architecture

```
py_llm_hosting/
├── src/
│   ├── api/                 # FastAPI endpoints
│   │   ├── chat.py         # Chat completions
│   │   ├── embeddings.py   # Embedding generation
│   │   └── ranking.py      # Document reranking
│   ├── backends/           # Model backend implementations
│   │   ├── llama_cpp_backend.py
│   │   ├── mlx_backend.py
│   │   ├── transformers_backend.py
│   │   ├── vllm_backend.py
│   │   └── reranker_backend.py
│   ├── cli/                # CLI commands
│   │   └── commands.py
│   ├── models/             # Data models
│   │   ├── server.py       # Server configuration
│   │   └── backend.py      # Backend types
│   ├── services/           # Business logic
│   │   ├── model_manager.py # Model lifecycle
│   │   └── proxy.py        # Request proxying
│   └── lib/                # Utilities
│       ├── persistence.py  # JSON storage
│       ├── formatters.py   # Response formatting
│       └── config.py       # Configuration
├── examples/               # Example scripts
│   ├── test_all_endpoints.py
│   ├── test_embeddings.py
│   └── test_ranking.py
├── tests/                  # Unit tests
├── main.py                # Entry point
└── requirements.txt       # Dependencies
```

## 🔌 API Endpoints

### Chat Completions
- **POST** `/v1/chat/completions`
- Compatible with OpenAI's chat completion API
- Supports streaming via `stream=true`
- Parameters: `model`, `messages`, `temperature`, `max_tokens`, `stream`

### Embeddings
- **POST** `/v1/embeddings`
- Generate vector embeddings from text
- Supports batch processing
- Parameters: `model`, `input`

### Reranking
- **POST** `/v1/rerank`
- Rerank documents by relevance to a query
- Returns sorted results with relevance scores
- Parameters: `model`, `query`, `documents`, `top_n`

### Health Check
- **GET** `/health`
- Check API server status

### API Documentation
Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## � Performance Tips

1. **Use Quantized Models**: Q4_K_M or Q8 GGUF models offer good quality/size trade-off
2. **Enable GPU Acceleration**: Use `--gpu-layers` for llama-cpp or GPU-enabled backends
3. **Apple Silicon**: Always use MLX backend for best performance on M-series Macs
4. **Batch Requests**: Use batch embeddings for processing multiple texts
5. **Model Selection**: Smaller models (1-3B params) are often sufficient for many tasks

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test all endpoints (uses PORT from .env or defaults to 8000)
python examples/test_all_endpoints.py

# Test with custom port
python examples/test_all_endpoints.py --port 3000

# Test embeddings
python examples/test_embeddings.py

# Test reranking
python examples/test_ranking.py

# Test streaming
python examples/test_streaming.py

# Run unit tests
pytest
```

**Note**: All test scripts in the `examples/` folder support the `--port` flag to override the port specified in `.env`.

## 🔧 Configuration

### Environment Variables (.env file)

Create a `.env` file in the project root to configure the server:

```bash
# Server configuration
PORT=8000                    # Server port (default: 8000)

# Optional: Logging and data configuration
LOG_LEVEL=INFO               # Logging level (DEBUG, INFO, WARNING, ERROR)
```

**Note**: The `PORT` environment variable is automatically loaded from `.env` and used by both the server and test scripts. Test scripts can override this with the `--port` flag.

### Data Storage

Server configurations are stored in `servers.json`:

```json
{
  "servers": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "TinyLlama Chat",
      "model_name": "tinyllama",
      "mode": "self-hosted",
      "backend_type": "llama-cpp",
      "model_path": "/path/to/model.gguf",
      "status": "active",
      "config": {},
      "created_at": "2025-11-05T10:00:00Z",
      "updated_at": "2025-11-05T10:00:00Z"
    }
  ]
}
```

## 🐛 Troubleshooting

### Model Not Loading
- Check that the model path is correct using `python main.py list-servers`
- Ensure sufficient memory/VRAM available
- Verify backend dependencies are installed
- Check logs for detailed error messages

### Slow Performance
- For Apple Silicon, use MLX backend for best performance
- For NVIDIA GPUs, enable GPU layers: `--gpu-layers 35`
- Consider using smaller quantized models (Q4, Q8)
- Check CPU/GPU utilization

### Import Errors
- Ensure all required dependencies are installed: `pip install -r requirements.txt`
- Check that the correct Python environment is activated: `source .venv/bin/activate`
- For MLX: Requires Apple Silicon Mac
- For vLLM: Requires CUDA-compatible GPU

### Port Already in Use
- The default port 8000 might be occupied
- Stop other services using the port
- Or specify a different port (check configuration)

### Server Not Found Error
- Verify server configuration: `python main.py list-servers`
- Ensure model name matches the one you're requesting
- Check that the server was added successfully

### Connection Errors (Proxy Mode)
- Verify the external endpoint is accessible
- Check network connectivity
- Ensure API keys are configured if required

## 📖 Documentation

- [Reranker Implementation Guide](RERANKER_IMPLEMENTATION.md)
- [Testing Summary](changelogs/20251105/TESTING_SUMMARY.md)
- [API Specification](specs/001-llm-endpoint-hosting/contracts/openapi.yaml)

## 🛠️ Development

### Code Formatting

```bash
# Format code with black
black .

# Lint with ruff
ruff check .
```

### Running Tests

```bash
# Run all unit tests
pytest

# Run specific test file
pytest tests/unit/test_persistence.py

# Run with coverage
pytest --cov=src tests/
```

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code is formatted with `black`
- Tests pass with `pytest`
- Linting passes with `ruff check .`
- Add tests for new features
- Update documentation as needed

## 📄 License

MIT License

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF model support
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon optimization
- [sentence-transformers](https://www.sbert.net/) - Reranker backend
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [HuggingFace](https://huggingface.co/) - Model hosting

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Made with ❤️ for the open-source LLM community**