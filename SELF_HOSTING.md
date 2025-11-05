# Self-Hosting Models Guide

## Overview

The LLM Endpoint Hosting system now supports **two modes**:

1. **Proxy Mode**: Forward requests to external LLM servers (original functionality)
2. **Self-Hosted Mode**: Download, load, and run models locally with multiple backend options

## Self-Hosted Backends

### Option 1: llama-cpp (Recommended for CPU)
- ✅ Efficient CPU/GPU inference
- ✅ Supports GGUF quantized models
- ✅ Low memory footprint
- ✅ Fast inference on CPU
- 📦 Models: Any GGUF format from HuggingFace

### Option 2: Transformers (Recommended for GPU)
- ✅ Wide model support
- ✅ Full HuggingFace ecosystem
- ✅ 4-bit and 8-bit quantization
- ✅ Multiple architectures
- 📦 Models: Any HuggingFace model

## Quick Start - Self-Hosted Models

### 1. Install Dependencies

```bash
# For llama-cpp backend
pip install llama-cpp-python huggingface-hub

# For transformers backend (includes GPU support)
pip install transformers torch accelerate bitsandbytes
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### 2. Add a Self-Hosted Model

#### Using llama-cpp (GGUF models):

```bash
python main.py add-server \
  --name "Local Llama" \
  --model "llama-2-7b" \
  --mode self-hosted \
  --model-path "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf" \
  --backend llama-cpp \
  --gpu-layers 0
```

#### Using Transformers:

```bash
python main.py add-server \
  --name "Local Mistral" \
  --model "mistral-7b" \
  --mode self-hosted \
  --model-path "mistralai/Mistral-7B-Instruct-v0.1" \
  --backend transformers \
  --load-in-4bit
```

### 3. Start the API Server

```bash
python main.py start --port 8000
```

The server will automatically:
- Download the model from HuggingFace (first time)
- Load it into memory
- Make it available via OpenAI-compatible API

### 4. Use the Model

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ]
  }'
```

## Popular Models

### llama-cpp Backend (GGUF)

```bash
# Llama 2 7B (4-bit quantized)
python main.py add-server \
  --name "Llama-2-7B" \
  --model "llama-2-7b" \
  --mode self-hosted \
  --model-path "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf" \
  --backend llama-cpp

# Mistral 7B (4-bit quantized)
python main.py add-server \
  --name "Mistral-7B" \
  --model "mistral-7b" \
  --mode self-hosted \
  --model-path "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf" \
  --backend llama-cpp

# Phi-2 (2.7B - fast on CPU)
python main.py add-server \
  --name "Phi-2" \
  --model "phi-2" \
  --mode self-hosted \
  --model-path "TheBloke/phi-2-GGUF/phi-2.Q4_K_M.gguf" \
  --backend llama-cpp
```

### Transformers Backend

```bash
# Mistral 7B with 4-bit quantization
python main.py add-server \
  --name "Mistral-7B-4bit" \
  --model "mistral-7b" \
  --mode self-hosted \
  --model-path "mistralai/Mistral-7B-Instruct-v0.1" \
  --backend transformers \
  --load-in-4bit

# Llama 2 13B with 8-bit quantization
python main.py add-server \
  --name "Llama-2-13B-8bit" \
  --model "llama-2-13b" \
  --mode self-hosted \
  --model-path "meta-llama/Llama-2-13b-chat-hf" \
  --backend transformers \
  --load-in-8bit

# Phi-2 (smaller, faster)
python main.py add-server \
  --name "Phi-2" \
  --model "phi-2" \
  --mode self-hosted \
  --model-path "microsoft/phi-2" \
  --backend transformers
```

## Backend Comparison

| Feature | llama-cpp | Transformers |
|---------|-----------|--------------|
| CPU Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| GPU Performance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Model Support | GGUF only | All HF models |
| Memory Usage | ⭐⭐⭐⭐⭐ (Low) | ⭐⭐⭐ (Higher) |
| Setup Complexity | Easy | Medium |
| Quantization | Built-in | Requires bitsandbytes |

## Configuration Options

### llama-cpp Backend

```bash
--backend llama-cpp \
--gpu-layers 32        # Number of layers to offload to GPU (0 = CPU only)
```

### Transformers Backend

```bash
--backend transformers \
--load-in-4bit         # 4-bit quantization (lowest memory)
--load-in-8bit         # 8-bit quantization (balanced)
# (no flags = full precision)
```

## Mixed Mode Setup

You can run both proxy and self-hosted models simultaneously:

```bash
# Add a self-hosted model
python main.py add-server \
  --name "Local Mistral" \
  --model "mistral-local" \
  --mode self-hosted \
  --model-path "mistralai/Mistral-7B-Instruct-v0.1" \
  --backend transformers

# Add a proxy to external API
python main.py add-server \
  --name "OpenAI GPT" \
  --model "gpt-3.5-turbo" \
  --mode proxy \
  --endpoint "https://api.openai.com/v1/chat/completions"

# Start server - both models available
python main.py start
```

## Performance Tips

### For CPU-Only Systems
1. Use llama-cpp backend
2. Choose 4-bit quantized GGUF models
3. Use smaller models (7B or less)
4. Set `--gpu-layers 0`

### For GPU Systems
1. Use transformers backend with quantization
2. Set `--load-in-4bit` for large models
3. Monitor GPU memory usage
4. Consider model size vs GPU VRAM

### Memory Requirements

| Model Size | 4-bit GGUF | 8-bit | Full Precision |
|------------|------------|-------|----------------|
| 7B | ~4 GB | ~7 GB | ~14 GB |
| 13B | ~8 GB | ~13 GB | ~26 GB |
| 70B | ~35 GB | ~70 GB | ~140 GB |

## Troubleshooting

### Model Download Issues
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Use HuggingFace CLI to pre-download
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF
```

### Out of Memory
- Use smaller quantized models (Q4_K_M)
- Reduce context window (`--ctx-size`)
- Try llama-cpp instead of transformers
- Close other applications

### Slow Performance
- Enable GPU layers: `--gpu-layers 32`
- Use quantized models
- Reduce batch size
- Try llama-cpp for CPU inference

## Advanced Usage

### Custom Backend Configuration

You can pass additional configuration via the config file or API:

```python
# In your application
config = {
    "n_ctx": 4096,  # Larger context window
    "n_batch": 512,  # Batch size
    "temperature": 0.7,
}
```

## Next Steps

1. Try different models from HuggingFace
2. Experiment with quantization levels
3. Benchmark performance on your hardware
4. Set up model caching for faster restarts
5. Configure auto-loading of frequently used models