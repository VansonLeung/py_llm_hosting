# Qwen3 8B 4-bit with vLLM - Setup Guide

## Server Configuration

A new server has been added to support **Qwen2.5 7B Instruct** with 4-bit AWQ quantization using the vLLM backend.

### Server Details

- **Name**: Qwen3 8B 4bit vLLM
- **Model Name**: `qwen3-8b-vllm`
- **ID**: `vllm-qwen3-8b-4bit`
- **Backend**: vLLM
- **Model Path**: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Quantization**: AWQ 4-bit (efficient GPU inference)
- **Status**: Stopped (ready to start)

### Backend Configuration

```json
{
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.9,
  "max_model_len": 4096,
  "dtype": "auto",
  "trust_remote_code": false,
  "quantization": "awq"
}
```

## Prerequisites

### 1. Install vLLM with AWQ support

```bash
# Install vLLM with AWQ quantization support
pip install vllm transformers
pip install autoawq
```

### 2. GPU Requirements

- **CUDA-compatible GPU** (NVIDIA)
- **CUDA 11.8 or newer**
- **Minimum 8GB VRAM** (recommended 12GB+)
- The AWQ 4-bit quantization makes this model very memory efficient

### 3. Model Download

The model will be automatically downloaded from HuggingFace on first use:
- **Model**: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Size**: ~4-5GB (4-bit quantized)
- **Cache**: `~/.cache/huggingface/hub/`

## Usage

### Start the Server

You have two options to start the server:

#### Option 1: Using Python CLI (if dependencies are installed)

```bash
# Install dependencies first if needed
pip install -r requirements.txt

# Start the server
python main.py server start vllm-qwen3-8b-4bit
```

#### Option 2: Manual start using the API

Start your main application server, then the model will be loaded on first request.

### Test the Server

Once the server is running, you can test it:

```bash
# Using curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-vllm",
    "messages": [{"role": "user", "content": "Hello! Introduce yourself."}],
    "temperature": 0.7,
    "max_tokens": 200,
    "stream": false
  }'
```

#### Using the test scripts

```bash
# Run comprehensive tests
./examples/test_vllm.py --model qwen3-8b-vllm

# Run streaming tests
./examples/test_vllm_streaming.py --model qwen3-8b-vllm
```

#### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-8b-vllm",
        "messages": [
            {"role": "user", "content": "Write a haiku about AI"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
)

print(response.json())
```

## Performance Characteristics

### Expected Performance (depends on GPU)

- **TTFT** (Time to First Token): 50-150ms
- **Throughput**: 40-120+ tokens/sec
- **Memory Usage**: ~4-6GB VRAM (with AWQ 4-bit)
- **Context Length**: 4096 tokens (configurable via `max_model_len`)

### Advantages of AWQ 4-bit

1. **Memory Efficient**: Uses ~4GB instead of ~16GB (FP16)
2. **Fast Inference**: Minimal performance loss vs FP16
3. **High Quality**: Better than GPTQ for same bit-width
4. **vLLM Optimized**: Native support in vLLM

## Configuration Options

You can modify the backend configuration in `servers.json`:

```json
"backend_config": {
  "tensor_parallel_size": 1,        // Use multiple GPUs (1, 2, 4, 8)
  "gpu_memory_utilization": 0.9,    // GPU memory fraction (0.7-0.95)
  "max_model_len": 4096,             // Max sequence length
  "dtype": "auto",                   // Data type (auto, float16, bfloat16)
  "trust_remote_code": false,        // Trust remote code execution
  "quantization": "awq"              // Quantization method (awq, gptq)
}
```

### Tuning Tips

**For lower memory usage:**
```json
{
  "gpu_memory_utilization": 0.7,
  "max_model_len": 2048
}
```

**For multiple GPUs:**
```json
{
  "tensor_parallel_size": 2,  // Use 2 GPUs
  "gpu_memory_utilization": 0.85
}
```

**For longer contexts:**
```json
{
  "max_model_len": 8192,      // Extend to 8K tokens
  "gpu_memory_utilization": 0.95
}
```

## Troubleshooting

### Issue: Model not loading

```bash
# Check if vLLM is installed
python3 -c "import vllm; print(vllm.__version__)"

# Check AWQ support
python3 -c "import awq; print('AWQ installed')"

# Verify CUDA
nvidia-smi
```

### Issue: Out of memory

1. Reduce `gpu_memory_utilization` to 0.7 or 0.8
2. Reduce `max_model_len` to 2048
3. Make sure no other processes are using GPU

```bash
# Check GPU usage
nvidia-smi

# Kill other GPU processes if needed
# Then restart your server
```

### Issue: Slow performance

1. Check GPU utilization: `nvidia-smi -l 1`
2. Increase `gpu_memory_utilization` if you have room
3. Consider using multiple GPUs with `tensor_parallel_size`

### Issue: Model download fails

```bash
# Set HuggingFace token if model is gated
export HF_TOKEN="your_token_here"

# Or download manually
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ
```

## Model Information

### Qwen2.5-7B-Instruct-AWQ

- **Developer**: Alibaba Cloud (Qwen Team)
- **Size**: 7.6B parameters (4-bit quantized to ~4GB)
- **Context**: Up to 32K tokens (we set 4K for memory efficiency)
- **Languages**: English, Chinese, and many others
- **Strengths**: 
  - Strong reasoning capabilities
  - Excellent code generation
  - Good multilingual support
  - Fast inference with AWQ

### Capabilities

✅ Chat and instruction following  
✅ Code generation and debugging  
✅ Reasoning and problem-solving  
✅ Multilingual conversations  
✅ Function calling (with proper prompting)  
✅ Long context understanding (up to 32K)  

## Comparison with Other Backends

| Feature | vLLM + AWQ | MLX (on Mac) | llama.cpp + GGUF |
|---------|------------|--------------|------------------|
| **GPU** | NVIDIA | Apple Silicon | NVIDIA/AMD/CPU |
| **Memory** | ~4-6GB | ~4-6GB | ~4-8GB |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Multi-GPU** | ✅ | ❌ | ❌ |
| **Best For** | Production GPU | Mac development | CPU/Mixed |

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ)
- [AWQ Quantization Paper](https://arxiv.org/abs/2306.00978)
- [vLLM Test Scripts](./examples/VLLM_TESTING.md)

## Next Steps

1. **Install dependencies**: `pip install vllm autoawq transformers`
2. **Start the server**: Use your preferred method above
3. **Test it**: Run `./examples/test_vllm.py --model qwen3-8b-vllm`
4. **Monitor performance**: Use `nvidia-smi` to track GPU usage
5. **Tune configuration**: Adjust settings based on your needs

---

**Status**: ✅ Server configured and ready to start  
**Model**: Qwen2.5-7B-Instruct-AWQ (4-bit quantized)  
**Backend**: vLLM with AWQ support  
**Date Added**: 2025-11-06
