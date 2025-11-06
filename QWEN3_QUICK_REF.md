# Quick Reference: Qwen3 8B 4-bit vLLM Server

## Server Info

```
Name:       Qwen3 8B 4bit vLLM
Model:      qwen3-8b-vllm
ID:         vllm-qwen3-8b-4bit
Backend:    vLLM
HF Model:   Qwen/Qwen2.5-7B-Instruct-AWQ
Status:     Stopped (ready to start)
```

## Quick Start

```bash
# 1. Install dependencies
pip install vllm autoawq transformers

# 2. Start server (if using CLI)
python main.py server start vllm-qwen3-8b-4bit

# 3. Test it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-vllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Requirements

- ✅ NVIDIA GPU with CUDA 11.8+
- ✅ 8GB+ VRAM (12GB recommended)
- ✅ vLLM + AWQ libraries

## Key Features

- 🚀 **4-bit AWQ quantization** - Only ~4-6GB VRAM
- ⚡ **High performance** - 40-120+ tokens/sec
- 🎯 **Production ready** - vLLM optimizations
- 🌍 **Multilingual** - English, Chinese, and more
- 📝 **Long context** - Up to 4096 tokens (configurable)

## Test Commands

```bash
# Basic test
./examples/test_vllm.py --model qwen3-8b-vllm

# Streaming test
./examples/test_vllm_streaming.py --model qwen3-8b-vllm

# Custom test
curl localhost:8000/v1/chat/completions \
  -d '{"model":"qwen3-8b-vllm","messages":[{"role":"user","content":"Hi"}]}'
```

## Configuration

Located in `servers.json`:

```json
{
  "tensor_parallel_size": 1,      // Number of GPUs
  "gpu_memory_utilization": 0.9,  // GPU memory %
  "max_model_len": 4096,           // Context length
  "quantization": "awq"            // 4-bit AWQ
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `gpu_memory_utilization` to 0.7 |
| Slow inference | Check `nvidia-smi`, close other GPU apps |
| Can't load model | Install: `pip install vllm autoawq` |
| Download fails | Set `HF_TOKEN` environment variable |

## Performance Tips

**Lower memory:**
- Set `gpu_memory_utilization`: 0.7
- Set `max_model_len`: 2048

**Higher throughput:**
- Set `tensor_parallel_size`: 2 (use 2 GPUs)
- Set `gpu_memory_utilization`: 0.95

**Longer context:**
- Set `max_model_len`: 8192
- Increase VRAM allocation

## More Info

See `QWEN3_8B_VLLM_SETUP.md` for detailed documentation.
