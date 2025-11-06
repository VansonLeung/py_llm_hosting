# vLLM Backend Testing

This directory contains comprehensive tests for the vLLM backend implementation.

## Test Files

### 1. `test_vllm.py`
Comprehensive test suite covering:
- ✅ Non-streaming chat completions
- ✅ Streaming chat completions
- ✅ Multi-turn conversations
- ✅ Temperature variations
- ✅ Performance metrics
- ✅ Token usage tracking

### 2. `test_vllm_streaming.py`
Focused streaming tests including:
- ✅ Basic streaming
- ✅ Streaming vs non-streaming comparison
- ✅ Long response streaming
- ✅ Rapid-fire streaming requests
- ✅ Streaming with system messages
- ✅ Time-to-first-token (TTFT) metrics
- ✅ Chunk-by-chunk delivery

## Prerequisites

### 1. Install vLLM
```bash
pip install vllm transformers
```

**Requirements:**
- CUDA-compatible GPU (NVIDIA)
- CUDA 11.8 or newer
- Python 3.8-3.11
- Sufficient GPU memory for your model

### 2. Start the Server with vLLM Backend

#### Using a HuggingFace Model
```bash
# Add a model with vLLM backend
python main.py server add vllm-gpt2 \
  --endpoint-url "self-hosted" \
  --model-path "gpt2" \
  --backend vllm \
  --tensor-parallel 1

# Start the server
python main.py server start vllm-gpt2
```

#### For Larger Models (Multi-GPU)
```bash
# Add a larger model with tensor parallelism
python main.py server add vllm-llama \
  --endpoint-url "self-hosted" \
  --model-path "meta-llama/Llama-2-7b-chat-hf" \
  --backend vllm \
  --tensor-parallel 2 \
  --max-model-len 4096

# Start the server
python main.py server start vllm-llama
```

#### Configuration Options
- `--tensor-parallel`: Number of GPUs for tensor parallelism (default: 1)
- `--max-model-len`: Maximum sequence length (default: auto-detected)
- `--gpu-memory-utilization`: GPU memory fraction (default: 0.9)
- `--dtype`: Data type (auto, float16, bfloat16, float32)

## Running the Tests

### Basic Test Suite
```bash
# Run all vLLM tests (default model: gpt2)
./examples/test_vllm.py

# Test with a specific model
./examples/test_vllm.py --model vllm-gpt2

# Test with custom port
./examples/test_vllm.py --port 8080

# Test with both custom model and port
./examples/test_vllm.py --model my-vllm-model --port 8080
```

### Streaming-Focused Tests
```bash
# Run streaming tests (default model: gpt2)
./examples/test_vllm_streaming.py

# Test streaming with specific model
./examples/test_vllm_streaming.py --model vllm-llama

# Test with custom port
./examples/test_vllm_streaming.py --port 8080
```

## Test Output Examples

### Non-Streaming Test
```
======================================================================
Test: Non-Streaming Chat Completion
Model: gpt2
Backend: vLLM
Prompt: Write a haiku about artificial intelligence
======================================================================

Response: Silicon minds dream,
          Algorithms dance in code,
          Future blooms in bits.

Metadata:
  Finish Reason: stop
  Tokens: 15 prompt + 24 completion = 39 total
  Time: 0.45s
  Tokens/sec: 53.33
```

### Streaming Test
```
======================================================================
Model: gpt2
Backend: vLLM (streaming)
Max Tokens: 150
======================================================================
User: Tell me a short story about a robot learning to paint
Assistant: In a small workshop, a robot named Pixel discovered colors...

======================================================================
Streaming Metrics:
  Total chunks: 47
  Time to first token (TTFT): 0.123s
  Total time: 1.234s
  Generation time: 1.111s
  Avg time per chunk: 0.024s
  Finish reason: stop
======================================================================
```

## Performance Characteristics

### vLLM Advantages
1. **High Throughput**: Optimized for serving many requests
2. **PagedAttention**: Efficient memory management
3. **Continuous Batching**: Dynamic request batching
4. **Multi-GPU Support**: Tensor and pipeline parallelism
5. **Low Latency**: Fast token generation

### Typical Metrics
- **TTFT** (Time to First Token): 50-200ms
- **Token Generation**: 20-100+ tokens/sec (depends on GPU)
- **Memory Efficiency**: ~50% better than naive implementations

## Streaming Implementation Details

The vLLM backend implements streaming in `src/backends/vllm_backend.py`:

```python
async def generate_chat(self, messages, max_tokens, temperature, stream=False):
    if stream:
        async def stream_wrapper():
            # Generate full response
            outputs = await loop.run_in_executor(
                None,
                lambda: self.llm.generate([prompt], sampling_params)
            )
            
            # Simulate streaming by yielding words
            # Note: For production, consider AsyncLLMEngine for true streaming
            words = generated_text.split()
            for word in words:
                yield word
                await asyncio.sleep(0.01)
        
        return stream_wrapper()
```

### Current Implementation
- ✅ Simulated streaming (word-by-word)
- ✅ Compatible with OpenAI streaming format
- ✅ Async/await support
- ⚠️ Uses `LLM` class (batch mode)

### Production Streaming
For true token-by-token streaming in production:
1. Use `AsyncLLMEngine` instead of `LLM`
2. Implement `AsyncEngine.generate()` with async iteration
3. Benefits: True streaming, better resource utilization

Example:
```python
from vllm import AsyncLLMEngine, AsyncEngineArgs

async def true_streaming():
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(...))
    async for output in engine.generate(prompt, sampling_params, request_id):
        yield output.outputs[0].text
```

## Troubleshooting

### GPU Memory Issues
```bash
# Reduce GPU memory utilization
python main.py server add model \
  --backend vllm \
  --gpu-memory-utilization 0.7
```

### Model Loading Errors
```bash
# Check vLLM installation
python -c "import vllm; print(vllm.__version__)"

# Verify CUDA
nvidia-smi

# Check model path
ls ~/.cache/huggingface/hub/
```

### Connection Errors
```bash
# Verify server is running
curl http://localhost:8000/health

# Check server logs
python main.py server list

# Test with different port
./examples/test_vllm.py --port 8001
```

### Slow Performance
1. **Check GPU utilization**: `nvidia-smi -l 1`
2. **Reduce max_model_len**: Free up memory
3. **Increase tensor_parallel**: Use multiple GPUs
4. **Enable FP16/BF16**: Faster computation

## Comparison with Other Backends

| Feature | vLLM | llama.cpp | MLX | Transformers |
|---------|------|-----------|-----|--------------|
| **GPU Support** | ✅ NVIDIA | ✅ NVIDIA/AMD/Metal | ✅ Apple Silicon | ✅ NVIDIA/AMD |
| **Streaming** | ✅ | ✅ | ✅ | ✅ |
| **Multi-GPU** | ✅ Native | ❌ | ❌ | ⚠️ Limited |
| **Throughput** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Memory Efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Quantization** | ✅ | ✅ GGUF | ✅ Native | ⚠️ Some |
| **Best For** | Production serving | CPU inference | Mac M-series | Research/Dev |

## Integration Examples

### Python Client
```python
import requests

def chat_with_vllm(prompt: str, stream: bool = True):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream
        },
        stream=stream
    )
    
    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
    else:
        print(response.json())
```

### cURL
```bash
# Non-streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }' \
  --no-buffer
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [HuggingFace Models](https://huggingface.co/models)

## Contributing

To add more tests:
1. Follow existing test patterns in `test_vllm.py`
2. Use descriptive test names
3. Include performance metrics
4. Document expected behavior
5. Handle errors gracefully

## Support

For issues:
1. Check [vLLM Issues](https://github.com/vllm-project/vllm/issues)
2. Verify GPU compatibility
3. Review server logs
4. Test with smaller models first
