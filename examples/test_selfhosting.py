#!/usr/bin/env python3
"""
Example script demonstrating self-hosted model usage.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.server import LLMServer, ServerMode, ServerStatus
from src.services.model_manager import model_manager


async def test_llamacpp_backend():
    """Test llama-cpp backend (requires a GGUF model file)."""
    print("\n=== Testing llama-cpp Backend ===")
    
    # Create a test server config
    # Note: Replace with actual path to a GGUF model
    server = LLMServer(
        name="test-llamacpp",
        model_name="test-model",
        mode=ServerMode.SELF_HOSTED,
        model_path="path/to/model.gguf",  # Update this!
        backend_type="llama-cpp",
        backend_config={"n_gpu_layers": 0}
    )
    
    try:
        # Load the model
        print(f"Loading model from {server.model_path}...")
        backend = await model_manager.load_model(server)
        print("✓ Model loaded successfully")
        
        # Test text generation
        print("\nTesting text generation...")
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: backend.generate(
                prompt="What is the capital of France?",
                max_tokens=50
            )
        )
        print(f"Response: {response}")
        
        # Test chat
        print("\nTesting chat completion...")
        messages = [
            {"role": "user", "content": "Hello! Tell me a joke."}
        ]
        chat_response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: backend.generate_chat(messages=messages, max_tokens=100)
        )
        print(f"Response: {chat_response}")
        
        # Unload model
        print("\nUnloading model...")
        await model_manager.unload_model(server.id)
        print("✓ Model unloaded")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


async def test_mlx_backend():
    """Test MLX backend (requires Apple Silicon and MLX model)."""
    print("\n=== Testing MLX Backend ===")
    
    # Create a test server config
    # Note: Replace with actual MLX model path
    server = LLMServer(
        name="test-mlx",
        model_name="test-model",
        mode=ServerMode.SELF_HOSTED,
        model_path="mlx-community/phi-2-mlx",  # Or local path
        backend_type="mlx",
        backend_config={}
    )
    
    try:
        # Load the model
        print(f"Loading model from {server.model_path}...")
        backend = await model_manager.load_model(server)
        print("✓ Model loaded successfully")
        
        # Test text generation
        print("\nTesting text generation...")
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: backend.generate(
                prompt="Write a haiku about coding:",
                max_tokens=50,
                temperature=0.7
            )
        )
        print(f"Response: {response}")
        
        # Unload model
        print("\nUnloading model...")
        await model_manager.unload_model(server.id)
        print("✓ Model unloaded")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


async def test_model_manager():
    """Test model manager functionality."""
    print("\n=== Testing Model Manager ===")
    
    # List loaded models
    loaded = model_manager.list_loaded()
    print(f"Currently loaded models: {len(loaded)}")
    for server_id, info in loaded.items():
        print(f"  - {server_id}: {info['backend_type']} ({info['capabilities']})")
    
    return True


async def main():
    """Main test runner."""
    print("Self-Hosted Model Test Suite")
    print("=" * 50)
    
    # Test model manager
    await test_model_manager()
    
    # Uncomment to test specific backends
    # Note: Update model paths before running!
    
    # await test_llamacpp_backend()
    # await test_mlx_backend()
    
    print("\n" + "=" * 50)
    print("Tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
