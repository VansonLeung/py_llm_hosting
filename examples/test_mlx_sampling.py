"""Test MLX backend with different sampling parameters."""

import requests
import json
import time
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test MLX sampling parameters')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)), 
                   help='Port number for the server (default: from PORT env var or 8000)')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}/v1"

def test_sampling(temperature, top_p, prompt="Once upon a time"):
    """Test generation with specific sampling parameters."""
    print(f"\n{'='*60}")
    print(f"Testing: temperature={temperature}, top_p={top_p}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen2.5-0.5b",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        usage = data['usage']
        
        print(f"Response: {content}")
        print(f"Tokens: {usage['completion_tokens']}")
        return content
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def main():
    """Run sampling tests."""
    print("Testing MLX Backend Sampling Parameters")
    print("="*60)
    
    # Test different temperatures
    print("\n## Testing Different Temperatures")
    print("(Higher temperature = more random/creative)")
    
    # Low temperature - more deterministic
    test_sampling(temperature=0.1, top_p=1.0, prompt="The capital of France is")
    
    # Medium temperature
    test_sampling(temperature=0.7, top_p=1.0, prompt="The capital of France is")
    
    # High temperature - more random
    test_sampling(temperature=1.5, top_p=1.0, prompt="The capital of France is")
    
    # Test different top_p values
    print("\n## Testing Different Top-P Values")
    print("(Lower top_p = more focused on likely tokens)")
    
    test_sampling(temperature=0.8, top_p=0.5, prompt="Write a creative story starter:")
    test_sampling(temperature=0.8, top_p=0.9, prompt="Write a creative story starter:")
    test_sampling(temperature=0.8, top_p=1.0, prompt="Write a creative story starter:")
    
    print("\n" + "="*60)
    print("✓ All sampling tests completed!")

if __name__ == "__main__":
    main()
