#!/bin/bash
# Quick test script for the LLM Hosting project

echo "=== Testing CLI Commands ==="
echo ""

echo "1. Listing servers..."
.venv/bin/python main.py list-servers
echo ""

echo "2. Adding another server..."
.venv/bin/python main.py add-server \
  --name "Second Server" \
  --endpoint "http://localhost:5678/v1/chat/completions" \
  --model "second-model"
echo ""

echo "3. Listing all servers..."
.venv/bin/python main.py list-servers
echo ""

echo "=== Starting API Server ==="
echo "Run: .venv/bin/python main.py start --port 8000"
echo ""
echo "Then test with:"
echo "curl http://localhost:8000/docs"
echo ""
echo "Or test an endpoint:"
echo 'curl -X POST http://localhost:8000/v1/chat/completions \\'
echo '  -H "Content-Type: application/json" \\'
echo '  -d '"'"'{"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}'"'"''