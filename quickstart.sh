#!/bin/bash
# Quick start script for LLM Endpoint Hosting

echo "🚀 LLM Endpoint Hosting - Quick Start"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if needed
if ! .venv/bin/python -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    .venv/bin/pip install -r requirements.txt
    echo "✅ Dependencies installed"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Available commands:"
echo "  1. Add server:    python main.py add-server --name 'My Server' --endpoint 'http://...' --model 'model-name'"
echo "  2. List servers:  python main.py list-servers"
echo "  3. Start API:     python main.py start --port 8000"
echo ""
echo "📖 API Documentation: http://localhost:8000/docs (after starting server)"
echo ""

# Check if any servers exist
if [ -f "servers.json" ]; then
    echo "📊 Current servers:"
    .venv/bin/python main.py list-servers
fi