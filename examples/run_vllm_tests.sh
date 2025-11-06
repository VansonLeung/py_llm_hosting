#!/bin/bash
# Quick start script for vLLM testing
# This script helps you quickly set up and test vLLM backend

set -e

echo "============================================"
echo "vLLM Backend Testing - Quick Start"
echo "============================================"
echo ""

# Check if vLLM is installed
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "⚠️  vLLM not installed!"
    echo ""
    echo "Install with:"
    echo "  pip install vllm transformers"
    echo ""
    exit 1
fi

echo "✓ vLLM is installed"

# Check if server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Server not running!"
    echo ""
    echo "Start a vLLM server with:"
    echo "  python main.py server add vllm-gpt2 \\"
    echo "    --endpoint-url 'self-hosted' \\"
    echo "    --model-path 'gpt2' \\"
    echo "    --backend vllm"
    echo ""
    echo "  python main.py server start vllm-gpt2"
    echo ""
    exit 1
fi

echo "✓ Server is running"
echo ""

# Parse command line args
TEST_TYPE=${1:-all}
MODEL=${2:-gpt2}

case $TEST_TYPE in
    basic|all)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Running Basic vLLM Tests"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        ./examples/test_vllm.py --model "$MODEL"
        ;;
esac

case $TEST_TYPE in
    streaming|all)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Running Streaming Tests"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        ./examples/test_vllm_streaming.py --model "$MODEL"
        ;;
esac

case $TEST_TYPE in
    basic|streaming|all)
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo ""
        echo "Usage: $0 [test-type] [model]"
        echo ""
        echo "Test types:"
        echo "  basic     - Run basic tests only"
        echo "  streaming - Run streaming tests only"
        echo "  all       - Run all tests (default)"
        echo ""
        echo "Example:"
        echo "  $0 all gpt2"
        echo "  $0 streaming my-model"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "✅ All tests completed!"
echo "============================================"
