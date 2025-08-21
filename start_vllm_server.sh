#!/bin/bash

# Start vLLM server with a small model (TinyLlama is ~1.1B params, very small)
# This uses the responses API endpoint which supports logprobs

echo "Starting vLLM server with TinyLlama model..."
echo "This will expose the server on port 8000"
echo ""

# Using TinyLlama-1.1B which is a very small model perfect for testing
python -m vllm.entrypoints.openai.api_server \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --api-key token-abc123 \
    --port 8000 \
    --max-model-len 2048 \
    --dtype auto \
    --enforce-eager

# Alternative even smaller models you can try:
# --model "facebook/opt-125m"  # 125M params, even smaller
# --model "gpt2"  # 124M params, classic small model
