#!/bin/bash

# Test script for vLLM responses API with logprobs using curl

echo "================================================"
echo "Testing vLLM Responses API with Logprobs (curl)"
echo "================================================"
echo ""

# Non-streaming request with logprobs
echo "1. NON-STREAMING REQUEST WITH LOGPROBS:"
echo "----------------------------------------"
echo ""

curl -X POST http://localhost:8000/v1/responses \
  -H "Authorization: Bearer token-abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is the capital of France?"
        }
      ]
    },
    "max_tokens": 30,
    "temperature": 0.7,
    "top_logprobs": 5,
    "include": ["message.output_text.logprobs"],
    "stream": false
  }' | jq '.'

echo ""
echo ""
echo "2. STREAMING REQUEST WITH LOGPROBS:"
echo "------------------------------------"
echo ""

curl -X POST http://localhost:8000/v1/responses \
  -H "Authorization: Bearer token-abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Say hello in three languages"
        }
      ]
    },
    "max_tokens": 50,
    "temperature": 0.7,
    "top_logprobs": 3,
    "include": ["message.output_text.logprobs"],
    "stream": true
  }'

echo ""
echo ""
echo "================================================"
echo "Key points about logprobs extraction:"
echo "================================================"
echo ""
echo "1. Required parameters:"
echo "   - top_logprobs: Must be > 0 to get logprobs"
echo "   - include: Must contain 'message.output_text.logprobs'"
echo ""
echo "2. Response structure (non-streaming):"
echo "   - logprobs appear in: outputs[].logprobs[]"
echo "   - Each token has: token, logprob, top_logprobs[]"
echo ""
echo "3. Response structure (streaming):"
echo "   - logprobs appear in: response.output_text.delta events"
echo "   - Final logprobs in: response.output_text.done event"
echo ""
