#!/usr/bin/env python3
"""
Test script for extracting logprobs from vLLM responses API.
This tests the new logprobs functionality added to the responses API.
"""

import json
import requests
import time
from typing import Any, Dict, List

# Server configuration
BASE_URL = "http://localhost:8000"
API_KEY = "token-abc123"

def test_responses_api_with_logprobs():
    """Test the /v1/responses endpoint with logprobs enabled."""
    
    # Endpoint for the responses API
    url = f"{BASE_URL}/v1/responses"
    
    # Headers with API key
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Request payload with logprobs configuration
    # The key fields for logprobs are:
    # - top_logprobs: number of top logprobs to return (must be > 0)
    # - include: must contain "message.output_text.logprobs" to get logprobs
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "top_logprobs": 5,  # Request top 5 logprobs for each token
        "include": ["message.output_text.logprobs"],  # CRITICAL: This enables logprobs in response
        "stream": False  # Non-streaming first
    }
    
    print("=" * 60)
    print("Testing Responses API with Logprobs (Non-Streaming)")
    print("=" * 60)
    print(f"\nRequest URL: {url}")
    print(f"Request payload:\n{json.dumps(payload, indent=2)}")
    
    # Make the request
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse status: {response.status_code}")
        print(f"\nFull response:\n{json.dumps(result, indent=2)}")
        
        # Extract and display logprobs
        if "outputs" in result:
            for idx, output in enumerate(result["outputs"]):
                print(f"\n--- Output {idx} ---")
                for item in output:
                    if item["type"] == "output_text":
                        print(f"Text: {item.get('text', '')}")
                        
                        # Extract logprobs if present
                        if "logprobs" in item and item["logprobs"]:
                            print("\nLogprobs for each token:")
                            for token_idx, token_logprobs in enumerate(item["logprobs"]):
                                print(f"\n  Token {token_idx}:")
                                print(f"    Token: '{token_logprobs.get('token', '')}'")
                                print(f"    Logprob: {token_logprobs.get('logprob', 'N/A')}")
                                
                                # Show top alternatives
                                if "top_logprobs" in token_logprobs:
                                    print("    Top alternatives:")
                                    for alt in token_logprobs["top_logprobs"]:
                                        print(f"      '{alt['token']}': {alt['logprob']:.4f}")
                        else:
                            print("No logprobs in response (check if 'include' parameter is set correctly)")
    else:
        print(f"\nError: {response.status_code}")
        print(response.text)


def test_responses_api_streaming_with_logprobs():
    """Test the /v1/responses endpoint with streaming and logprobs enabled."""
    
    url = f"{BASE_URL}/v1/responses"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {
                "role": "user",
                "content": "Count from 1 to 5."
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "top_logprobs": 3,  # Request top 3 logprobs for each token
        "include": ["message.output_text.logprobs"],  # Enable logprobs
        "stream": True  # Enable streaming
    }
    
    print("\n" + "=" * 60)
    print("Testing Responses API with Logprobs (Streaming)")
    print("=" * 60)
    print(f"\nRequest URL: {url}")
    print(f"Request payload:\n{json.dumps(payload, indent=2)}")
    
    # Make streaming request
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    if response.status_code == 200:
        print(f"\nResponse status: {response.status_code}")
        print("\nStreaming events:")
        
        accumulated_text = ""
        all_logprobs = []
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    
                    if data_str == "[DONE]":
                        print("\n[Stream finished]")
                        break
                    
                    try:
                        event = json.loads(data_str)
                        event_type = event.get("type", "")
                        
                        # Handle text delta events (these contain logprobs)
                        if event_type == "response.output_text.delta":
                            delta_text = event.get("delta", "")
                            accumulated_text += delta_text
                            
                            print(f"\n  Event: {event_type}")
                            print(f"    Delta text: '{delta_text}'")
                            
                            # Extract logprobs from delta event
                            if "logprobs" in event and event["logprobs"]:
                                for token_logprobs in event["logprobs"]:
                                    all_logprobs.append(token_logprobs)
                                    print(f"    Token: '{token_logprobs.get('token', '')}'")
                                    print(f"    Logprob: {token_logprobs.get('logprob', 'N/A')}")
                                    
                                    if "top_logprobs" in token_logprobs:
                                        print("    Alternatives:")
                                        for alt in token_logprobs["top_logprobs"][:3]:
                                            print(f"      '{alt['token']}': {alt['logprob']:.4f}")
                        
                        # Handle other event types
                        elif event_type in ["response.output_text.done", "response.output_item.done"]:
                            print(f"\n  Event: {event_type}")
                            if "text" in event:
                                print(f"    Final text: '{event['text']}'")
                            if "logprobs" in event and event["logprobs"]:
                                print(f"    Total tokens with logprobs: {len(event['logprobs'])}")
                        
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse: {data_str}")
        
        print(f"\n\nAccumulated text: '{accumulated_text}'")
        print(f"Total tokens with logprobs: {len(all_logprobs)}")
        
    else:
        print(f"\nError: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("Testing vLLM Responses API with Logprobs\n")
    print("Make sure the vLLM server is running on http://localhost:8000")
    print("You can start it with: bash start_vllm_server.sh")
    print("-" * 60)
    
    # Wait a moment for user to read
    time.sleep(2)
    
    try:
        # Test non-streaming
        test_responses_api_with_logprobs()
        
        # Test streaming
        test_responses_api_streaming_with_logprobs()
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to vLLM server.")
        print("Please make sure the server is running with: bash start_vllm_server.sh")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
