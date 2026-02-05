#!/usr/bin/env python3
"""
Test Client for Llama 3.3 70B vLLM API Server
==============================================

This script tests the OpenAI-compatible API server started by
llama33_online_server.sh.

Prerequisites:
1. Start the server: ./llama33_online_server.sh
2. Wait for server to be ready (shows "Application startup complete")
3. Run this client: python test_api_client.py

The client demonstrates both completion and chat endpoints.
"""

import requests
import json

# Server configuration
BASE_URL = "http://localhost:8000"
COMPLETIONS_URL = f"{BASE_URL}/v1/completions"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"

def test_completion():
    """Test the /v1/completions endpoint"""
    print("Testing /v1/completions endpoint...")
    print("-" * 60)
    
    payload = {
        "model": "models/Llama-3.3-70B-Instruct/",
        "prompt": "What is the capital of France?",
        "max_tokens": 100,
        "temperature": 0.0,
    }
    
    response = requests.post(COMPLETIONS_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prompt: {payload['prompt']}")
        print(f"Response: {result['choices'][0]['text']}")
        print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        print()

def test_chat():
    """Test the /v1/chat/completions endpoint"""
    print("Testing /v1/chat/completions endpoint...")
    print("-" * 60)
    
    payload = {
        "model": "models/Llama-3.3-70B-Instruct/",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    
    response = requests.post(CHAT_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"User: {payload['messages'][0]['content']}")
        print(f"Assistant: {result['choices'][0]['message']['content']}")
        print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        print()

def main():
    print("Llama 3.3 70B API Client Test")
    print("=" * 60)
    print()
    
    try:
        # Test completion endpoint
        test_completion()
        
        # Test chat endpoint
        test_chat()
        
        print("=" * 60)
        print("All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server at", BASE_URL)
        print("Make sure the server is running: ./llama33_online_server.sh")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
