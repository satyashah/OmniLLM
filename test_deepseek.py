#!/usr/bin/env python3
"""
This script tests the DeepSeek chat completions API directly.
Make sure you have installed the 'requests' package:
    pip install requests

Set the DEEPSEEK_API_KEY environment variable with your API key.
Optionally, set DEEPSEEK_BASE_URL if you are not using the default.
"""

import os
import requests
import json

def main():
    # Retrieve the DeepSeek API key from environment variables.
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

    # Define the base URL and completions endpoint.
    # Default is "https://api.deepseek.com"; update if needed.
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    endpoint = f"{base_url}/v1/chat/completions"

    # Set the required headers.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Define the request payload.
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Explain the capital of France."}
        ],
        "max_tokens": 100  # Adjust as needed.
    }

    try:
        # Send the POST request.
        response = requests.post(endpoint, headers=headers, json=payload)
        print("Status Code:", response.status_code)
        print("Response Body:")
        print(response.text)
    except Exception as e:
        print(f"Error while calling DeepSeek API: {e}")

if __name__ == "__main__":
    main()
