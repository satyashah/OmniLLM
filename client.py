import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_list_models() -> Dict[str, Any]:
    """Test the models listing endpoint"""
    response = requests.get(f"{BASE_URL}/v1/models")
    print("\n=== Available Models ===")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_chat_completion(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Test the chat completion endpoint"""
    
    request_data = {
        "model": model_info["id"],
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print(f"\n=== Testing Chat Completion with {model_info['name']} ===")
    print("Request:")
    print(json.dumps(request_data, indent=2))
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=request_data
    )
    
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def main():
    """Run all tests"""
    try:
        # Test 1: List models
        models_response = test_list_models()
        
        # Test 2: Test chat completion with each model
        for model in models_response["models"]:
            try:
                test_chat_completion(model)
            except requests.exceptions.RequestException as e:
                print(f"\nError testing {model['id']}: {str(e)}")
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()





