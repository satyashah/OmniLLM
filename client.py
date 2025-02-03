"""
This is the client that will be used to test the models.

It will be grabbing the models from the models list and then testing them with a test query.
"""
import os
from typing import Dict, List, Any
import requests

BASE_URL = "http://localhost:8000"

class ModelTester:
    def __init__(self):
        os.system('cls')
        self.models = self.get_available_models()
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from the API"""
        response = requests.get(f"{BASE_URL}/v1/models")
        models_data = response.json()
        # print("\n=== Available Models ===")
        # print(json.dumps(models_data, indent=2))
        return models_data["models"]


    def test_model(self, test_query, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test the chat completion endpoint for a specific model"""
        request_data = {
            "model": model_info["id"],
            "messages": [
                {
                    "role": "user",
                    "content": test_query
                }
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        # print(f"\n=== Testing Chat Completion with {model_info['id']} ===")
        # print("Request:")
        # print(json.dumps(request_data, indent=2))
        
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data
        )
        
        response_data = response.json()
        # print("\nResponse:")
        # print(json.dumps(response_data, indent=2))
        
        return {
            "content": response_data["content"],
            "provider": model_info.get("provider", "unknown")
        }

    def validate_models(self, test_query):
        print("\n=== Model Validation Results ===")
        print(f"Found {len(self.models)} models to test\n")
        
        all_passed = True
        for model in self.models:
            model_id = model["id"]
            try:
                response = self.test_model(test_query, model)
                content = response["content"]
                passed = bool(content.strip())
                
                status = "✅ PASS" if passed else "❌ FAIL"
                provider = f"[{response['provider']}]"
                
                print(f"{status} {model_id:<20} {provider:<12}")
                print(f"  Response: {content}\n")
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ FAIL {model_id:<20} [ERROR]")
                print(f"  Error: {str(e)}")
                all_passed = False
        
        print("Final Result:", "✅ ALL MODELS RESPONDING" if all_passed else "❌ SOME MODELS FAILED")

if __name__ == "__main__":
    test_query = "What is the capital of France?"
    tester = ModelTester()
    tester.validate_models(test_query)