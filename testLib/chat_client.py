"""
This is the client that will be used to test the chat models.

It will be grabbing the models from the models list and then testing them with a test query.
"""
import os
from typing import Dict, List, Any
from clientLib.APIClient import APIClient

class ChatModelTester:
    def __init__(self):
        os.system('cls')
        self.client = APIClient()
        self.models = self.client.get_available_models("chat")

    def test_model(self, test_query: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test the chat completion endpoint for a specific model"""
        messages = [{"role": "user", "content": test_query}]
        
        response = self.client.chat(
            messages=messages,
            model=model_info["id"],
            temperature=0.7,
            max_tokens=100
        )
        
        return {
            "content": response["content"],
            "provider": model_info.get("provider", "unknown")
        }

    def validate_models(self, test_query):
        print("\n=== Chat Model Validation Results ===")
        print(f"Found {len(self.models)} chat models to test\n")
        
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
    tester = ChatModelTester()
    tester.validate_models(test_query)