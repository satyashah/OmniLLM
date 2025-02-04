"""
This is the client that will be used to test the image generation endpoints.

It will be grabbing the models from the models list and then testing them with image generation prompts.
"""
import os
from typing import Dict, List, Any
import requests
import base64
from pathlib import Path

BASE_URL = "http://localhost:8000"
OUTPUT_DIR = "test_outputs"

class ImageModelTester:
    def __init__(self):
        os.system('cls')
        self.models = self.get_available_models()
        # Create output directory if it doesn't exist
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available image models from the API"""
        response = requests.get(f"{BASE_URL}/v1/models/image")
        models_data = response.json()
        # Filter for image models only
        return models_data["models"]

    def test_model(self, prompt: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test the image generation endpoint for a specific model"""
        request_data = {
            "model": model_info["id"],
            "prompt": prompt,
            "n": 1,  # Number of images to generate
            "size": "1024x1024"  # Default size
        }
        
        response = requests.post(
            f"{BASE_URL}/v1/images/generate",
            json=request_data
        )

        return response.json()

    def validate_models(self, test_prompt: str):
        print("\n=== Image Model Validation Results ===")
        print(f"Found {len(self.models)} image models to test\n")
        
        all_passed = True
        for model in self.models:
            model_id = model["id"]
            try:
                response = self.test_model(test_prompt, model)
                passed = len(response["urls"]) > 0
                
                status = "✅ PASS" if passed else "❌ FAIL"
                provider = f"[{response['provider']}]"
                
                print(f"{status} {model_id:<20} {provider:<12}")
                for url in response["urls"]:
                    print(f"  Image URL: {url}")
                print()
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ FAIL {model_id:<20} [ERROR]")
                print(f"  Error: {str(e)}")
                all_passed = False
        
        print("Final Result:", "✅ ALL MODELS RESPONDING" if all_passed else "❌ SOME MODELS FAILED")

if __name__ == "__main__":
    test_prompt = "A serene landscape with mountains and a lake at sunset"
    tester = ImageModelTester()
    tester.validate_models(test_prompt) 