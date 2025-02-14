"""
This is the client that will be used to test the image generation endpoints.

It will be grabbing the models from the models list and then testing them with image generation prompts.
"""
import os
from typing import Dict, List, Any
from clientLib.APIClient import APIClient

class ImageModelTester:
    def __init__(self):
        os.system('cls')
        self.client = APIClient()
        self.models = self.client.get_available_models("image")

    def test_model(self, prompt: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test the image generation endpoint for a specific model"""
        model_id = model_info["id"]
        provider = model_info["provider"]

        if provider == "gemini":  # Check provider type
            return self.client.generate_image(
                prompt=prompt,
                model=model_id,
                n=1,
                size="1024x1024",
                google_cloud_project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),  # Add project ID
                google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION")  # Add location
            )
        elif provider == "stabilityai":
            return self.client.generate_image(
                prompt=prompt,
                model=model_id,
                n=1,
                size="1024x1024",
                aspect_ratio="1:1",  # You can change this
                output_format="jpeg"  # or "png"
            )
        else:
            return self.client.generate_image(
                prompt=prompt,
                model=model_id,
                n=1,
                size="1024x1024"
            )

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
    test_prompt = "Pink panda on a skateboard at maryland university chasing a smoking snoop dogg poodle"
    tester = ImageModelTester()
    tester.validate_models(test_prompt)