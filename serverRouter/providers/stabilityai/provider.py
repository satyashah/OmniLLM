import os
import requests
import io
import logging
from typing import Dict, Any, Optional
from serverRouter.core.interfaces import ImageProvider
from serverRouter.core.datamodels import (
    ImageGenerationRequest,
    ImageGenerationResponse
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class StabilityAIProvider(ImageProvider):
    """Stability AI provider for image generation."""

    def __init__(self):
        api_key = os.getenv("STABILITY_API_KEY")
        if not api_key:
            raise ProviderError("STABILITY_API_KEY not found in environment")
        self.api_key = api_key
        self.base_url = "https://api.stability.ai/v2beta/stable-image"

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generates an image using Stability AI's API."""
        model_name = request.model.lower()
        endpoint = None

        if "ultra" in model_name:
            endpoint = "/generate/ultra"
        elif "core" in model_name:
            endpoint = "/generate/core"
        elif "sd3" in model_name or "sd3.5" in model_name:
            endpoint = "/generate/sd3"
        else:
            raise ProviderError(f"Unsupported Stability AI model: {request.model}")

        if not endpoint:
            raise ProviderError(f"Invalid Stability AI model specified {request.model}")

        url = f"{self.base_url}{endpoint}"

        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {self.api_key}"
        }

        data: Dict[str, Any] = {
            "prompt": request.prompt,
            "aspect_ratio": request.aspect_ratio,
            "seed": request.seed,
            "output_format": request.output_format,
        }

        # Handle the negative prompt only if it's NOT sd3.5-large-turbo and the endpoint is NOT generate/sd3
        if model_name != "sd3.5-large-turbo" and endpoint != "/generate/sd3":
            data["negative_prompt"] = request.negative_prompt

        # Handle style_preset only if the endpoint is NOT generate/sd3
        if endpoint != "/generate/sd3":
            data["style_preset"] = request.style_preset

        # Remove None values from the data dict
        data = {k: v for k, v in data.items() if v is not None}

        # ENCODE THE DATA AS MULTIPART/FORM-DATA
        files: Dict[str, Any] = {}
        for key, value in data.items():
            files[key] = (None, str(value)) #All values must be strings


        try:
            logging.debug(f"Request URL: {url}")
            logging.debug(f"Request Headers: {headers}")
            logging.debug(f"Request Data: {data}")
            logging.debug(f"Request Files: {files}")

            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()

            #Check for NSFW content filtering
            finish_reason = response.headers.get("finish-reason")
            if finish_reason == 'CONTENT_FILTERED':
                raise Warning("Generation failed NSFW classifier")

            # Check if the response is in JSON format (error case)
            if response.headers['content-type'] == 'application/json':
                error_data = response.json()
                raise ProviderError(f"Stability AI API Error: {error_data}")

            # Get the seed from the response headers
            seed = response.headers.get("seed")
            output_format = data.get("output_format", "png") #png is default

            image_bytes = io.BytesIO(response.content)
            image_url = f"data:image/{output_format};base64,{response.content.decode('latin-1')}"

            # Save the image to a file
            generated = f"generated_{seed}.{output_format}"
            with open(generated, "wb") as f:
                f.write(response.content)
            print(f"Saved image {generated}")


            return ImageGenerationResponse(
                urls=[image_url],  # Use the data URL
                model=request.model,
                provider="stabilityai"
            )
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            if hasattr(e.response, 'text'):
                logging.error(f"Response content: {e.response.text}")
            raise ProviderError(f"Stability AI API Request Error: {str(e)}")
        except Exception as e:
            logging.exception(f"An unexpected error occurred: {e}")
            raise ProviderError(f"Stability AI API Error: {str(e)}")