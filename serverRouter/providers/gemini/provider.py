from typing import Dict, Any, List
import os
from google import genai
from google.generativeai import types
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
load_dotenv()

# Remove or comment out any module-level client initialization:
# GEMINI_API_KEY = None
# try:
#     GEMINI_API_KEY = genai.Client(api_key="YOUR_GEMINI_API_KEY")  # Remove hard-coded key
# except Exception as e:
#     raise ProviderError(f"Failed to initialize Gemini client: {str(e)}")

class GeminiProvider(ChatProvider, ImageProvider):
    """Gemini provider supporting both chat and image generation (vision) capabilities."""
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")
        # Instead of using a module-level configure, instantiate a client:
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion using Gemini's API.
        For text-only input, we use the last message as the prompt.
        """
        try:
            prompt = request.messages[-1].content
            response = self.client.models.generate_content(
                model=request.model,  # e.g., "gemini-2.0-flash"
                contents=[prompt],    # pass as a list
                config=types.GenerateContentConfig(
                    max_output_tokens=request.max_tokens or 256,
                    temperature=request.temperature
                )
            )
            return ChatCompletionResponse(
                model=response.model,
                content=response.text,
                provider="gemini",
                usage={}
            )
        except Exception as e:
            raise ProviderError(f"Gemini API error (chat): {str(e)}")

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate an image using Gemini's API in multimodal mode.
        This example uses a text-only prompt; for vision use, extend it to pass image data.
        """
        try:
            response = self.client.models.generate_content(
                model=request.model,  # e.g., "gemini-2.0-flash-img"
                contents=[request.prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=256,
                    temperature=0.7
                )
            )
            return ImageGenerationResponse(
                urls=[response.text],
                model=response.model,
                provider="gemini"
            )
        except Exception as e:
            raise ProviderError(f"Gemini API error (image): {str(e)}")
