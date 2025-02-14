# serverRouter/providers/gemini/provider.py
import asyncio
from typing import Dict, Any, List, Union
from google import generativeai as genai
from google.generativeai import types
import asyncio
from typing import Dict, Any, List, Union
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse
)
from serverRouter.core.exceptions import ProviderError
from serverRouter.helpers.vertex_ai_helper import VertexAiHelper #Import our Helper
from dotenv import load_dotenv, find_dotenv
import os
import logging
import base64
import io
from PIL import Image
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

load_dotenv(find_dotenv())

class GeminiProvider(ChatProvider, ImageProvider):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")

        try:
            # genai.configure(api_key=api_key) #no longer using this
            pass
            # Removed self.model_name, as we'll handle models dynamically
        except Exception as e:
            logging.exception("Error initializing Gemini client:")
            raise ProviderError(f"Failed to initialize Gemini client: {str(e)}")

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = []
            for msg in request.messages:
                messages.append({"role": msg.role, "parts": [msg.content]})

            model = genai.GenerativeModel(model_name=request.model)

            response = await model.generate_content_async(
                contents=messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens or 2048,  # Increased default
                    temperature=request.temperature
                )
            )

            if response and response.text:
                return ChatCompletionResponse(
                    model=request.model,
                    content=response.text,
                    provider="gemini",
                    usage={}  #  Gemini doesn't directly provide usage in the same way
                )
        except Exception as e:
            logging.exception("Gemini API error (chat):")
            raise ProviderError(f"Gemini API error (chat): {str(e)}")
        
    #Google's Infrastructure: The Gemini image generation models (like Imagen 3) are hosted on Google's infrastructure (Vertex AI). To access them, you must have a Google Cloud project. It's not possible to abstract away this requirement completely.
    #Authentication: Google requires authentication to access its cloud services. Even if you could somehow bypass the project ID requirement, you would still need to provide valid credentials.

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
            try:
                # Call the Vertex AI Helper asynchronously.
                image_urls = await VertexAiHelper.call(
                    prompt=request.prompt,
                    model_name=request.model,  # This should now be "imagegeneration@006" from your registry.
                    google_cloud_project_id=request.google_cloud_project_id,
                    google_cloud_location=request.google_cloud_location,
                    num_images=request.n,
                )
                return ImageGenerationResponse(
                    urls=image_urls,
                    model=request.model,
                    provider="gemini"
                )
            except Exception as e:
                logging.exception("Gemini API error (image):")
                raise ProviderError(f"Gemini API error: {str(e)}")
