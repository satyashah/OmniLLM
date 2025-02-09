# serverRouter/providers/gemini/provider.py
from typing import Dict, Any, List
from google import generativeai as genai  # Import as genai
from google.generativeai import types
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv, find_dotenv
import os
import logging

load_dotenv(find_dotenv())

class GeminiProvider(ChatProvider, ImageProvider):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")

        try:
            # Configure the API key globally.  This is the preferred method.
            genai.configure(api_key=api_key)
            self.model_name = "gemini-2.0-flash"  # Keep this

        except Exception as e:
            logging.exception("Error initializing Gemini client:")
            raise ProviderError(f"Failed to initialize Gemini client: {str(e)}")

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = []
            for msg in request.messages:
                messages.append({"role": msg.role, "parts": [msg.content]})

            # Create the GenerativeModel instance *after* configuration.
            model = genai.GenerativeModel(model_name=request.model)

            response = await model.generate_content_async(
                contents=messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens or 256,
                    temperature=request.temperature
                )
            )

            if response and response.text:
                return ChatCompletionResponse(
                    model=request.model,
                    content=response.text,
                    provider="gemini",
                    usage={}
                )
            else:
                logging.warning("Empty response from Gemini API.")
                raise ProviderError("Empty response from Gemini API.")

        except Exception as e:
            logging.exception("Gemini API error (chat):")
            raise ProviderError(f"Gemini API error (chat): {str(e)}")

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise ProviderError("Gemini API does not support image generation yet.")