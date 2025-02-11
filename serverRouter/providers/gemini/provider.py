# serverRouter/providers/gemini/provider.py
import asyncio
from typing import Dict, Any, List, Union
from google import generativeai as genai
import asyncio
from typing import Dict, Any, List, Union
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv, find_dotenv
import os
import logging
load_dotenv(find_dotenv())

class GeminiProvider(ChatProvider):
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
        
