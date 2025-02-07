from typing import Dict, Any
import openai
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse
)
from serverRouter.core.exceptions import ProviderError

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
class OpenAIProvider(ChatProvider, ImageProvider):
    """OpenAI provider supporting chat completions and image generation,
       including GPT-4, GPT-3.5, GPT-4o, and O1 models."""
    
    def __init__(self):
        """Initialize the OpenAI provider with API key from environment"""
        try:
            openai.api_key = openai.api_key or None  # Expect key in env variable OPENAI_API_KEY
            if not openai.api_key:
                raise ProviderError("OPENAI_API_KEY not set in environment.")
            self.client = openai
        except Exception as e:
            raise ProviderError(f"Failed to initialize OpenAI client: {str(e)}")
            
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion using OpenAI's API.
        Supports GPT-4, GPT-3.5, GPT-4o, O1, etc.
        """
        try:
            response = await self.client.ChatCompletion.acreate(
                model=request.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens if request.max_tokens else None,
                stream=False
            )
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.get("prompt_tokens", 0),
                    "completion_tokens": response.usage.get("completion_tokens", 0)
                }
            )
        except openai.error.OpenAIError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate images using OpenAI's image API (DALL-E)"""
        try:
            response = await self.client.Image.create(
                model=request.model,
                prompt=request.prompt,
                n=request.n,
                size=request.size.value
            )
            return ImageGenerationResponse(
                urls=[image["url"] for image in response["data"]],
                model=request.model,
                provider="openai"
            )
        except openai.error.OpenAIError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")

