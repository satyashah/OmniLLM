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
from dotenv import load_dotenv

load_dotenv(".env")

class OpenAIProvider(ChatProvider, ImageProvider):
    """OpenAI provider supporting both chat and image generation"""
    
    def __init__(self):
        """Initialize the OpenAI provider with API key from environment"""
        try:
            # openai.api_key should be set via environment variable OPENAI_API_KEY
            self.client = openai.AsyncOpenAI()
        except Exception as e:
            raise ProviderError(f"Failed to initialize OpenAI client: {str(e)}")
            
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion using OpenAI's API
        
        Args:
            request: ChatCompletionRequest containing the input parameters
            
        Returns:
            ChatCompletionResponse containing the generated response
        """
        try:
            # Convert our generic request to OpenAI-specific format
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens if request.max_tokens else None
            )
            
            # Convert OpenAI response to our generic format
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except openai.APIError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate images using DALL-E"""
        try:
            response = await self.client.images.generate(
                model=request.model,
                prompt=request.prompt,
                size=request.size.value,
                quality=request.quality,
                n=request.n
            )
            
            return ImageGenerationResponse(
                urls=[image.url for image in response.data],
                model=request.model,
                provider="openai"
            )
            
        except openai.APIError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")
