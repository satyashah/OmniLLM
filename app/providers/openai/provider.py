from typing import Dict, Any
import openai
from app.core.interfaces import ChatProvider
from app.core.datamodels import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from app.core.exceptions import ProviderError
from dotenv import load_dotenv

load_dotenv(".env")

class OpenAIProvider(ChatProvider):
    """OpenAI chat completion provider"""
    
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
