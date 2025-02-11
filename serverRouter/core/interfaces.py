from abc import ABC, abstractmethod
from .datamodels import ChatCompletionRequest, ChatCompletionResponse, ImageGenerationRequest, ImageGenerationResponse

class ChatProvider(ABC):
    """Abstract base class for chat completion providers"""
    
    @abstractmethod
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion response for the given request
        """
        pass

class ImageProvider(ABC):
    """Abstract base class for image generation providers"""
    
    @abstractmethod
    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate an image based on the given request
        """
        pass





