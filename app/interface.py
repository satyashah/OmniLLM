# interface.py
## Interface for each provider

from typing import Any
from abc import ABC, abstractmethod
from .models import ChatCompletionRequest, ChatCompletionResponse

class ChatProvider(ABC):
    """Abstract base class for chat completion providers"""
    
    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion response for the given request
        
        Args:
            request: ChatCompletionRequest containing the input parameters
            
        Returns:
            ChatCompletionResponse containing the generated response
        """
        pass





