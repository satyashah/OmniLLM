from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum

class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class ModelInfo(BaseModel):
    """Information about a supported model"""
    name: str = Field(..., description="Name of the model")
    provider: ModelProvider = Field(..., description="Provider of the model")
    description: str = Field(..., description="Description of the model")
    max_tokens: int = Field(..., description="Maximum tokens supported")
    
class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation"""
    role: str = Field(..., description="Role of the message sender (e.g. 'user', 'assistant', 'system')")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    """Input parameters for a chat completion request"""
    model: str = Field(..., description="Name of the model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=1.0, description="Sampling temperature (0-2)")
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream the response")

class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request"""
    model: str = Field(..., description="Name of the model used")
    content: str = Field(..., description="Generated content")
    provider: str = Field(..., description="Provider that generated the response")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics"
    )