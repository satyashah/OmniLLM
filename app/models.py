from typing import List, Optional, Dict
from pydantic import BaseModel, Field

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
    message: ChatMessage = Field(..., description="Generated message")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)"
    ) 