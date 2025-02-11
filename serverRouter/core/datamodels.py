from typing import List, Optional, Dict, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum

class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"

class ModelInfo(BaseModel):
    """Information about a model"""
    name: str = Field(..., description="Full name/version of the model")
    provider: ModelProvider = Field(..., description="Provider of the model")
    description: str = Field(..., description="Description of the model")
    max_tokens: Optional[int] = Field(None, description="Maximum context length")

## Chat Completion Models

class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation"""
    role: str = Field(..., description="Role of the message sender (e.g. 'user', 'assistant', 'system')")
    content: str = Field(..., description="Content of the message")

class ToolFunctionParameters(BaseModel):
    type: Literal["object"] = Field(default="object")
    properties: Dict[str, Dict[str, str]] 
    required: Optional[List[str]] = None

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: ToolFunctionParameters

class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: ToolFunction

class ToolChoiceOption(BaseModel):
    type: Literal["function"]
    function: Dict[str, str]

class ChatCompletionRequest(BaseModel):
    """Input parameters for a chat completion request"""
    model: str = Field(..., description="Name of the model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature (0-2)")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream the response")
    tools: Optional[List[ToolDefinition]] = Field(default=None, description="List of tools available to the model")
    tool_choice: Optional[Union[Literal["none", "auto"], ToolChoiceOption]] = Field(
        default=None, 
        description="Tool choice configuration ('none', 'auto', or specific tool)"
    )
    response_format: Optional[Dict[str, str]] = Field(
        default=None,
        description="Response format specification (e.g., {'type': 'json_object'})"
    )

class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: Dict[str, str]

class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request"""
    model: str = Field(..., description="Name of the model used")
    content: str = Field(..., description="Generated content")
    provider: str = Field(..., description="Provider that generated the response")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="Tool calls requested by the model"
    )
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics"
    )

## Image Generation Models
class ImageSize(str, Enum):
    """Supported image sizes"""
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"

class ImageGenerationRequest(BaseModel):
    """Input parameters for an image generation request"""
    prompt: str = Field(..., description="Text description of the desired image")
    model: str = Field(default="dall-e-3", description="Name of the model to use")
    size: ImageSize = Field(default=ImageSize.LARGE, description="Size of the generated image")
    quality: Literal["standard", "hd"] = Field(default="standard", description="Quality of the generated image")
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate")
    #They are optional because they are only needed for Gemini.
    google_cloud_project_id: Optional[str] = Field(None, description="Google Cloud Project ID (required for Gemini)") # Added field
    google_cloud_location: Optional[str] = Field(None, description="Google Cloud Location (required for Gemini)") # Added field

class ImageGenerationResponse(BaseModel):
    """Response from an image generation request"""
    urls: List[str] = Field(..., description="URLs of the generated images")
    model: str = Field(..., description="Name of the model used")
    provider: str = Field(..., description="Provider that generated the images")