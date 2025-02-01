## main.py
## Entry point for the application

from fastapi import FastAPI, HTTPException
from typing import Dict, Type
from app.core.interfaces import ChatProvider
from app.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ModelInfo,
    ModelProvider
)
from app.core.exceptions import ProviderNotFoundException, ProviderError
from app.providers.anthropic.provider import AnthropicProvider
from app.providers.openai.provider import OpenAIProvider
from app.core.models import (
    MODELS,
)

app = FastAPI(title="OmniLLM", description="One Key, One API, Hundreds of Models")

# Provider instances cache
PROVIDERS = {
    ModelProvider.OPENAI: OpenAIProvider(),
    ModelProvider.ANTHROPIC: AnthropicProvider()
}

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """
    Create a chat completion using the specified model
    
    Args:
        request: The chat completion request
        
    Returns:
        ChatCompletionResponse containing the generated response
    """
    try:
        # Look up the model info
        model_info = MODELS.get(request.model)
        if not model_info:

            raise HTTPException(
                status_code=400, 
                detail=f"Unknown model: {request.model}"
            )
        request.model = model_info.name
        
        # Get the provider for this model
        provider = PROVIDERS.get(model_info.provider)
        if not provider:

            raise HTTPException(
                status_code=500, 
                detail=f"Provider not configured: {model_info.provider}"
            )
        
        # Create the completion
        response = await provider.chat_complete(request)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List all available models"""
    return {
        "models": [
            {
                "id": model_id,
                **model_info.model_dump()
            }
            for model_id, model_info in MODELS.items()
        ]
    }


