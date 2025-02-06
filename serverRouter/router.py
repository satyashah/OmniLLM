## main.py
## Entry point for the application

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ModelProvider,
    ImageGenerationRequest,
    ImageGenerationResponse
)
from serverRouter.providers.anthropic.provider import AnthropicProvider
from serverRouter.providers.openai.provider import OpenAIProvider

from serverRouter.core.models import (
    MODELS,
    CHAT_MODELS,
    IMAGE_MODELS
)


app = FastAPI(title="OmniLLM", description="One Key, One API, Hundreds of Models")
security = HTTPBearer()

# Provider instances cache
PROVIDERS = {
    ModelProvider.OPENAI: OpenAIProvider(),
    ModelProvider.ANTHROPIC: AnthropicProvider()
}

VALID_API_KEYS = {
    "test-sk1o83e",
}

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials.credentials

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
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

@app.get("/v1/models/chat")
async def list_chat_models(api_key: str = Depends(verify_api_key)):
    """List all available chat models"""

    return {
        "models": [
            {
                "id": model_id,
                **model_info.model_dump()
            }
            for model_id, model_info in CHAT_MODELS.items()
        ]
    }

@app.get("/v1/models/image")
async def list_image_models(api_key: str = Depends(verify_api_key)):
    """List all available image models"""
    return {
        "models": [
            {
                "id": model_id,
                **model_info.model_dump()
            }
            for model_id, model_info in IMAGE_MODELS.items()
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatCompletionResponse:

    """
    Create a chat completion using the specified model
    """
    try:
        # Look up the model info
        model_info = CHAT_MODELS.get(request.model)
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

@app.post("/v1/images/generate")
async def create_image(
    request: ImageGenerationRequest,
    api_key: str = Depends(verify_api_key)
) -> ImageGenerationResponse:
    """
    Generate images using the specified model
    """
    try:
        # For now, we only support DALL-E models from OpenAI
        model_info = IMAGE_MODELS.get(request.model)
        if not model_info:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown model: {request.model}"
            )
        request.model = model_info.name
        
        # Get the OpenAI provider
        provider = PROVIDERS.get(model_info.provider)
        if not provider:
            raise HTTPException(
                status_code=500, 
                detail=f"Provider not configured: {model_info.provider}"
            )
        
        # Generate the images
        response = await provider.generate_image(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


