from serverRouter.core.datamodels import ModelInfo, ModelProvider

# Primary chat models registry
CHAT_MODELS = {
    "gpt-4": ModelInfo(
        name="gpt-4",
        provider=ModelProvider.OPENAI,
        description="OpenAI's most capable model for both language understanding and generation",
        max_tokens=8192
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        description="OpenAI's fast and efficient model with good capabilities",
        max_tokens=4096
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        description="OpenAI's GPT-4o variant (mini) for lightweight generation (requires reasoning access, Tier 5)",
        max_tokens=4096
    ),
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider=ModelProvider.OPENAI,
        description="OpenAI's GPT-4o model for generation without heavy reasoning (suitable for non–Tier 5 accounts)",
        max_tokens=4096
    ),
    "claude-3-opus": ModelInfo(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        description="Anthropic's most capable model",
        max_tokens=4096,
    ),
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        description="Anthropic's balanced model for performance and efficiency",
        max_tokens=4096,
    ),
    "gemini-2.0-flash": ModelInfo(
        name="gemini-2.0-flash",
        provider=ModelProvider.GEMINI,
        description="Google Gemini 2.0 Flash for chat completions",
        max_tokens=4096,
    )
}

# Primary image models registry
IMAGE_MODELS = {
    "dall-e-3": ModelInfo(
        name="dall-e-3",
        provider=ModelProvider.OPENAI,
        description="OpenAI's most advanced image generation model"
    ),
    "dall-e-2": ModelInfo(
        name="dall-e-2",
        provider=ModelProvider.OPENAI,
        description="OpenAI's efficient image generation model",
    ),
    "gemini-2.0-flash-img": ModelInfo(
        name="gemini-2.0-flash-img",
        provider=ModelProvider.GEMINI,
        description="Google Gemini 2.0 Flash for image generation"
    )
}

# Combined models dictionary
MODELS = {**CHAT_MODELS, **IMAGE_MODELS}

# Index models by provider for easy lookup
MODELS_BY_PROVIDER = {}
for model_id, model in MODELS.items():
    MODELS_BY_PROVIDER.setdefault(model.provider, []).append(model_id)

def get_model_by_id(model_id: str) -> ModelInfo | None:
    return MODELS.get(model_id)

def get_models_by_provider(provider: ModelProvider) -> list[ModelInfo]:
    model_ids = MODELS_BY_PROVIDER.get(provider, [])
    return [MODELS[model_id] for model_id in model_ids]
