from serverRouter.core.datamodels import ModelInfo, ModelProvider

# Primary chat models registry
# Primary chat models registry with benchmark information embedded:
CHAT_MODELS = {

    "learnlm-1.5-pro-experimental": ModelInfo(
        name="learnlm-1.5-pro-experimental",
        provider=ModelProvider.GEMINI,
        description=(
            "LearnLM 1.5 Pro Experimental is a versatile multimodal model. "
            "Benchmarks are experimental; it is designed for integrating audio, images, video, and text."
        ),
        max_tokens=8192
    ),
    "gpt-4": ModelInfo(
        name="gpt-4",
        provider=ModelProvider.OPENAI,
        description=(
            "OpenAI's flagship model excelling in complex reasoning and coding. "
            "Strengths: Technical explanations, API integrations, and multi-step problem solving. "
            "Weaknesses: Higher cost, slower response times."
        ),
        max_tokens=8192,
        benchmarks={
            "MMLU": 0.864,
            "GPQA": 0.414,
            "HumanEval": 0.866,
            "MATH": 0.645,
            "BFCL": 0.883,
            "MGSM": 0.86
        }
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        description=(
            "GPT-3.5 (ChatGPT) is a versatile conversational model adept at general Q&A, basic reasoning, text completion, and casual dialogue. "
            "Strengths: Fast, cost-efficient, and well-suited for everyday tasks. "
            "Weaknesses: Weaker complex reasoning and limited knowledge compared to GPT-4."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.700,
            "GPQA": 0.308,
            "HumanEval": 0.680,
            "MATH": 0.341,
            "BFCL": 0.644,
            "MGSM": 0.563
        }
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        description=(
            "GPT-4o mini enables a broad range of tasks with its low cost and latency, such as applications that chain or parallelize multiple model calls (e.g., calling multiple APIs), pass a large volume of context to the model (e.g., full code base or conversation history), or interact with customers through fast, real-time text responses (e.g., customer support chatbots)."
            "GPT-4o mini surpasses GPT-3.5 Turbo and other small models on academic benchmarks across both textual intelligence and multimodal reasoning, and supports the same range of languages as GPT-4o."
            "It also demonstrates strong performance in function calling, which can enable developers to build applications that fetch data or take actions with external systems, and improved long-context performance compared to GPT-3.5 Turbo."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.820,
            "GPQA": 0.308,
            "HumanEval": 0.402,
            "MATH": 0.702,
            "BFCL": 0.641,
            "MGSM": 0.870
        }
    ),
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider=ModelProvider.OPENAI,
        description=(
            "GPT-4o strikes a balance between quality and efficiency, offering moderate benchmark performance."
            "It can respond to audio inputs in as little as 232 milliseconds, with an average of 320 milliseconds, which is similar to human response time⁠(opens in a new window) in a conversation."
            "It matches GPT-4 Turbo performance on text in English and code, with significant improvement on text in non-English languages, while also being much faster and 50% cheaper in the API."
            "GPT-4o is especially better at vision and audio understanding compared to existing models."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.887,
            "GPQA": 0.536,
            "HumanEval": 0.902,
            "MATH": 0.776,
            "BFCL": 0.805,
            "MGSM": 0.905
        }
    ),
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Anthropic's top performer for complex Q&A and multilingual tasks. "
            "Strengths: Research-grade answers, non-English queries, and precise tool usage. "
            "Weaknesses: Less creative compared to GPT-4."
        ),
        max_tokens=200000,
        benchmarks={
            "MMLU": 0.883,
            "GPQA": 0.594,
            "HumanEval": 0.920,
            "MATH": 0.711,
            "BFCL": 0.902,
            "MGSM": 0.916
        }
    ),
    "claude-3-opus": ModelInfo(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        description=(
            "Claude 3 Opus is designed for extremely complex tasks with an extended context. "
            "Strengths: High knowledge, advanced reasoning, and excellent multilingual performance. "
            "Weaknesses: May be less creative than GPT-4 and has higher latency."
        ),
        max_tokens=200000,
        benchmarks={
            "MMLU": 0.857,
            "GPQA": 0.504,
            "HumanEval": 0.849,
            "MATH": 0.601,
            "BFCL": 0.884,
            "MGSM": 0.907
        }
    ),
    "gemini-2.0-pro": ModelInfo(
        name="gemini-2.0-pro",
        provider=ModelProvider.GEMINI,
        description=(
            "Google's most advanced model for technical and scientific tasks. "
            "Strengths: STEM subjects, code generation, and multimodal reasoning. "
            "Weaknesses: Less conversational."
        ),
        max_tokens=1048576,
        benchmarks={
            "MMLU": 0.899,
            "GPQA": 0.624,
            "HumanEval": 0.929,
            "MATH": 0.897,
            "BFCL": 0.891,
            "MGSM": 0.887
        }
    ),
    "gemini-2.0-flash": ModelInfo(
        name="gemini-2.0-flash",
        provider=ModelProvider.GEMINI,
        description=(
            "Gemini 2.0 Flash is a lightweight, high-speed variant optimized for low latency. "
            "Strengths: Fast response and efficient for general queries. "
            "Weaknesses: May provide less detailed responses than Gemini Pro."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.800,  # estimated range
            "GPQA": 0.580,
            "HumanEval": 0.900,
            "MATH": 0.890,
            "BFCL": 0.870,
            "MGSM": 0.880
        }
    ),
    # Additional Gemini variants and experimental models:
    "gemini-2.0-pro-exp-02-05": ModelInfo(
        name="gemini-2.0-pro-exp-02-05",
        provider=ModelProvider.GEMINI,
        description=(
            "Google Gemini 2.0 Pro is a high-end model for extremely complex tasks. "
            "Expected benchmarks: MMLU in the high 80s, GPQA ~60%+, HumanEval ~90%+, Math ~90%, and excellent BFCL and MGSM performance."
        ),
        max_tokens=8192
    ),
    "gemini-2.0-flash-thinking-exp-01-21": ModelInfo(
        name="gemini-2.0-flash-thinking-exp-01-21",
        provider=ModelProvider.GEMINI,
        description=(
            "A variant of Gemini 2.0 Flash optimized for transparent, step-by-step reasoning. "
            "Expected to have similar benchmark performance to Gemini 2.0 Flash."
        ),
        max_tokens=8192
    ),
    "gemini-2.0-flash-exp": ModelInfo(
        name="gemini-2.0-flash-exp",
        provider=ModelProvider.GEMINI,
        description=(
            "Google Gemini 2.0 Flash (experimental) focuses on rapid, multimodal responses with benchmark performance similar to its Flash counterpart."
        ),
        max_tokens=8192
    ),
    "gemini-exp-1206": ModelInfo(
        name="gemini-exp-1206",
        provider=ModelProvider.GEMINI,
        description=(
            "Google Gemini (Quality improvements) emphasizes consistency and output quality. "
            "Benchmarks are expected to be on par with other Gemini 2.0 series models."
        ),
        max_tokens=8192
    ),
    "deepseek-chat": ModelInfo(
        name="deepseek-chat",
        provider=ModelProvider.DEEPSEEK,
        description=(
            "DeepSeek Chat is an open‑source conversational model known for cost‑effective, high‐quality dialogue. "
            "Strengths: Excellent general conversation and Q&A. "
            "Weaknesses: May lag behind in complex coding or reasoning tasks."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.880,
            "GPQA": 0.590,
            "HumanEval": 0.830,
            "MATH": 0.900,
            "BFCL": 0.0,    # Not available
            "MGSM": 0.850
        }
    ),
    "deepseek-reasoner": ModelInfo(
        name="deepseek-reasoner",
        provider=ModelProvider.DEEPSEEK,
        description=(
            "DeepSeek Reasoner (R1) is specialized for in-depth logical reasoning and multi-step problem solving. "
            "Strengths: Exceptional reasoning and math problem-solving; competitive with top-tier models on complex tasks. "
            "Weaknesses: May be slower and less polished in casual conversation."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.908,
            "GPQA": 0.715,
            "HumanEval": 0.830,
            "MATH": 0.840,
            "BFCL": 0.0,    # Not available
            "MGSM": 0.900
        }
    ),
    "llama-2-chat": ModelInfo(
        name="llama-2-chat",
        provider=ModelProvider.OPENAI,  # or a separate provider if self-hosted
        description=(
            "LLaMA-2 Chat is an open‑source conversational model optimized for on‑device deployments. "
            "Strengths: Efficient, customizable, and cost‑effective for simple interactions. "
            "Weaknesses: May require fine‑tuning for complex tasks and has a smaller context window."
        ),
        max_tokens=4096,
        benchmarks={
            "MMLU": 0.800,
            "GPQA": 0.500,
            "HumanEval": 0.750,
            "MATH": 0.550,
            "BFCL": 0.0,    # Not available
            "MGSM": 0.800
        }
    )
}

# Primary image models registry
IMAGE_MODELS = {
    """dall-e-3": ModelInfo(
        name="dall-e-3",
        provider=ModelProvider.OPENAI,
        description="OpenAI's most advanced image generation model"
    ),
    "dall-e-2": ModelInfo(
        name="dall-e-2",
        provider=ModelProvider.OPENAI,
        description="OpenAI's efficient image generation model",
    ),
    "imagen-3.0-generate-002": ModelInfo( #image generation
            name="imagegeneration@006",
            provider=ModelProvider.GEMINI,
            description="Google Imagen 3.0 for image generation",
    ),"""
    # New Stability AI models
    "stable-image-ultra": ModelInfo(
        name="stable-image-ultra",
        provider=ModelProvider.STABILITYAI,
        description="Stability AI Stable Image Ultra"
    ),
    "stable-image-core": ModelInfo(
        name="stable-image-core",
        provider=ModelProvider.STABILITYAI,
        description="Stability AI Stable Image Core"
    ),
     "sd3.5-large": ModelInfo(
        name="sd3.5-large",
        provider=ModelProvider.STABILITYAI,
        description="Stability AI Stable Diffusion 3.5 Large"
    ),
    "sd3.5-large-turbo": ModelInfo(
        name="sd3.5-large-turbo",
        provider=ModelProvider.STABILITYAI,
        description="Stability AI Stable Diffusion 3.5 Large Turbo"
    ),
    "sd3.5-medium": ModelInfo(
        name="sd3.5-medium",
        provider=ModelProvider.STABILITYAI,
        description="Stability AI Stable Diffusion 3.5 Medium"
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
