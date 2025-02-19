
# Smart_Routing_Agent/config/config.py

from serverRouter.core.models import ModelProvider

class Config:
    weak_model_name: str = "gpt-3.5-turbo"   # For simple queries
    strong_model_name: str = "gpt-4"           # For ensemble mode
    ranker_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    fuser_model_name: str = "gpt-4o-mini"
    router_length_threshold: int = 50  
    num_candidates: int = 5  
    top_k_fusion: int = 3    
    max_generation_tokens: int = 50
    device: str = "cuda"  # Change to "cpu" if necessary

    # New parameters for dynamic weighting in smart routing
    sim_weight: float = 0.4
    bench_weight: float = 0.6
    num_models_to_query: int = 3
    num_candidates_per_model: int = 5

    ensemble_model_names: list = [
        "gpt-4",                           # OpenAI's GPT-4
        "claude-3-5-sonnet",               # Anthropic Claude 3-5 Sonnet
        "gemini-2.0-pro",                  # Google Gemini 2.0 Pro
        "deepseek-chat",                   # DeepSeek Chat
        "t5-small",                        # Example open-source model
        "claude-3-opus-20240229",          # Another Anthropic model
        "deepseek-reasoner",               # DeepSeek Reasoner
        "gpt-4o",                          # OpenAI's GPT-4o
        "gemini-2.0-flash-thinking-exp-01-21",  # Another Gemini variant
        "gemini-2.0-pro-exp-02-05",       # Another Gemini variant
    ]
    
    mixinstruct_dataset_name: str = "llm-blender/mix-instruct"
    processed_data_dir: str = "Smart_Routing_Agent/data/processed"
