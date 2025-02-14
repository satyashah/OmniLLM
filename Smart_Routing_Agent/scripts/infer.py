# Smart_Routing_Agent/scripts/infer.py

import argparse
import os
import logging
from dotenv import load_dotenv
from typing import List
from Smart_Routing_Agent.config.config import Config
from Smart_Routing_Agent.models.smart_router import SmartRouter
from Smart_Routing_Agent.models.ranker import PairRanker
from Smart_Routing_Agent.models.fuser import Fuser
from serverRouter.core.models import get_model_by_id, ModelProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

def generate_candidates(model_id: str, prompt: str, num_candidates: int, max_tokens: int, device: str) -> List[str]:
    model_info = get_model_by_id(model_id)
    if not model_info:
        raise ValueError(f"Model {model_id} not found in registry")
    logger.info(f"Generating {num_candidates} candidates using {model_id}...")
    try:
        if model_info.provider == ModelProvider.OPENAI:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            responses = []
            for _ in range(num_candidates):
                response = client.chat.completions.create(
                    model=model_info.name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                responses.append(response.choices[0].message.content.strip())
            return responses
        else:
            raise NotImplementedError(f"Provider {model_info.provider} not supported in this demo")
    except Exception as e:
        logger.error(f"Generation failed for {model_id}: {str(e)}")
        return []

def main(args):
    config = Config()
    router = SmartRouter(config)
    ranker = PairRanker(config)
    fuser = Fuser(config)
    
    query = args.query or input("Enter your query: ")
    task_type = args.task
    decision = router.route(query, task_type)
    
    print("\n" + "="*50)
    print(decision["explanation"])
    print("\nDetailed Scores:")
    for model_id, score in decision["models"][:5]:
        print(f"{model_id}:")
        print(f"  Semantic: {decision['scores']['semantic'][model_id]:.3f}")
        print(f"  Benchmark: {decision['scores']['benchmark'][model_id]:.3f}")
        print(f"  Combined: {score:.3f}")
    print("="*50 + "\n")
    
    # For simplicity, query top N models where N = config.num_candidates_to_query (if defined) or top_k_fusion
    top_models = [model_id for model_id, _ in decision["models"][:config.top_k_fusion]]
    all_candidates = []
    
    for model_id in top_models:
        candidates = generate_candidates(
            model_id=model_id,
            prompt=query,
            num_candidates=config.num_candidates,
            max_tokens=config.max_generation_tokens,
            device=config.device
        )
        all_candidates.extend(candidates)
        logger.info(f"Generated {len(candidates)} candidates from {model_id}")
    
    if not all_candidates:
        raise RuntimeError("No candidates generated")
    
    ranked = ranker.rank(query, all_candidates)
    print("\nRanked Candidates (by score):")
    for cand, score in ranked:
        print(f"  Score: {score:.3f} | {cand}")
    
    top_candidates = [cand for cand, _ in ranked[:config.top_k_fusion]]
    fused_answer = fuser.fuse(query, top_candidates, max_new_tokens=config.max_generation_tokens)
    
    print("\n" + "="*50)
    print("Final Answer:")
    print(fused_answer)
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniLLM Smart Routing System")
    parser.add_argument("--query", type=str, help="Input query/prompt")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "knowledge", "coding", "math", "function", "multilingual", "general"],
                        help="Specify task type for routing")
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise
