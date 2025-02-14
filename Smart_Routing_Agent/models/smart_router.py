# Smart_Routing_Agent/models/smart_router.py

"""
The Smart Router works by evaluating each candidate model on two main dimensions—semantic similarity to the user’s prompt and task‑specific 
benchmark performance—then combining those scores to determine which models are most likely to produce a high‑quality response for that particular query. Here’s a detailed breakdown:

Embedding Model Descriptions:
When the router initializes, it loops over a curated list of candidate models (as specified in your configuration). For each model, 
it uses a SentenceTransformer (by default “all‑MiniLM‑L6‑v2”) to compute an embedding of the model’s description (which includes details about its strengths, weaknesses, and benchmark scores).

Normalizing Benchmark Data:
Each model in your registry has benchmark scores (like MMLU, GPQA, HumanEval, etc.) normalized on a 0–1 scale. 
These scores reflect each model’s performance on key capabilities (for example, GPT‑4 is very strong in complex reasoning and coding, while GPT‑3.5 might be better for fast, simple queries). The router precomputes these normalized benchmark scores for each model.

Task-Specific Weighting:
The router includes a task configuration that maps task types (e.g., “knowledge”, “coding”, “math”, “function”, “multilingual”, “general”) to:

Keyword lists: Used for automatic task detection (by matching query text).
Benchmark weights: These define the importance of each benchmark for a given task. For example, 
for coding tasks, the weight on “HumanEval” might be high, whereas for math, “MATH” gets more weight.
Automatic Task Detection:
Given a user query, if no task type is specified, the router converts the query to lowercase and checks if it contains any keywords 
from the pre‑defined task categories. If it finds a match (or detects math symbols), it sets the task type accordingly; otherwise, it falls back to “general.”

Semantic Scoring:
The router computes an embedding of the user’s query using the same SentenceTransformer. 
It then calculates the cosine similarity between this query embedding and each model’s precomputed description embedding. This similarity score reflects 
how semantically “aligned” the model’s documented capabilities are with the content of the query.

Benchmark Scoring:
Based on the detected task type, the router uses the corresponding benchmark weights to compute a benchmark score for each model. 
For each model, it takes the normalized benchmark scores for the benchmarks that are relevant to that task, multiplies each by its weight, and averages them.

Combining Scores:
The router then combines the semantic score and the benchmark score for each model into a final “combined score.” 
In the provided implementation, it uses a dynamic weighting approach (for example, using 40% weight for semantic similarity if there are enough benchmark entries, and 60% weight for benchmark performance) to produce an overall score for each model.

Ranking and Explanation:
After calculating the combined scores, the router sorts the candidate models in descending order (highest score first). It then returns a structured result that includes:

The detected task type.
A ranked list of models with their combined scores.
A breakdown of the semantic and benchmark scores.
A human‑readable explanation that describes the reasoning behind the top selection (e.g., summarizing the key strengths of the top model relative to the query).


"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import logging
from serverRouter.core.models import MODELS

logger = logging.getLogger(__name__)

class SmartRouter:
    """
    Advanced router combining semantic similarity and benchmark performance
    with automatic task type detection.
    """
    def __init__(self, config, embedding_model: str = "all-MiniLM-L6-v2"):
        self.config = config
        self.embedder = SentenceTransformer(embedding_model)
        self.model_ids = config.ensemble_model_names
        self.model_metadata = self._load_model_metadata()
        self.task_config = {
            "knowledge": {
                "weights": {"MMLU": 0.6, "GPQA": 0.4},
                "keywords": ["explain", "what is", "define", "describe", "theory"]
            },
            "coding": {
                "weights": {"HumanEval": 0.8, "MATH": 0.2},
                "keywords": ["code", "program", "algorithm", "debug", "function"]
            },
            "math": {
                "weights": {"MATH": 0.7, "GPQA": 0.3},
                "keywords": ["calculate", "solve", "equation", "math", "algebra"]
            },
            "function": {
                "weights": {"BFCL": 1.0},
                "keywords": ["api", "call", "function", "tool", "execute"]
            },
            "multilingual": {
                "weights": {"MGSM": 0.9, "MMLU": 0.1},
                "keywords": ["translate", "french", "spanish", "chinese", "german"]
            },
            "general": {
                "weights": {"MMLU": 0.3, "GPQA": 0.3, "HumanEval": 0.2, "MGSM": 0.2},
                "keywords": []
            }
        }

    def _load_model_metadata(self) -> Dict:
        metadata = {}
        for model_id in self.model_ids:
            model_info = MODELS.get(model_id)
            if not model_info:
                logger.warning(f"Model {model_id} not found in registry")
                continue
            embedding = self.embedder.encode(model_info.description)
            benchmarks = model_info.benchmarks.copy()
            max_scores = {
                "MMLU": 0.9,
                "GPQA": 0.72,
                "HumanEval": 0.95,
                "MATH": 0.91,
                "BFCL": 0.95,
                "MGSM": 0.93
            }
            for bench, score in benchmarks.items():
                benchmarks[bench] = score / max_scores.get(bench, 1.0)
            metadata[model_id] = {
                "embedding": embedding,
                "benchmarks": benchmarks,
                "provider": model_info.provider
            }
        return metadata

    def route(self, query: str, task_type: str = "auto") -> Dict:
        task_type = self._detect_task_type(query) if task_type == "auto" else task_type
        semantic_scores = self._calculate_semantic_scores(query)
        benchmark_scores = self._calculate_benchmark_scores(task_type)
        combined_scores = self._combine_scores(semantic_scores, benchmark_scores)
        sorted_models = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "task_type": task_type,
            "models": sorted_models,
            "scores": {
                "semantic": semantic_scores,
                "benchmark": benchmark_scores,
                "combined": combined_scores
            },
            "explanation": self._generate_explanation(query, task_type, sorted_models)
        }

    def _calculate_semantic_scores(self, query: str) -> Dict[str, float]:
        query_embedding = self.embedder.encode(query)
        scores = {}
        for model_id, data in self.model_metadata.items():
            similarity = np.dot(query_embedding, data["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(data["embedding"])
            )
            scores[model_id] = float(similarity)
        return scores

    def _calculate_benchmark_scores(self, task_type: str) -> Dict[str, float]:
        weights = self.task_config[task_type]["weights"]
        scores = {}
        for model_id, data in self.model_metadata.items():
            score = 0.0
            total_weight = 0.0
            for bench, weight in weights.items():
                if bench in data["benchmarks"]:
                    score += data["benchmarks"][bench] * weight
                    total_weight += weight
            scores[model_id] = score / total_weight if total_weight > 0 else 0.0
        return scores

    def _combine_scores(self, semantic: Dict, benchmark: Dict) -> Dict[str, float]:
        combined = {}
        for model_id in semantic.keys():
            sem_weight = 0.4 if len(self.model_metadata[model_id]["benchmarks"]) > 3 else 0.6
            bench_weight = 1 - sem_weight
            combined[model_id] = sem_weight * semantic[model_id] + bench_weight * benchmark[model_id]
        return combined

    def _detect_task_type(self, query: str) -> str:
        query = query.lower()
        for task, config in self.task_config.items():
            if task == "general":
                continue
            if any(kw in query for kw in config["keywords"]):
                return task
        if any(c in query for c in ['+', '-', '*', '/', '=', '^']):
            return "math"
        return "general"

    def _generate_explanation(self, query: str, task_type: str, models: List[Tuple]) -> str:
        top_model = models[0][0]
        model_info = MODELS.get(top_model)
        explanation = [
            f"Query: '{query}'",
            f"Detected task type: {task_type}",
            f"Top model selected: {top_model}",
            f"Reason: {model_info.description.split('. ')[0]}",
            f"Key benchmarks for this task:",
        ]
        for bench, weight in self.task_config[task_type]["weights"].items():
            explanation.append(f"- {bench}: {model_info.benchmarks.get(bench, 0):.1%} (weight: {weight*100}%)")
        return "\n".join(explanation)
